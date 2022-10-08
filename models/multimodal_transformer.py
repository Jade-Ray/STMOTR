import torch
import torch.nn as nn

from einops import rearrange, repeat

from .detr_helper import (TransformerEncoderLayer, TransformerEncoder,
                          TransformerDecoderLayer, TransformerDecoder,
                          DeformableTransformerEncoderLayer, DeformableTransformerEncoder,
                          DeformableTransformerDecoderLayer, DeformableTransformerDecoder,
                          MSDeformAttn)
from .position_encoding import PositionEmbeddingSine2D


class MultimodalTransformer(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3, **kwargs):
        super().__init__()
        self.d_model = kwargs['d_model']
        encoder_layer = TransformerEncoderLayer(**kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(**kwargs)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, norm=nn.LayerNorm(self.d_model),
            return_intermediate=True)
        self.pos_encoder_2d = PositionEmbeddingSine2D()
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, vid_embeds, vid_pad_mask, obj_queries):
        t, b, _, h, w = vid_embeds.shape
        
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c')
        seq_mask = rearrange(vid_pad_mask, 't b h w -> (t b) (h w)')
        vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
        pos_embed = rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c')
        
        memory = self.encoder(vid_embeds, src_key_padding_mask=seq_mask, pos=pos_embed) # [S, T*B, C]
        vid_memory = rearrange(memory, '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
        
        # add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        obj_queries = repeat(obj_queries, 'n c -> n (t b) c', t=t, b=b)
        tgt = torch.zeros_like(obj_queries)  # [N, T*B, C]

        # hs is [L, N, T*B, C] where L is number of layers in the decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=seq_mask, pos=pos_embed, query_pos=obj_queries)
        hs = rearrange(hs, 'l n (t b) c -> l t b n c', t=t, b=b)
        
        return hs, vid_memory
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class MultimodalDeformableTransformer(nn.Module):
    def __init__(self, feature_levels=4, num_encoder_layers=3, num_decoder_layers=3, **kwargs):
        super().__init__()
        self.d_model = kwargs['d_model']
        encoder_layer = DeformableTransformerEncoderLayer(nlevels=feature_levels, **kwargs)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(nlevels=feature_levels, **kwargs)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate=True)
        self.pos_encoder_2d = PositionEmbeddingSine2D()
        self.level_embed = nn.Parameter(torch.Tensor(feature_levels, self.d_model))
        self.reference_points = nn.Linear(self.d_model, 2)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.)
        nn.init.normal_(self.level_embed)
    
    def get_valid_ratio(self, mask):
        T, _, H, W = mask.shape
        valid_H = torch.sum(~mask[0, :, :, 0], 1) # [bs]
        valid_W = torch.sum(~mask[0, :, 0, :], 1) # [bs]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_h, valid_ratio_w], -1) # [bs, 2]
        return valid_ratio
    
    def forward(self, vid_embeds, vid_pad_mask, obj_queries):
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, valid_ratios = [], [], [], [], []
        for src, mask, lvl_emb in zip(vid_embeds, vid_pad_mask, self.level_embed):
            t, b, _, h, w = src.shape
            spatial_shapes.append((h, w))
            valid_ratios.append(self.get_valid_ratio(mask))
            src_flatten.append(rearrange(src, 't b c h w -> (t b) (h w) c'))
            mask_flatten.append(rearrange(mask, 't b h w -> (t b) (h w)'))
            vid_pos_embed = self.pos_encoder_2d(rearrange(mask, 't b h w -> (t b) h w'), self.d_model)
            lvl_pos_embed_flatten.append(
                rearrange(vid_pos_embed, 'tb h w c -> tb (h w) c') + lvl_emb.view(1, 1, -1)) # (t b) (h w) c
            
        src_flatten = torch.cat(src_flatten, 1) # (t b) (l h w) c
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) 
        
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,  )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [lvl]
        valid_ratios = repeat(torch.stack(valid_ratios, 1), 'b l c -> (t b) l c', t=t)
        
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten) # [T*B, lvl*h*w, C]
        vid_memory = memory.split(list(spatial_shapes.prod(1).cpu().numpy()), dim=1)
        
        # prepare input for encoder
        # add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        query_embed, tgt = torch.split(obj_queries, self.d_model, dim=1) # [n_q, c]
        query_embed = repeat(query_embed, 'n c -> (t b) n c', t=t, b=b)
        tgt = repeat(tgt, 'n c -> (t b) n c', t=t, b=b)
        reference_points = self.reference_points(query_embed).sigmoid()
        
        # decoder
        # hs is [L, T*B, N, C] where L is number of layers in the decoder
        # reference_points is [L, T*B, N, 2]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_embed, mask_flatten)

        hs = rearrange(hs, 'l (t b) n c -> l t b n c', t=t, b=b)
        inter_references = rearrange(inter_references, 'l (t b) n c -> l t b n c', t=t, b=b)
        
        return hs, vid_memory, inter_references
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
