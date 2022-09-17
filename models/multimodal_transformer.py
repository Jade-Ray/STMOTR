import torch
import torch.nn as nn

from einops import rearrange, repeat

from .detr_helper import (TransformerEncoderLayer, TransformerEncoder,
                          TransformerDecoderLayer, TransformerDecoder)
from .position_encoding import PositionEmbeddingSine2D


class MultimodalTransformer(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3, **kwargs):
        super().__init__()
        self.d_model = kwargs['d_model']
        encoder_layer = TransformerEncoderLayer(**kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(**kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(self.d_model),
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
        
        memory = self.encoder(vid_embeds, src_key_padding_mask=seq_mask, pos=vid_pos_embed) # [S, T*B, C]
        vid_memory = rearrange(memory, '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
        
        # add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        obj_queries = repeat(obj_queries, 'n c -> n (t b) c', t=t, b=b)
        tgt = torch.zeros_like(obj_queries)  # [N, T*B, C]

        # hs is [L, N, T*B, C] where L is number of layers in the decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=seq_mask, pos=vid_pos_embed, query_pos=obj_queries)
        hs = rearrange(hs, 'l n (t b) c -> l t b n c', t=t, b=b)
        
        return hs, vid_memory
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
