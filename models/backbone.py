import torch
import torch.nn as nn
from einops import rearrange

from .swin_transformer import SwinTransformer3D
from utils.misc import NestedTensor, nested_tensor_from_ori_mask


class VideoSwinTransformerBackbone(nn.Module):
    def __init__(self, backbone_pretrained, backbone_pretrained_path, train_backbone, **kwargs):
        super(VideoSwinTransformerBackbone, self).__init__()
        # default Swin-T
        swin_backbone = SwinTransformer3D(
            patch_size=(1, 4, 4), embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), 
            window_size=(8, 7, 7), drop_path_rate=0.1, patch_norm=True)
        if backbone_pretrained:
            state_dict = torch.load(backbone_pretrained_path, map_location='cpu')['state_dict']
            # extract swinT's kinetics-400 pretrained weights and ignore the rest (prediction head etc.)
            state_dict = {k[9:]: v for k, v in state_dict.items() if 'backbone.' in k}
            
            # sum over the patch embedding weight temporal dim  [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
            patch_embed_weight = state_dict['patch_embed.proj.weight']
            patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
            state_dict['patch_embed.proj.weight'] = patch_embed_weight
            swin_backbone.load_state_dict(state_dict)

        self.patch_embed = swin_backbone.patch_embed
        self.pos_drop = swin_backbone.pos_drop
        self.layers = swin_backbone.layers[:-1]
        self.downsamples = nn.ModuleList()
        for layer in self.layers:
            self.downsamples.append(layer.downsample)
            layer.downsample = None
        self.downsamples[-1] = None # downsampling after the last layer is not necessary
        
        self.layer_output_channels = [swin_backbone.embed_dim * 2 ** i for i in range(len(self.layers))]
        self.train_backbone = train_backbone
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
    
    def forward(self, samples: NestedTensor):
        vid_frames = rearrange(samples.tensors, 't b c h w -> b c t h w')
        
        vid_embeds = self.patch_embed(vid_frames)
        vid_embeds = self.pos_drop(vid_embeds)
        
        outputs = []
        for layer, downsample in zip(self.layers, self.downsamples):
            vid_embeds = layer(vid_embeds.contiguous())
            output = rearrange(vid_embeds, 'b c t h w -> t b c h w')
            outputs.append(nested_tensor_from_ori_mask(output, samples.mask))
            if downsample:
                vid_embeds = rearrange(vid_embeds, 'b c t h w -> b t h w c')
                vid_embeds = downsample(vid_embeds)
                vid_embeds = rearrange(vid_embeds, 'b t h w c -> b c t h w')
        return outputs
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(backbone_name, **kwargs):
    if backbone_name == 'swin-t':
        return VideoSwinTransformerBackbone(**kwargs)
    assert False, f'error: backbone "{backbone_name}" is not supported'
