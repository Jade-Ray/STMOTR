import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from models.backbone import build as backbone_build
from models.matcher import build_matcher
from models.multimodal_transformer import MultimodalTransformer
from models.criterion import SetCriterion
from models.postprocessing import BasePostProcess
from utils.misc import NestedTensor


class MMOTR(nn.Module):
    def __init__(self, num_queries, aux_loss=False, **kwargs):
        super().__init__()
        self.backbone = backbone_build(**kwargs)
        self.transformer = MultimodalTransformer(**kwargs)
        
        d_model = self.transformer.d_model
        self.is_referred_head = nn.Linear(d_model, 2)  # binary 'is referred?' prediction head for object queries
        self.box_head = MLP(d_model, d_model, 4, num_layers=2)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.backbone_proj = nn.Conv2d(self.backbone.layer_output_channels[-1], d_model, kernel_size=1)
        self.aux_loss = aux_loss
        
    def forward(self, samples: NestedTensor):
        backbone_output = self.backbone(samples) # t b c h w
        vid_embeds, vid_pad_mask = backbone_output.decompose()
        
        T, B, _, _, _ = vid_embeds.shape
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (t b) c h w')
        vid_embeds = self.backbone_proj(vid_embeds)
        vid_embeds = rearrange(vid_embeds, '(t b) c h w -> t b c h w', t=T, b=B)
        
        transformer_out = self.transformer(vid_embeds, vid_pad_mask, self.query_embed.weight)
        # hs is: [L, T, B, N, D] where L is number of decoder layers
        # vid_memory is: [T, B, D, H, W]
        # encoder_middle_layer_outputs is a list of [T, B, H, W, D]
        hs, vid_memory = transformer_out
        
        outputs_is_referred = self.is_referred_head(hs)  # [L, T, B, N, 2]
        outputs_coords = self.box_head(hs) # [L, T, B, N, 4]
        outputs_coords = outputs_coords.sigmoid()
        
        layer_outputs = []
        for pb, pir in zip(outputs_coords, outputs_is_referred):
            layer_out = {'pred_boxes': pb,
                         'pred_is_referred': pir}
            layer_outputs.append(layer_out)
        out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        if self.aux_loss:
            out['aux_outputs'] = layer_outputs[:-1]
        return out
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    model = MMOTR(**vars(args))
    matcher = build_matcher(args)
    weight_dict = {'loss_is_referred': args.is_referred_loss_coef,
                   'loss_boxes': args.boxes_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef)
    if args.dataset_name == 'tunnel':
        postprocessor = BasePostProcess()
    elif args.dataset_name == 'ua':
        postprocessor = BasePostProcess()
    else:
        assert False, f'postprocessing for dataset: {args.dataset_name} is not supported'
    return model, criterion, postprocessor
