import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from models.backbone import build as backbone_build
from models.matcher import build_matcher
from models.multimodal_transformer import MultimodalDeformableTransformer
from models.criterion import SetCriterion
from models.postprocessing import BasePostProcess
from utils.misc import NestedTensor, inverse_sigmoid


class DeformableMMOTR(nn.Module):
    def __init__(self, num_queries, start_level=1, extra_levels=1, aux_loss=False, **kwargs):
        super().__init__()
        self.backbone = backbone_build(**kwargs)
        input_num_channels = self.backbone.layer_output_channels
        self.start_level = start_level
        self.backbone_end_level = len(input_num_channels)
        self.feature_levels = self.backbone_end_level - self.start_level + extra_levels
        self.nheads = kwargs['nheads']
        self.npoints = kwargs['npoints']
        
        self.transformer = MultimodalDeformableTransformer(
            feature_levels=self.feature_levels, **kwargs)
        d_model = self.transformer.d_model
        
        self.is_referred_head = nn.Linear(d_model, 2)  # binary 'is referred?' prediction head for object queries
        self.box_head = MLP(d_model, d_model, 4, num_layers=2)
        self.query_embed = nn.Embedding(num_queries, d_model*2)
        self.backbone_proj = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            conv = nn.Sequential(
                nn.Conv2d(input_num_channels[i], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model))
            self.backbone_proj.append(conv)
            
        # add extra conv layers
        if extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = input_num_channels[self.backbone_end_level - 1]
                else:
                    in_channels = d_model
                extra_conv = nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model))
                self.backbone_proj.append(extra_conv)
        
        self.aux_loss = aux_loss
        self._reset_parameters()
    
    def _reset_parameters(self):
        for proj in self.backbone_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
    def forward(self, samples: list[NestedTensor]):
        backbone_outputs = self.backbone(samples) # l t b c h w
        
        vid_embeds, vid_pad_mask = [], []
        for level in range(self.feature_levels):
            if level+self.start_level < self.backbone_end_level:
                src, mask = backbone_outputs[level+self.start_level].decompose()
            elif level+self.start_level == self.backbone_end_level:
                src, mask = backbone_outputs[-1].decompose()
            T, B, _, _, _  = src.shape
            src = rearrange(src, 't b c h w -> (t b) c h w')
            src = self.backbone_proj[level](src)
            src = rearrange(src, '(t b) c h w -> t b c h w', t=T, b=B)
            mask = F.interpolate(mask.float(), size=src.shape[-2:]).to(torch.bool)
            vid_embeds.append(src)
            vid_pad_mask.append(mask)
        
        transformer_out = self.transformer(vid_embeds, vid_pad_mask, self.query_embed.weight)
        # hs is: [L, T, B, N, D] where L is number of decoder layers
        # vid_memory is lev list: [(T, B), (H, W), C]
        # reference_points is [L, T, B, N, 2]
        hs, vid_memory, inter_references = transformer_out
        
        outputs_is_referred = self.is_referred_head(hs)  # [L, T, B, N, 2]
        outputs_coords = self.box_head(hs) # [L, T, B, N, 4]
        
        layer_outputs = []
        for i in range(len(hs)):
            pb, pir = outputs_coords[i], outputs_is_referred[i]
            # outputs_coord is formmed (cx, cy, w, h)
            reference = inverse_sigmoid(inter_references[i])
            assert reference.shape[-1] == 2
            pb[..., :2] += reference
            layer_out = {'pred_boxes': pb.sigmoid(),
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
    model = DeformableMMOTR(**vars(args))
    matcher = build_matcher(args, referred_focal_loss=True)
    weight_dict = {'loss_is_referred': args.is_referred_loss_coef,
                   'loss_boxes': args.boxes_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, alpha=args.alpha, gamma=args.gamma,
        referred_loss_type='sigmoid_focal_loss')
    if args.dataset_name == 'tunnel':
        postprocessor = BasePostProcess(referred_sigmoid=True)
    elif args.dataset_name == 'ua':
        postprocessor = BasePostProcess(referred_sigmoid=True)
    elif args.dataset_name == 'mot20':
        postprocessor = BasePostProcess(referred_sigmoid=True)
    else:
        assert False, f'postprocessing for dataset: {args.dataset_name} is not supported'
    return model, criterion, postprocessor
