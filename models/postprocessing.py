import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from utils.box_ops import box_cxcywh_to_xyxy


class BasePostProcess(nn.Module):
    def __init__(self):
        super(BasePostProcess, self).__init__()
    
    @torch.inference_mode()
    def forward(self, outputs, orig_sample_sizes):
        """ Perform the computation
        
        Args:
            outputs: raw outputs of the model.
            orig_sample_sizes: original size [batch_size, 2] of the samples (no augmentations or padding).
        """
        pred_is_referred = outputs['pred_is_referred'] # [T, B, N, 2]
        prob = rearrange(pred_is_referred, 't b n c -> b n t c').softmax(-1) # [B, N, T, 2]
        scores = prob[..., 1] # [B, N, T]
        
        pred_boxes = outputs["pred_boxes"] # [T, B, N, 4]
        pred_boxes = rearrange(pred_boxes, 't b n c -> b n t c') # [B, N, T, 4]
        
        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(pred_boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = orig_sample_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1) # [batch_size, 4]
        boxes = boxes * scale_fct[:, None, None]
        
        predictions = [{'scores': s, 'boxes': b} for s, b in zip(scores, boxes)]

        return predictions

