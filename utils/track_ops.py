"""
Utilities for track of bounding box manipulation and GIoU.
"""
import torch

area = lambda track: (track[..., 2] - track[..., 0]) * (track[..., 3] - track[..., 1])
wh_ratio = lambda track: (track[..., 2] - track[..., 0]) / (track[..., 3] - track[..., 1])


def track_distance(tracks1, tracks2, tgt_vis_mask, mode='mean'):
    """
    Euler distance of bbox of tracks, and the format is [num, T, 4]
    
    tgt_vis_mask is mask of whether bbox referred in every T [num, T]
    
    mode is method of evaluate all bbox giou in each track, supported mean and max
    
    Returns a [N, M] pairwise matrix, where N = len(tracks1) and M = len(tracks2)
    """
    assert tgt_vis_mask is not None
    diff = torch.sum(torch.abs(tracks1[:, None, ...] - tracks2), -1) * tgt_vis_mask # [N, M, T]
    
    if mode == 'mean':
        return torch.sum(diff, -1) / torch.sum(tgt_vis_mask != 0, -1).clamp(min=1)
    elif mode == 'min':
        return torch.min(diff, -1)[0]
    else:
        raise ValueError(f'not supported mode {mode}')


def track_iou(tracks1, tracks2):   
    area1 = area(tracks1) # [N, T]
    area2 = area(tracks2) # [M, T]
    
    lt = torch.max(tracks1[:, None, ..., :2], tracks2[..., :2])  # [N, M, T, 2]
    rb = torch.min(tracks1[:, None, ..., 2:], tracks2[..., 2:]) # [N, M, T, 2]
    
    wh = (rb - lt).clamp(min=0) # [N, M, T, 2]
    inter = wh[..., 0] * wh[..., 1] # [N, M, T]
    
    union = area1[:, None] + area2 - inter # [N, M, T]
    
    iou = inter / union # [N, M, T]
    
    return iou, union


def generalized_track_iou(tracks1, tracks2, tgt_vis_mask, mode='mean'):
    """
    Generalized IoU of bbox of tracks, and the format is [num, T, 4]
    
    tgt_vis_mask is mask of whether bbox referred in every T [num, T]
    
    mode is method of evaluate all bbox giou in each track, supported mean and max
    
    Returns a [N, M] pairwise matrix, where N = len(tracks1) and M = len(tracks2)
    """
    assert (tracks1[..., 2:] >= tracks1[..., :2]).all()
    assert (tracks2[..., 2:] >= tracks2[..., :2]).all()
    assert tgt_vis_mask is not None
    iou, union = track_iou(tracks1, tracks2)
    
    lt = torch.min(tracks1[:, None, ..., :2], tracks2[..., :2])
    rb = torch.max(tracks1[:, None, ..., 2:], tracks2[..., 2:])
    
    wh = (rb - lt).clamp(min=0) # [N, M, T, 2]
    area = wh[..., 0] * wh[..., 1] # [N, M, T]
    
    giou = (iou - (area - union) / area) * tgt_vis_mask    
    
    if mode == 'mean':
        return torch.sum(giou, -1) / torch.sum(tgt_vis_mask != 0, -1).clamp(min=1)
    elif mode == 'max':
        return torch.max(giou, -1)[0]
    else:
        raise ValueError(f'not supported mode {mode}')
