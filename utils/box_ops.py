"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import Union
import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if isinstance(x, Tensor):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate((x[..., 0:2] - 0.5*x[..., 2:4], x[..., 0:2] + 0.5*x[..., 2:4]))
    else:
        raise TypeError('Argument cxxywh must be a Tensor or numpy array.')


def box_xyxy_to_cxcywh(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if isinstance(x, Tensor):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate(((x[..., 0:2] + x[..., 2:4])/2, x[..., 2:4] - x[..., 0:2]), axis=-1)
    else:
        raise TypeError('Argument xyxy must be a Tensor or numpy array.')


def box_xyxy_to_xywh(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if isinstance(x, Tensor):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [x0, y0, (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate((x[..., 0:2], x[..., 2:4] - x[..., 0:2]), axis=-1)
    else:
        raise TypeError('Argument xyxy must be a Tensor or numpy array.')


def box_xywh_to_xyxy(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if isinstance(x, Tensor):
        x_lt, y_lt, w, h = x.unbind(-1)
        b = [x_lt, y_lt, x_lt + w, y_lt + h]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate((x[..., 0:2], x[..., 0:2] + x[..., 2:4]), axis=-1)
    else:
        raise TypeError('Argument xywh must be a Tensor or numpy array.')
    

def box_xyxy_to_botcenter(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    if isinstance(x, Tensor):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [x0 + (x1 -x0) / 2, y1]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        return np.concatenate((x[..., 0] + (x[..., 2] - x[..., 0]) / 2, x[..., 3]), axis=-1)
    else:
        raise TypeError('Argument xyxy must be a Tensor or numpy array.')


def box_nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """
    Non-max Suppression Algorithm. Return after NMS index
    
    Args:
        bboxes (np.ndarray): The predicted bounding box (x0, y0, x1, y1).
        socres (np.ndarray): The predicted bounding box confidence scores.
        iou_thresh (float): The iou threshold.
    """
    indexes = []
    if bboxes.size == 0:
        return indexes
    
    # coordinates of bounding boxes
    start_x = bboxes[:, 0]
    start_y = bboxes[:, 1]
    end_x = bboxes[:, 2]
    end_y = bboxes[:, 3]
    
    # Compute areas of bounding boxes
    areas = (end_x - start_x) * (end_y - start_y)
    
    # Sort by confidence score of bounding boxes
    order = np.argsort(scores)
    
    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        indexes.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < iou_thresh)
        order = order[left]

    return indexes


def box_iou(boxes1: Union[Tensor, np.ndarray], boxes2: Union[Tensor, np.ndarray],
            first_union: bool=False) -> Tensor:
    """
    Return [N, M] IoU distance matrix between two boxes, with [x0, y0, x1, y1] format
    
        IoU(a,b) = isect(a, b) / union(a, b)
    
    If set first_union, custom IoU just compute first boxes area as union
    
        IoU(a,b) = isect(a, b) / union(a)
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.tensor(boxes1)
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.tensor(boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]
    
    if first_union:
        union = area1[:, None]
    else:
        union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


# modified from torchvision to also return the union
def box_iou_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return Iou and Union area matrix
    
    The boxes should be [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor):
    """
    Generalized IoU from https://giou.stanford.edu/
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou_union(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return  iou - (area - union) / area