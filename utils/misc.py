import math
import numpy as np
import os
from datetime import datetime
import psutil
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

import utils.logging as logging

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def check_finite_losses(loss):
    """
    Determine whether the loss is infinity or NaN (not a number).
    Args:
        loss (loss): loss to check whether is infinity or NaN.
    """
    if not math.isfinite(loss):
        raise RuntimeError("ERROR: Got infinity or NaN losses {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.LayerNorm)):
                for p in m.parameters(recurse=False):
                    if p.requires_grad:
                        count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def log_model_info(model):
    """
    Log info, includes number of parameters, gpu usage.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_period_epoch(cur_epoch, max_epoch, period) -> bool:
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
        max_epoch (int): max number of epoch of the model.
        period (int): the period of epoch.
    """
    if cur_epoch + 1 == max_epoch:
        return True
    return (cur_epoch + 1) % period == 0


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def cuda(self, device=None, non_blocking=False):
        cast_tensor = self.tensors.cuda(device, non_blocking)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.cuda(device, non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def to(self, device=None, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_ori_mask(tensor: Tensor, ori_mask: Tensor) -> NestedTensor:
    """
    Repadding mask with new tensor from original mask.
    """
    pad_mask = F.interpolate(ori_mask.float(), size=tensor.shape[-2:]).to(torch.bool)
    return NestedTensor(tensor, pad_mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    This function receives a list of image tensors and returns a NestedTensor of the padded images, along with their
    padding masks (true for padding areas, false otherwise).
    """
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    return NestedTensor(tensor, mask)


def nested_tensor_from_videos_list(videos_list: List[Tensor]):
    """
    This function receives a list of videos (each of shape [T, C, H, W]) and returns a NestedTensor of the padded
    videos (shape [T, B, C, PH, PW], along with their padding masks (true for padding areas, false otherwise, of shape
    [T, B, PH, PW].
    """
    max_size = _max_by_axis([list(img.shape) for img in videos_list])
    padded_batch_shape = [len(videos_list)] + max_size
    b, t, c, h, w = padded_batch_shape
    dtype = videos_list[0].dtype
    device = videos_list[0].device
    padded_videos = torch.zeros(padded_batch_shape, dtype=dtype, device=device)
    videos_pad_masks = torch.ones((b, t, h, w), dtype=torch.bool, device=device)
    for vid_frames, pad_vid_frames, vid_pad_m in zip(videos_list, padded_videos, videos_pad_masks):
        pad_vid_frames[:vid_frames.shape[0], :, :vid_frames.shape[2], :vid_frames.shape[3]].copy_(vid_frames)
        vid_pad_m[:vid_frames.shape[0], :vid_frames.shape[2], :vid_frames.shape[3]] = False
    # transpose the temporal and batch dims and create a NestedTensor:
    return NestedTensor(padded_videos.transpose(0, 1), videos_pad_masks.transpose(0, 1))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
