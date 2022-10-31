"""Functions that handle saving and loading of checkpoints."""
import os
from pathlib import Path

import torch

import utils.distributed as du
import utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = Path(path_to_job) / "checkpoints"
    # Create the checkpoint dir from the master process
    if du.is_master_proc():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_dir(path_to_job) -> Path:
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return Path(path_to_job) / "checkpoints"


def get_path_to_checkpoint(path_to_job, epoch, task=""):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if task != "":
        name = "{}_checkpoint_epoch_{:05d}.pyth".format(task, epoch)
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return get_checkpoint_dir(path_to_job) / name


def get_last_checkpoint(path_to_job, task=""):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = [_.name for _ in d.iterdir()] if d.exists() else []
    if task != "":
        names = [f for f in names if "{}_checkpoint".format(task) in f]
    else:
        names = [f for f in names if f.startswith("checkpoint")]
    if len(names) == 0:
        return None
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return d / name


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = list(map(str, d.iterdir())) if d.exists() else []
    return any("checkpoint" in f for f in files)


def save_checkpoint(path_to_job, model, optimizer, lr_scheduler, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        lr_scheduler (StepLR): lr scheduler state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (GradScaler): the mixed precision scale.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.num_gpus * cfg.num_shards):
        return
    # Ensure that the checkpoint dir exists.
    get_checkpoint_dir(path_to_job).mkdir(parents=True, exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.distributed else model.state_dict()
    
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(
        path_to_job, epoch + 1,
    )
    with open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    lr_scheduler=None,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        lr_scheduler (StepLR): lr scheduler state.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    epoch = -1
    
    # Load the checkpoint on CPU to avoid GPU mem spike.
    with open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    miss_keys, unexpected_keys = ms.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if len(miss_keys) > 0:
        logger.warning(f"Model state dict has {len(miss_keys)} miss keys. There are {miss_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Model state dict has {len(unexpected_keys)} unexpected keys. May cause to ERROR. There are {miss_keys}")
    
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    if 'optimizer_state_dict' in checkpoint.keys() and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint.keys() and lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return epoch
