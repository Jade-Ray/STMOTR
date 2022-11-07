import datetime

import torch
import torch.distributed as dist


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=100),
        world_size=world_size,
        rank=proc_rank,
    )
    dist.barrier(device_ids=[local_rank])


def is_master_proc(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def is_root_proc():
    """
    Determines if the current process is the root process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def init_distributed_training(cfg):
    """
    Initialize variables needed for distributed training.
    """
    if cfg.num_gpus <= 1:
        return
    num_gpus_per_machine = cfg.num_gpus
    num_machines = dist.get_world_size() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == cfg.shard_id:
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_dict(input_dict, group=None):
    """
    Args:
        input_dict (dict): all the values will be gather
        group (ProcessGroup, optional): The process group to work on. If None, the default process group will be used. Defaults to None.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(
            outputs,
            input_dict,
            group=group
        )
        
        output_dict = {}
        for output in outputs:
            output_dict.update(output)
    return output_dict
