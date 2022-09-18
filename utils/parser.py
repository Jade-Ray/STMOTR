#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Argument parser functions."""

import argparse
import yaml
import sys
import torch

import utils.checkpoint as cu


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide MMOTR training and testing pipeline."
    )
    parser.add_argument(
        "--running_mode", 
        choices=['train', 'eval'], 
        required=True,
        help="mode to run, either 'train' or 'eval'"
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_gpus", 
        help='number of CUDA gpus to run on.',
        default=1,
        type=int, 
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg", 
        dest="cfg_files", 
        help="Path to the config files", 
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See config/*.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    cfg = {}
    # Load config from cfg.
    if path_to_config is not None:
        with open(path_to_config, 'r') as stream:
            config = yaml.safe_load(stream)
        config = {k: v['value'] for k, v in config.items()}
        cfg = {**cfg, **config}
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        for key, value in zip(args.opts[0::2], args.opts[1::2]):
            if hasattr(cfg, key):
                if isinstance(cfg[key], int):
                    cfg[key] = int(value)
                elif isinstance(cfg[key], float):
                    cfg[key] = float(value)
                else:
                    cfg[key] = value
    cfg = argparse.Namespace(**cfg)

    # Inherit parameters from args.
    if hasattr(args, "num_gpus"):
        cfg.num_gpus = max(min(args.num_gpus, torch.cuda.device_count()), 1)
    if hasattr(args, "num_shards") and hasattr(args, "shard_id") and hasattr(args, "init_method"):
        cfg.num_shards = args.num_shards
        cfg.shard_id = args.shard_id
        cfg.init_method = args.init_method
    if hasattr(args, "rng_seed"):
        cfg.rng_seed = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.output_dir = args.output_dir
        
    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.output_dir)
    return cfg
