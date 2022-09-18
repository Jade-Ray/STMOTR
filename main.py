import torch

from trainer import Trainer
from utils.parser import parse_args, load_config
import utils.distributed as du


def run(process_id, cfg, running_mode):
    if cfg.num_gpus > 1:
        cfg.distributed = True
        # Initialize the process group.
        du.init_process_group(
            local_rank=process_id, 
            local_world_size=cfg.num_gpus, 
            shard_id=cfg.shard_id, 
            num_shards=cfg.num_shards, 
            init_method=cfg.init_method)
    else:
        cfg.distributed = False
    
    trainer = Trainer(cfg)
    if running_mode == 'train':
        trainer.train()
    else:  # eval mode:
        trainer.evaluate()


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        
        if cfg.num_gpus > 1:
            torch.multiprocessing.spawn(
                run, 
                nprocs=cfg.num_gpus, 
                args=(cfg, args.running_mode))
        else:  # run on a single GPU or CPU
            run(process_id=0, cfg=cfg, 
                running_mode=args.running_mode)


if __name__ == '__main__':
    main()
