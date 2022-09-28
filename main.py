import torch
import torch.distributed as dist

from trainer import Trainer
from utils.parser import parse_args, load_config
import utils.distributed as du
import utils.logging as logging

logger = logging.get_logger(__name__)


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
    elif running_mode == 'ablation':
        trainer.visualization(vis_ablation=False)
    else:  # eval mode:
        logger.info(f"Only Eval")
        trainer.eval_epoch(trainer.epochs)
        trainer.clear_memory()
        if cfg.distributed:
            dist.barrier()

        if trainer.writer is not None:
            trainer.writer.close()
        logger.info(f"Eval Done")


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
