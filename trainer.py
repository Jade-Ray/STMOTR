"""
This file contains a Trainer class which handles the training and evaluation of MMOTR.
"""
import pprint
import gc

import numpy as np
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR

from datasets import build_dataset, get_parser_data_from_dataset
from models import build_model
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
import visualization.tensorboard_vis as tb
import visualization.utils as vis_utils
from utils.meters import EpochTimer, TrainMeter, MotValMeter

logger = logging.get_logger(__name__)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epochs = cfg.epochs
        self.log_period = cfg.log_period
        self.eval_period = cfg.eval_period
        self.save_period = cfg.save_period
        self.output_dir = cfg.output_dir
        
        # Set up environment.
        du.init_distributed_training(cfg)
        # Set random seed from configs.
        np.random.seed(cfg.rng_seed)
        torch.manual_seed(cfg.rng_seed)
        self.is_master_proc = du.is_master_proc(cfg.num_gpus * cfg.num_shards)
        
        # Setup logging format.
        logging.setup_logging(cfg.output_dir)
        
        # Print config.
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

        model, criterion, postprocessor = build_model(cfg)
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        model.cuda(device=cur_device)
        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_device],
                output_device=cur_device,
                find_unused_parameters=False,)
        if self.is_master_proc:
            misc.log_model_info(model)
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor
        
        # Optimizer, LR-Scheduler, AMP Grad Scaler:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": cfg.lr_backbone},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[50], gamma=0.4, verbose=True)
        self.grad_scaler = amp.GradScaler(enabled=cfg.enable_amp)
        self.max_norm = cfg.clip_max_norm
        
        # Load a checkpoint to resume training if applicable.
        if cfg.auto_resume and cu.has_checkpoint(cfg.output_dir):
            logger.info("Load from last checkpoint.")
            last_checkpoint = cu.get_last_checkpoint(cfg.output_dir)
            if last_checkpoint is not None:
                checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                self.model,
                cfg.distributed,
                self.optimizer,
                self.lr_scheduler,)
                self.epoch = checkpoint_epoch + 1
            else:
                self.epoch = 0
        elif cfg.resume != "":
            logger.info("Load from given checkpoint file.")
            checkpoint_epoch = cu.load_checkpoint(
                cfg.resume,
                self.model,
                cfg.distributed,
                self.optimizer,
                self.lr_scheduler,)
            self.epoch = checkpoint_epoch + 1
        else:
            self.epoch = 0
        
        # Create the video train and val loaders.
        dataset_train = build_dataset(image_set='train', **vars(cfg))
        dataset_val = build_dataset(image_set='test', **vars(cfg))
        if cfg.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        self.data_loader_train = DataLoader(
            dataset_train, batch_size=cfg.batch_size, 
            sampler=sampler_train,
            collate_fn=dataset_train.collator, 
            num_workers=cfg.num_workers,
            pin_memory=True)
        self.data_loader_val = DataLoader(
            dataset_val, batch_size=cfg.eval_batch_size, 
            sampler=sampler_val,
            collate_fn=dataset_val.collator, 
            num_workers=cfg.num_workers,
            pin_memory=True)
        
        # Create meters.
        self.train_meter = TrainMeter(
            len(self.data_loader_train), self.epochs, 
            self.log_period, cfg.output_dir)
        self.val_meter = MotValMeter(
            get_parser_data_from_dataset(dataset_val), 
            self.epochs, cfg.output_dir, cfg.referred_threshold)
        
        # set up writer for logging to Tensorboard format.
        if cfg.board_enable and self.is_master_proc:
            self.writer = tb.TensorboardWriter(cfg.board_dir)
        else:
            self.writer = None

    def train(self):
        # Perform the training loop.
        logger.info(f"Start epoch: {(self.epoch + 1)}")
        
        epoch_timer = EpochTimer()
        for cur_epoch in range(self.epoch, self.epochs):
            # Shuffle the dataset.
            if self.cfg.distributed:
                self.data_loader_train.sampler.set_epoch(cur_epoch)
            # Train for one epoch.
            epoch_timer.epoch_tic()
            self.train_epoch(cur_epoch)
            self.lr_scheduler.step()
            epoch_timer.epoch_toc()
            logger.info(
                f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
                f"from {self.epoch} to {cur_epoch} take "
                f"{epoch_timer.avg_epoch_time():.2f}s in average and "
                f"{epoch_timer.median_epoch_time():.2f}s in median."
            )
            logger.info(
                f"For epoch {cur_epoch}, each iteraction takes "
                f"{epoch_timer.last_epoch_time()/len(self.data_loader_train):.2f}s in average. "
                f"From epoch {self.epoch} to {cur_epoch}, each iteraction takes "
                f"{epoch_timer.avg_epoch_time()/len(self.data_loader_train):.2f}s in average."
            )
            
            # Save a checkpoint.
            if misc.is_period_epoch(cur_epoch, self.epochs, self.save_period):
                cu.save_checkpoint(self.output_dir, self.model, self.optimizer, self.lr_scheduler, cur_epoch, self.cfg)
            # Evaluate the model on validation set.
            if misc.is_period_epoch(cur_epoch, self.epochs, self.eval_period):
                self.eval_epoch(cur_epoch)
                
            # run gc collection before starting a new epoch to avoid possible OOM errors due to swinT caching :
            self.clear_memory()
            if self.cfg.distributed:
                dist.barrier()

        if self.writer is not None:
            self.writer.close()
        logger.info(f"testing done:\n {str(self.val_meter)}")
        
        return str(self.val_meter)

    def train_epoch(self, cur_epoch: int):
        # Enable train mode.
        self.model.train()
        self.criterion.train()
        self.train_meter.iter_tic()
        data_size = len(self.data_loader_train)
        
        for cur_iter, (samples, targets) in enumerate(self.data_loader_train):
            # Transfer the data to the current GPU device.
            if self.cfg.num_gpus:
                samples = samples.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            
            self.train_meter.data_toc()

            with amp.autocast(enabled=self.cfg.enable_amp):
                outputs = self.model(samples)
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = du.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
                
            # check infinity or NaN loss
            misc.check_finite_losses(loss_value)
                
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()
            if self.max_norm > 0:
                self.grad_scaler.unscale_(self.optimizer)  # gradients must be unscaled before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            
            # Update and log stats.
            self.train_meter.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            self.train_meter.update(lr=self.optimizer.param_groups[0]["lr"])
            # write to tensorboard format if available.
            if self.writer is not None and cur_iter % self.cfg.board_freq == 0:
                self.writer.add_scalars(
                    {'Train/loss': loss_value, 
                    'Train/loss_is_referred': loss_dict_reduced_scaled['loss_is_referred'],
                    'Train/traceL1_loss': loss_dict_reduced_scaled['loss_boxes'],
                    'Train/traceGIOU_loss': loss_dict_reduced_scaled['loss_giou'],
                    'Train/lr': self.optimizer.param_groups[0]["lr"],
                    'No_grad/cardinality_error': loss_dict_reduced['cardinality_error']}, 
                    global_step= data_size * cur_epoch + cur_iter)
            
            torch.cuda.synchronize()
            self.train_meter.iter_toc()  # do measure allreduce for this meter
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            torch.cuda.synchronize()
            self.train_meter.iter_tic()
        del samples
        # Log epoch stats.
        self.train_meter.synchronize_between_processes()
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
            
    @torch.no_grad()
    def eval_epoch(self, cur_epoch):
        # Evaluation mode enabled. The running stats would not be updated.
        self.model.eval()
        self.val_meter.iter_tic()
        epoch_step = (cur_epoch+1)//self.eval_period
        if ((cur_epoch+1) == self.epochs and self.epochs % self.eval_period !=0):
            epoch_step += 1
        epoch_step *= (len(self.data_loader_val) // self.cfg.board_freq)
        
        for cur_iter, (samples, targets) in enumerate(self.data_loader_val):
            # Transfer the data to the current GPU device.
            if self.cfg.num_gpus:
                samples = samples.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            
            self.val_meter.data_toc()

            outputs = self.model(samples)
            orig_sample_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).cuda(non_blocking=True)
            predictions = self.postprocessor(outputs, orig_sample_sizes)
            predictions = {target['item'].item(): prediction for target, prediction in zip(targets, predictions)}
            predictions_gathered = du.gather_dict(predictions)
            
            torch.cuda.synchronize()
            self.val_meter.iter_toc()
            self.val_meter.update(predictions_gathered)
            if self.writer is not None and cur_iter % self.cfg.board_freq == 0:
                medium_video = vis_utils.plot_midresult_as_video(
                    self.val_meter, list(predictions_gathered.keys()))
                self.writer.add_video(
                    medium_video, tag="Video Medium Result", 
                    global_step= epoch_step + cur_iter)
                del medium_video
            
            torch.cuda.synchronize()
            self.val_meter.iter_tic()
        del samples
        # Log epoch stats.
        self.val_meter.synchronize_between_processes()
        self.val_meter.summarize(save_pred=False)
        self.val_meter.log_epoch_stats(cur_epoch)
        if self.writer is not None:
            tb.plot_motmeter_table(
                self.writer, self.val_meter.summary, global_step=cur_epoch)
            self.writer.add_scalars({
                'MOT_METER': {'MOTA': self.val_meter.summary.loc['OVERALL', 'mota'],
                              'MOTP': self.val_meter.summary.loc['OVERALL', 'motp']},
            }, global_step=cur_epoch)
            
        self.val_meter.reset()

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
