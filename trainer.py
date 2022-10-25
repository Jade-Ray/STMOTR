"""
This file contains a Trainer class which handles the training and evaluation of MMOTR.
"""
import pprint
import gc
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from einops import rearrange

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
                find_unused_parameters=True,)
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
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[40], gamma=0.4, verbose=True)
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
        logger.info(f"Training done:\n {str(self.val_meter)}")
        
        return str(self.val_meter)

    @torch.no_grad()
    def test(self):
        logger.info("Running Test DataLoader...")
        self.model.eval()
        total_results = {}
        for samples, targets in tqdm(self.data_loader_val):
        
            # Transfer the data to the current GPU device.
            if self.cfg.num_gpus:
                samples = samples.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            
            outputs = self.model(samples)
            orig_sample_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).cuda(non_blocking=True)
            predictions = self.postprocessor(outputs, orig_sample_sizes)
            predictions = {target['item'].item(): prediction for target, prediction in zip(targets, predictions)}
            total_results.update(du.gather_dict(predictions))
            
            torch.cuda.synchronize()
        del samples
        
        logger.info("Getten total results, Then calculate PR_CURVE...")
        # Calculate PR test in every log_threshold
        log_thresholds = np.linspace(0, 1, 10)
        recall_record, precision_record, mota_record = defaultdict(list), defaultdict(list), defaultdict(list)
        
        for log_threshold in log_thresholds:
            self.mot_meter.update(total_results, log_threshold=log_threshold)
            self.mot_meter.summarize()
            logger.info(f"In {log_threshold} Threshold:\n {str(self.mot_meter)}")
            
            for sequence_name in self.mot_meter.summary.index:
                recall_record[sequence_name].append(self.mot_meter.summary.loc[sequence_name, 'recall'])
                precision_record[sequence_name].append(self.mot_meter.summary.loc[sequence_name, 'precision'])
                mota_record[sequence_name].append(self.mot_meter.summary.loc[sequence_name, 'mota'])
            
            self.mot_meter.reset()

        scores = {}
        if self.writer is not None:
            for sequence_name in self.mot_meter.summary.index:
                score = tb.plot_pr_curve(self.writer, 
                                        np.array(recall_record[sequence_name]), 
                                        np.array(precision_record[sequence_name]), 
                                        motas=np.array(mota_record[sequence_name]),
                                        tag=f'{sequence_name} PR CURVE')
                scores[sequence_name] = score
        
        return scores

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
        
        for cur_iter, (samples, targets) in enumerate(tqdm(self.data_loader_val)):
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

    @torch.no_grad()
    def visualization(self, vis_input=True, vis_mid=True, 
                      vis_res=True, vis_ablation=True):
        logger.info('Model Visulization.')
        self.model.eval()
        
        if isinstance(self.cfg.board_vis_item, (list, tuple)):
            vis_item = self.cfg.board_vis_item
        elif self.cfg.board_vis_item == -1:
            vis_item = list(range(len(self.data_loader_val)))
            
        if vis_ablation:
            if self.model._get_name() == 'DeformableMMOTR':
                spatial_shapes, point_offsets, dec_attn_weights = [], [], []
                reference_points, embed_points = [], []
                
                def referPoints_hook(module, input, output):
                    reference_points.append(input[1])
                    spatial_shapes.append(input[3])
                
                hooks = [
                    self.model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                        referPoints_hook
                    ),
                    self.model.transformer.decoder.layers[-1].cross_attn.sampling_offsets.register_forward_hook(
                        lambda self, input, output: point_offsets.append(output)
                    ),
                    self.model.transformer.decoder.layers[-1].cross_attn.attention_weights.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output)  
                    ),
                    self.model.box_head.layers[-1].register_forward_hook(
                        lambda self, input, output: embed_points.append(output[-1])
                    )
                ]
            elif self.model._get_name() == 'MMOTR':
                conv_features, dec_attn_weights = [], []
                hooks = [
                    self.model.transformer.register_forward_hook(
                        lambda self, input, output: conv_features.append(input[0]) # t b c h w
                    ),
                    self.model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])  # (t b) q (h w)
                    )
                ]
            else:
                logger.warning(f'Not supported Ablation transformer model {self.model._get_name()}')
                hooks = []
        
        pbar = tqdm(self.data_loader_val)
        for cur_iter, (samples, targets) in enumerate(pbar):
            pbar.set_description(
                f'processing: {targets[0]["frame_indexes"][0]}~{targets[-1]["frame_indexes"][-1]} frames')
            # visualization input images
            if self.writer is not None and vis_input and cur_iter in vis_item:
                input_video = vis_utils.plot_inputs_as_video(
                    rearrange(samples.tensors, 't b c h w -> b c t h w'), 
                    [t['boxes'] for t in targets],
                    [t['frame_indexes'] for t in targets],
                    [t['referred'] for t in targets],
                )
                self.writer.add_video(input_video, tag='Video Input', global_step=cur_iter)
                del input_video

            # Transfer the data to the current GPU device.
            if self.cfg.num_gpus:
                samples = samples.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            
            self.val_meter.data_toc()

            outputs = self.model(samples)
            orig_sample_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).cuda(non_blocking=True)
            frameids = torch.stack([t["frame_indexes"] for t in targets], dim=0).cuda(non_blocking=True)
            predictions = self.postprocessor(outputs, orig_sample_sizes)
            for frameid, prediction in zip(frameids, predictions):
                prediction.update({'frameids': frameid})
            predictions = {target['item'].item(): prediction
                           for target, prediction in zip(targets, predictions)}
            predictions_gathered = du.gather_dict(predictions)
            
            torch.cuda.synchronize()
            self.val_meter.iter_toc()
            self.val_meter.update(predictions_gathered)
            
            # Visualization medium images
            if self.writer is not None and vis_mid and cur_iter in vis_item:
                medium_video = vis_utils.plot_midresult_as_video(
                    self.val_meter, list(predictions_gathered.keys()))
                self.writer.add_video(
                    medium_video, tag="Video Medium Result", 
                    global_step=cur_iter)
                del medium_video
            
            # visualization ablation images
            if self.writer is not None and vis_ablation and cur_iter in vis_item:
                if self.model._get_name() == 'DeformableMMOTR':
                    t, b = embed_points[-1].shape[:2]
                    nlevels, nheads, npoints = self.model.feature_levels, self.model.nheads, self.model.npoints
                    # calculate weights for every level to [bs, lq, t, nh, nl, np]:
                    attn_weights = rearrange(dec_attn_weights[-1], '(t b) q (h l p) -> b q t h (l p)',
                                             t=t, b=b, h=nheads, l=nlevels, p=npoints)
                    attn_weights = rearrange(attn_weights.softmax(-1), 'b q t h (l p) -> b q t h l p', l=nlevels, p=npoints)
                    # calculate attn points for every level to [bs, lq, t, nh, nl, np, 2]:
                    offsets = rearrange(point_offsets[-1], '(t b) q (h l p c) -> b q t h l p c',
                                        t=t, b=b, h=nheads, l=nlevels, p=npoints, c=2)
                    offset_normalizer = torch.stack([spatial_shapes[-1][..., 1], spatial_shapes[-1][..., 0]], -1) # (h, w) to (w, h)
                    reference_points = rearrange(reference_points[-1], '(t b) q l c -> b q t l c', t=t, b=b)
                    attn_points = reference_points[:, :, :, None, :, None, :] \
                        + offsets / offset_normalizer[None, None, None, None, :, None, :]
                    # calculate expand points for every frame to [bs, lq, t, nh, nl, np, 2] formed (cx, xy)
                    expand_points = misc.inverse_sigmoid(attn_points) \
                        + rearrange(embed_points[-1], 't b q c -> b q t c')[:, :, :, None, None, None, :2]
                    
                    attn_dict = {'dec_attn_weights': attn_weights,
                                'attn_points': attn_points.clamp(min=0, max=1),
                                'reference_points': reference_points,
                                'spatial_shapes': offset_normalizer,
                                'expand_points': expand_points.sigmoid()}
                    
                elif self.model._get_name() == 'MMOTR':
                    t, b, _, h, w = conv_features[-1].shape
                    attn_dict = {
                        'dec_attn_weights': rearrange(dec_attn_weights[-1], '(t b) q (h w) -> b q t h w', t=t, b=b, h=h, w=w),
                        'conv_features': rearrange(conv_features[-1], 't b c h w -> b t c h w')} 
                else:
                    attn_dict = {}
                tb.plot_dec_atten(self.writer, attn_dict, predictions_gathered, 
                                  self.val_meter.base_ds, cur_epoch=cur_iter)
            
            torch.cuda.synchronize()
            self.val_meter.iter_tic()
        del samples
        # Log epoch stats.
        logger.info('Summarizing meter...')
        self.val_meter.synchronize_between_processes()
        self.val_meter.summarize(save_pred=False)
        
        if vis_ablation:
            for hook in hooks:
                hook.remove()
            
        # visualization final images
        logger.info('Visualizing predict video...')
        if self.writer is not None and vis_res:
            for i, (sequence_name, meter) in enumerate(self.val_meter.meters.items()):
                final_video = vis_utils.plot_pred_as_video(
                    sequence_name, meter, self.val_meter.base_ds, 
                    save_video=self.cfg.board_vis_res_save, 
                    output_dir=self.output_dir,
                    plot_interval=self.cfg.board_vis_res_interval,
                    resize=self.cfg.board_vis_res_size)
                self.writer.add_video(final_video, tag="Video Pred Result", global_step=i)
                del final_video
            
        if self.writer is not None:
            tb.plot_motmeter_table(self.writer, self.val_meter.summary)
        
        time.sleep(1)
        logger.info('Visulization Done.')

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
