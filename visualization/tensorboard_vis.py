import logging as log
import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import interpolate, integrate
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import utils.logging as logging
import visualization.utils as vis_utils

log.getLogger('PIL').setLevel(log.WARNING)
log.getLogger("matplotlib").setLevel(log.ERROR)

logger = logging.get_logger(__name__)


class TensorboardWriter(object):
    """
    Helper class to log information to Tensorboard.
    """

    def __init__(self, log_dir='runs'):
        """
        Args:
            log_dir(str): Save directory localtion.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(
            "To see logged results in Tensorboard, please launch using the command \
            `tensorboard  --port=<port-number> --logdir {}`".format(log_dir)
        )
        
    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Tensorboard logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optinal[int]): Global step value to record.
        """
        if self.writer is not None:
            for key, item in data_dict.items():
                if isinstance(item, dict):
                    self.writer.add_scalars(key, item, global_step)
                else:
                    self.writer.add_scalar(key, item, global_step)

    def add_figure(self, figure, tag="Figure Output", global_step=None):
        """
        Render matplotlib figure into an image and add it to summary.
        Args:
            figure (matplotlib.pyplot.figure): Figure or a list of figures
            tag (Optional[str]): name of the figure.
            global_step(Optional[int]): current step.
        """
        self.writer.add_figure(tag, figure, global_step)

    def add_video(self, vid_tensor, tag="Video Input", global_step=None, fps=4):
        """
        Add input to tensorboard SummaryWriter as a video.
        Args:
            vid_tensor (tensor): shape of (B, T, C, H, W). Values should lie
                [0, 255] for type uint8 or [0, 1] for type float.
            tag (Optional[str]): name of the video.
            global_step(Optional[int]): current step.
            fps (int): frames per second.
        """
        self.writer.add_video(tag, vid_tensor, global_step=global_step, fps=fps)
    
    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary.
        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int, optional): Global step value to record. Defaults to None.
            walltime (float, optional): Optional override default walltime (time.time()) seconds after epoch of event. Defaults to None.
        """
        self.writer.add_text(tag, text_string, global_step=global_step, walltime=walltime)
    
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        """
        Add a set of hyperparameters to be compared in TensorBoard.
        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the name of the hyper parameter and its corresponding value.
            metric_dict (dict): Each key-value pair in the dictionary is the name of the metric and its corresponding value. 
            hparam_domain_discrete (Optional[Dict[str, List[Any]]]): A dictionary that contains names of the hyperparameters and all discrete values they can hold.
            run_name (str): Name of the run, to be included as part of the logdir. If unspecified, will use current timestamp.
        """
        self.writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete, run_name)
    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        # Manual remove temp gif file by moviepy
        for path in Path(tempfile.gettempdir()).glob('*.gif'):
            os.remove(path)
        
        self.writer.flush()
        self.writer.close()


def plot_motmeter_table(writer: TensorboardWriter, 
                        motmeter_summary:pd.DataFrame, 
                        global_step=None,):
    import motmetrics as mm
    figure = vis_utils.plot_table(
        figsize=(20, 5),
        cellText=np.around(motmeter_summary.values, 2),
        colLabels=[mm.io.motchallenge_metric_names[name] for name in motmeter_summary.columns],
        rowLabels=motmeter_summary.index,
        cellLoc='center'
    )
    writer.add_figure(figure, "MOT Meter Summary💡", global_step)


def plot_dec_atten(writer: TensorboardWriter, attn_dict, results, 
                   base_ds, obj_num=1, frame_step=1, cur_epoch=0,
                   logit_threshold=0.2):
    obj_counter = 0
    if 'attn_points' in attn_dict.keys():
        N, Q, Nh, Nl, Np = attn_dict['dec_attn_weights'].shape
        attn_weights = attn_dict['dec_attn_weights'].transpose(2, 3).reshape(N, Q, Nl, -1)
        attn_points = attn_dict['attn_points'].transpose(2, 3).reshape(N, Q, Nl, -1, 2) * attn_dict['spatial_shapes'][None, None, :, None, :]
        refer_points = attn_dict['reference_points'] * attn_dict['spatial_shapes'][None, None, :, :]
        expand_points = attn_dict['expand_points'].permute(0,1,3,5,2,4,6).reshape(N, Q, Nl, -1, Nh*Np, 2)
        for attn_weight, attn_point, refer_point, expand_point, (key, value) in zip \
            (attn_weights, attn_points, refer_points, expand_points, results.items()):
            sequence_parser, _ = base_ds(key)
            
            pred_scores = value['scores']
            pred_frameids = value['frameids'].cpu().numpy()[::frame_step]
            # filter less than logit_threshold obj 
            obj_index = torch.nonzero(pred_scores > logit_threshold, as_tuple=True)[0]
            pred_scores = pred_scores[obj_index].cpu().numpy()
            pred_boxes = value['boxes'][obj_index].cpu().numpy()[:, ::frame_step]
            pred_queryids = value['queryids'][obj_index].cpu().numpy()
            pil_imgs = sequence_parser.get_images(pred_frameids)
            expand_point *= torch.tensor([sequence_parser.imWidth, sequence_parser.imHeight],
                                         device=expand_point.device)
            
            for boxes, queryid in zip(pred_boxes, pred_queryids):
                if obj_counter >= obj_num:
                    break
                figure = vis_utils.plot_deformable_attention_weights(
                    attn_weight[queryid].cpu().numpy(),
                    attn_point[queryid].cpu().numpy(),
                    attn_dict['spatial_shapes'].cpu().numpy(),
                    refer_point[queryid].cpu().numpy(),
                    expand_point[queryid, :, ::frame_step].cpu().numpy(),
                    pil_imgs, boxes, queryid, pred_frameids)
                writer.add_figure(figure, 'Deformable DETR encoder-decoder multi-head attention weights', 
                                  obj_counter + obj_num * cur_epoch)
                obj_counter += 1
        
    elif 'dec_attn_weights' in attn_dict.keys():
        attn_weights = attn_dict['dec_attn_weights']
        for attn_weight, (key, value) in zip(attn_weights, results.items()):
            sequence_parser, _ = base_ds(key)
            
            pred_scores = value['scores'] # n t
            # get mask of query above threshold and less than two elements are True in T dim.
            mask = (pred_scores > logit_threshold).sum(-1) > 2 # n
            pred_queryids = torch.nonzero(mask, as_tuple=True)[0].cpu().numpy() # n
            pred_boxes = value['boxes'][mask].cpu().numpy()[:, ::frame_step]
            pred_frameids = value['frameids'].cpu().numpy()[::frame_step]
            pil_imgs = sequence_parser.get_images(pred_frameids)
            
            for boxes, queryid in zip(pred_boxes, pred_queryids):
                if obj_counter >= obj_num:
                    break
                weights = attn_weight[queryid].cpu().numpy() # t h w
                scores = pred_scores[queryid].cpu().numpy() # t
                figure = vis_utils.plot_multi_head_attention_weights(
                    weights, pil_imgs, boxes, queryid, pred_frameids, scores=scores)
                writer.add_figure(figure, 'DETR encoder-decoder multi-head attention weights', 
                                  obj_counter + obj_num * cur_epoch)
                obj_counter += 1
    else:
        logger.warning('The dec attn dict has non-understand key.')


def plot_pr_curve(writer: TensorboardWriter, recalls: np.ndarray, precisions: np.ndarray, 
                  num_thresholds: int=None, tag:str='PR CURVE', kind='linear',
                  motas: np.ndarray=None):
    if num_thresholds is None:
        num_thresholds = len(recalls)
    
    pr_f = interpolate.interp1d(np.nan_to_num(recalls), np.nan_to_num(precisions), 
                                kind=kind, fill_value='extrapolate')
    new_x = np.linspace(0, 1, num_thresholds)
    new_y = np.nan_to_num(pr_f(new_x))
    
    if motas is None:
        area = integrate.quad(pr_f, 0, 1)[0]
        figure = vis_utils.plot_pr_curve(new_x, new_y, area)
        writer.add_figure(figure, tag)
    else:
        pr_mota_f = interpolate.interp2d(np.nan_to_num(recalls), np.nan_to_num(precisions),
                                         np.nan_to_num(motas, -1.0), kind=kind)
        area = integrate.dblquad(pr_mota_f, 0, 1, 0, 1)[0]
        new_z = np.nan_to_num(np.diagonal(pr_mota_f(new_x, new_y)), -1.0)
        figure = vis_utils.plot_pr_mota_curve(new_x, new_y, new_z, area)
        writer.add_figure(figure, tag+'(MOTA version)')
        
