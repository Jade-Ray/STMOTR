import logging as log
import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

import utils.logging as logging
import visualization.utils as vis_utils
from utils.mot_tools import PRMotEval
        
log.getLogger('torch.utils.tensorboard.summary').setLevel(log.ERROR)
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
    
    def add_text(self, tag, text_string, global_step=None, walltime=None, format_trans=False):
        """Add text data to summary.
        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int, optional): Global step value to record. Defaults to None.
            walltime (float, optional): Optional override default walltime (time.time()) seconds after epoch of event. Defaults to None.
            format_trans (bool, optional): Transform string format to markdown reader.
        """
        if format_trans:
            # Tensorboard understands markdown so you can actually replace \n with <br/> and   with &nbsp;
            text_string = text_string.replace( '\n', '<br/>').replace(' ', '&nbsp;&nbsp;')
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
        figsize=(20, 10),
        cellText=np.around(motmeter_summary.values, 2),
        colLabels=[mm.io.motchallenge_metric_names[name] for name in motmeter_summary.columns],
        rowLabels=motmeter_summary.index,
        cellLoc='center'
    )
    writer.add_figure(figure, "MOT Meter Summary💡", global_step)


def plot_dec_atten(writer: TensorboardWriter, attn_dict, results, 
                   base_ds, obj_num=1, frame_step=1, cur_epoch=0,
                   logit_threshold=0.5):
    obj_counter = 0
    if 'attn_points' in attn_dict.keys():
        for i, (key, value) in enumerate(results.items()):
            sequence_parser, _ = base_ds(key)
            pred_scores = value['scores'] # n t
            # get mask of query above threshold and less than two elements are True in T dim.
            mask = (pred_scores > logit_threshold).sum(-1) > 2 # n
            pred_queryids = torch.nonzero(mask, as_tuple=True)[0].cpu().numpy() # n
            pred_boxes = value['boxes'][mask].cpu().numpy()[:, ::frame_step]
            pred_frameids = value['frameids'].cpu().numpy()[::frame_step]
            pil_imgs = sequence_parser.get_images(pred_frameids, opacity=0.75)
            img_size = torch.tensor([sequence_parser.imWidth, sequence_parser.imHeight],
                                    device=mask.device)
            
            for boxes, queryid in zip(pred_boxes, pred_queryids):
                if obj_counter >= obj_num:
                    break
                attn_weight = rearrange(attn_dict['dec_attn_weights'][i, queryid],
                                        't h l p -> t l (h p)')
                attn_point = rearrange(attn_dict['expand_points'][i, queryid, ::frame_step], 
                                       't h l p c -> t l (h p) c') * img_size
                refer_point = attn_dict['reference_points'][i, queryid, ::frame_step] * img_size
                
                figure = vis_utils.plot_deformable_lvl_attn_weights(
                    attn_weight.cpu().numpy(), attn_point.cpu().numpy(),
                    refer_point.cpu().numpy(), pil_imgs, boxes, pred_frameids)
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


def plot_prmot(writer: TensorboardWriter, meter: PRMotEval, sequence_name: str):
    figure = vis_utils.plot_pr_curve(meter.precisions, meter.recalls, meter.ap)
    writer.add_figure(figure, 'PR CURVE〽️')
    
    figure = vis_utils.plot_pr_mot_curve(meter.precisions, meter.recalls, meter.motas,
                                         f'PR-MOTA {meter.pr_mota:.2f}')
    writer.add_figure(figure, f'{sequence_name} PR MOTA CURVE〽️')
    
    figure = vis_utils.plot_pr_mot_curve(meter.precisions, meter.recalls, meter.motps,
                                         f'PR-MOTP {meter.pr_motp:.2f}')
    writer.add_figure(figure, 'PR MOTP CURVE〽️')

    pr_record_md = meter.get_record_frame().to_markdown(
        floatfmt=('.2f', '.3f','.3f', '.1%', '.1%', '.1f', '.1%', '.1%', '.1f', '.1f', '.1f'))
    writer.add_text(f'{sequence_name} PR RECORD',
                    f'\n- {sequence_name} 📄\n\n' + pr_record_md)
