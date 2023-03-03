"""Meters."""

import datetime
from functools import partial
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import motmetrics as mm

from utils.timer import Timer, start_timer
import utils.logging as logging
import utils.misc as misc
import utils.distributed as du
from utils.mot_tools import MotTracker, MotEval, PRMotEval, prmot_formatters

logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size, fmt=None):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value
        
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the depue!
        """
        if not du.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count
    
    def __str__(self):
        return self.fmt.format(
            median=self.get_win_median(),
            avg=self.get_win_avg(),
            global_avg=self.get_global_avg()
        )


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, epochs, log_period, output_dir=None):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            epochs (int): the max epoch.
            log_period (int): the log period of epoch.
            output_dir (str): the output directory for log
        """
        self.epoch_iters = epoch_iters
        self.epochs = epochs
        self.log_period = log_period
        self.max_epoch = epochs * epoch_iters
        self.init_timer()
        self.meters = defaultdict(partial(ScalarMeter, log_period))
        self.lr = None
        self.output_dir = output_dir
    
    def init_timer(self):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
    
    def reset(self):
        """
        Reset the Meter.
        """
        for meter in self.meters.values():
            meter.reset()
        self.lr = None
        self.init_timer()
    
    def iter_tic(self):
        """
        Start to record time.
        """
        start_timer(self.iter_timer)
        start_timer(self.data_timer)

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()
        
    def data_toc(self):
        self.data_timer.pause()
        start_timer(self.net_timer)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k == 'lr':
                self.lr = v
            else:
                self.meters[k].add_value(v)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self.log_period != 0:
            return
        eta_sec = self.iter_timer.avg_seconds() * (
            self.max_epoch - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.epochs),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.avg_seconds(),
            "dt_data": self.data_timer.avg_seconds(),
            "dt_net": self.net_timer.avg_seconds(),
            "eta": eta,
            "loss": {name: str(meter) for name, meter in self.meters.items()},
            "lr": str(self.lr),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.max_epoch - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.epochs),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": str(self.lr),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        stats.update(**{f'train_{name}': meter.get_global_avg() for name, meter in self.meters.items()})
        
        logging.log_json_stats(stats, self.output_dir)


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)


class MotValMeter(object):
    """
    Measures validation stats.
    
    Args:
            base_ds (dataset): the base dataset of dataloader.
            epochs (int): the max number of iteration of the current epoch.
            mot_type (str): the type of moteval, `'track'`, `'mot'`, `'prmot'`
    """

    def __init__(self, base_ds, epochs, output_dir=None, referred_threshold=0.5, start_frameid=1, mot_type='mot'):
        self.base_ds = base_ds
        self.epochs = epochs
        self.mot_type = mot_type
        self.init_meters(referred_threshold, start_frameid)
        self.init_timer()
        self.output_dir = output_dir
    
    @property
    def sequence_names(self):
        return list(self.meters.keys())
    
    def init_timer(self):
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.infer_timer = Timer()
    
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
        self.init_timer()
    
    def iter_tic(self):
        """
        Start to record time.
        """
        start_timer(self.iter_timer)
        start_timer(self.data_timer)

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()
        self.infer_tic()

    def data_toc(self):
        self.data_timer.pause()
        start_timer(self.net_timer)
    
    def infer_tic(self):
        start_timer(self.infer_timer)
        
    def infer_toc(self):
        self.infer_timer.pause()
    
    def init_meters(self, logit_threshold, start_frameid):
        if self.mot_type == 'track':
            self.meters = defaultdict(partial(MotTracker, logit_threshold=logit_threshold))
        elif self.mot_type == 'mot':
            self.meters = defaultdict(partial(MotEval, auto_id=False, 
                                              logit_threshold=logit_threshold,
                                              start_frameid=start_frameid))
        elif self.mot_type == 'prmot':
            self.meters = defaultdict(partial(PRMotEval, auto_id=False, 
                                              logit_threshold=logit_threshold,
                                              start_frameid=start_frameid))
        else:
            raise ValueError(f'{self.mot_type} is not supported!')
    
    def is_pure_track(self):
        return self.mot_type == 'track'
    
    def update(self, predictions: dict, log_threshold: float = None):
        items = list(np.unique(list(predictions.keys())))
        
        for item in items:
            sequence_info = self.get_sequence_info(item)
            if log_threshold is not None:
                self.meters[sequence_info['name']].logit_threshold = log_threshold
            self.meters[sequence_info['name']].add_value(predictions[item], sequence_info)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def get_sequence_info(self, item):
        sequence_parser, sequence_item = self.base_ds(item)
        sequence_name = sequence_parser.sequence_name
        if sequence_name not in self.meters.keys() or not self.is_pure_track():
            self.meters[sequence_name].gt = sequence_parser.gt
        frame_ids = sequence_parser.get_frame_ids(sequence_item)
        sampling_frame_ids = sequence_parser._get_sampling_frame_ids(sequence_item)
        sequence_imgs = sequence_parser.get_images(frame_ids, opacity=0.75)
        sequence_mate = sequence_parser.convert2mate(frame_ids)
        info = {'name': sequence_name,
                'item': sequence_item,
                'frame_ids': frame_ids,
                'sampling_frame_ids': sampling_frame_ids,
                'imgs': sequence_imgs,
                'mate': sequence_mate}
        if 'ignored_region' in sequence_parser.__dict__:
            info.update({'ignored_region': sequence_parser.ignored_region})
        return info
    
    def get_sequence_data(self, sequence_name, sequence_item):
        """Get sequence item Mot pred data from sequence name moteval"""
        return self.meters[sequence_name].pred_data[sequence_item]
    
    def summarize(self, save_pred=False):
        for sequence_name, meter in self.meters.items():
            logger.info(f'Summarizing {sequence_name} Video.')
            meter.match_pred_data()
            meter.summarize()
            if save_pred:
                meter.pred_dataframe.to_csv(f'{self.output_dir}/{sequence_name}_pred.txt', header=None, index=None, sep=',', mode='w')
        
        if self.is_pure_track():
            self.summary = pd.DataFrame()
            self.summary_str = 'Pure Track, NO MOT'
            self.summary_markdown = 'Pure Track, NO MOT'
        else:
            self.render_summary()
            
    def render_summary(self):
        mh = mm.metrics.create()
        summary = mh.compute_many(
            [meter.acc.events for meter in self.meters.values()],
            metrics=mm.metrics.motchallenge_metrics,
            names=[name for name in self.meters.keys()],
            generate_overall=True
        )
        
        namemap = mm.io.motchallenge_metric_names
        formatters = mh.formatters
        
        summary = summary.rename(columns=namemap)
        formatters = {namemap.get(c, c): f for c, f in formatters.items()}
        markdownfmt = ('.1%', '.1%', '.1%', '.1%', '.1%', '.1%',
                      '.0f', '.0f', '.0f', '.0f', '.0f', '.0f', '.0f', '.0f', '.1%', '.3')
        
        self.summary_str = summary.to_string(formatters=formatters)
        markdown_headline = "\n### MOT METRIC RESULT 📄\n\n"
        self.summary_markdown = markdown_headline + summary.to_markdown(floatfmt=markdownfmt)
        self.summary = summary
        
        if self.mot_type == 'prmot':
            prmota = np.array([meter.pr_mota for meter in self.meters.values()])
            prmotp = np.array([meter.pr_motp for meter in self.meters.values()])
            prids = np.array([meter.pr_ids for meter in self.meters.values()])
            prmt = np.array([meter.pr_mt for meter in self.meters.values()])
            prml = np.array([meter.pr_ml for meter in self.meters.values()])
            prfm = np.array([meter.pr_fm for meter in self.meters.values()])
            prfp = np.array([meter.pr_fp for meter in self.meters.values()])
            prfn = np.array([meter.pr_fn for meter in self.meters.values()])
            thresholds = np.array([meter.logit_threshold for meter in self.meters.values()])
            
            summary = pd.DataFrame({
                'PR-MOTA': np.hstack((prmota, prmota.mean())),
                'PR-MOTP': np.hstack((prmotp, prmotp.mean())),
                'PR-IDS': np.hstack((prids, prids.mean())),
                'PR-MT': np.hstack((prmt, prmt.mean())),
                'PR-ML': np.hstack((prml, prml.mean())),
                'PR-FM': np.hstack((prfm, prfm.mean())),
                'PR-FP': np.hstack((prfp, prfp.mean())),
                'PR-FN': np.hstack((prfn, prfn.mean())),
                'TH': np.hstack((thresholds, thresholds.mean())),
            }, index=[name for name in self.meters.keys()]+['OVERALL'])
            
            formatters = prmot_formatters
            markdownfmt = ('.1%', '.1%', '.1%', '.1f', '.1%', '.1%', '.1f', '.1f', '.1f', '.2f')
            
            self.summary_str = summary.to_string(formatters=formatters) + '\n' + self.summary_str
            markdown_headline = "\n### PR-MOT METRIC RESULT 📄\n\n"
            self.summary_markdown = markdown_headline + summary.to_markdown(floatfmt=markdownfmt) + '\n' + self.summary_markdown
                
    def log_epoch_stats(self, cur_epoch: int, anno: str=""):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            anno (str): the annotation of current state.
        """
        total_sequence_length = sum([seq.seqLength for seq in self.base_ds().data])
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self.epochs),
            "total_seq_len": total_sequence_length,
            "iter_time_diff": self.iter_timer.seconds(),
            "iter_time_avg": self.iter_timer.avg_seconds(),
            "iter_fps": "{:.2f}".format(total_sequence_length / self.iter_timer.seconds()),
            "infer_time_diff": self.infer_timer.seconds(),
            "infer_time_avg": self.infer_timer.avg_seconds(),
            "infer_fps": "{:.2f}".format(total_sequence_length / self.infer_timer.seconds()),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if anno != "":
            stats.update({"anno": anno})
        if not self.is_pure_track():
            for sequence_name in self.summary.index:
                stats.update({f'{sequence_name}_mota': self.summary.loc[sequence_name, 'MOTA']})
                stats.update({f'{sequence_name}_motp': self.summary.loc[sequence_name, 'MOTP']})
                
        logging.log_json_stats(stats, self.output_dir)

    def __str__(self):
        return self.summary_str

