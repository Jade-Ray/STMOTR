from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
import motmetrics as mm

from datasets.interpolations import InterpolateTrack as interpolate
from utils.box_ops import box_xyxy_to_xywh, box_iou, box_nms


class MOTeval(object):
    
    def __init__(self, auto_id=False, logit_threshold=0.2,
                start_frameid=1, iou_threshold=0.75):
        self.acc = mm.MOTAccumulator(auto_id=auto_id)
        self.mh = mm.metrics.create()
        self.pred_data = defaultdict(dict)
        self.pred_dataframe = self._create_mot_dataframe()
        self.gt = None
        
        self.logit_threshold = logit_threshold
        self.iou_threshold = iou_threshold
        self.start_frameid = start_frameid
    
    @property
    def frameids(self):
        return np.unique(self.gt['frame_index'].to_numpy())
    
    def reset(self):
        self.acc.reset()
        self.pred_data = defaultdict(dict)
        self.pred_dataframe = self._create_mot_dataframe()
        
    def add_value(self, pred: dict, tgt: dict):
        """
        In this process, we need to postprocessing predictions.
            
            - filter of boxes with NMS, 
            - filter of boxes with ignore region in GT, 
            - selection of above threshold score boxes, 
            - poly fitting of skiped frame.
        """
        boxes = pred['boxes'].cpu().numpy() # [N, T, 4]
        scores = pred['scores'].cpu().numpy() # [N, T]
        
        # filter of boxes with NMS
        for t in range(boxes.shape[1]):
            index = box_nms(boxes[:, t], scores[:, t], self.iou_threshold)
            nms_mask = np.ones(scores[:, t].size, dtype=bool)
            nms_mask[index] = False
            scores[nms_mask, t] = 0.0
        
        # filter ignored_region boxes
        if 'ignored_region' in tgt.keys():
            ignore_mask = np.apply_along_axis(
                tgt['ignored_region'].in_ignored_region, -1, box_xyxy_to_xywh(boxes))
            scores[ignore_mask] = 0.0
            
        # selection of above threshold score boxes
        # get mask of query above threshold and less than two elements are True in T dim.
        mask = (scores > self.logit_threshold).sum(-1) > 2
        query_ids = np.where(mask)
        boxes = boxes[mask]
        scores = scores[mask]

        # poly fit all results if frame skiped
        pred_frameids = tgt['sampling_frame_ids']
        tgt_frameids = tgt['frame_ids']
        
        if len(pred_frameids) != len(tgt_frameids):
            boxes = self.interpolate_tracks(pred_frameids, boxes, tgt_frameids, 'avg')
            scores = self.interpolate_tracks(pred_frameids, scores, tgt_frameids, 'avg')
        
        self.pred_data[tgt['item']].update({
            'frame_ids': tgt_frameids,
            'scores': scores,
            'boxes': boxes,
            'query_ids': query_ids,
        })
    
    def interpolate_tracks(self, frameids, tracks, new_frameids, type='copy'):
        """interpolate tracks [N, T, c] new frameids."""
        if tracks.shape[0] == 0:
            # fill empty tracks to new frameids.
            return np.zeros((0, len(new_frameids), tracks.shape[-1]))
        if type == 'poly2':
            y = np.stack(
                [interpolate.poly(frameids, t, new_frameids, 2) for t in tracks], axis=0,)
            return np.clip(y, 0, None)
        elif type == 'copy':
            return np.stack(
                [interpolate.copy(frameids, t, new_frameids) for t in tracks], axis=0,)
        elif type == 'avg':
            return np.stack(
                [interpolate.average(frameids, t, new_frameids) for t in tracks], axis=0,)
        else:
            raise ValueError(f'Unknonw interpolate way {type}')
            
    
    def synchronize_between_processes(self):
        pass
    
    def _create_mot_dataframe(self, data=[]):
        return pd.DataFrame(data, columns= ["frame_index", "track_id", "l", "t", "r", "b", 
                "confidence", "object_type", "visibility"])
            
    def match_pred_data(self):
        ai_id = AutoIncreseId()
        last_frameid = float('inf')
        last_pred_boxes = []
        last_trackids = []
        # because the ground-truth align first frame, so matched inversion.
        for _, pred in sorted(self.pred_data.items(), key=lambda item: int(item[0]), reverse=True):
            frameids = list(pred['frame_ids'])
            frameid_index = 0 if last_frameid == float('inf') else frameids.index(last_frameid)
            current_pred_boxes = pred['boxes'][:, frameid_index]
            
            if len(last_trackids) == 0:
                track_ids = [ai_id.id for _ in range(pred['boxes'].shape[0])]
            else:
                track_ids = []
                cur_ind, last_ind, mask = hungarian_match_iou(current_pred_boxes, last_pred_boxes)
                for i in np.arange(current_pred_boxes.shape[0]):
                    if i in cur_ind:
                        pos = list(cur_ind).index(i)
                        if mask[pos]:
                            track_ids.append(last_trackids[last_ind[pos]])
                        else:
                            track_ids.append(ai_id.id)
                    else:
                        track_ids.append(ai_id.id)
                        
            for boxes, scores, track_id in zip(pred['boxes'], pred['scores'], track_ids):
                boxes = boxes.astype(int)
                scores = np.around(scores, 5)
                data = [[frameid, track_id, l, t, r, b, score, 1, score] 
                        for frameid, (l, t, r, b), score in zip(frameids, boxes, scores) 
                        if score > self.logit_threshold and frameid < last_frameid]
                dataframe = self._create_mot_dataframe(data)
                self.pred_dataframe = pd.concat([self.pred_dataframe, dataframe])
            last_frameid = min(frameids)
            last_trackids = track_ids
            last_pred_boxes = pred['boxes'][:, 0]
            
    def summarize(self):
        for frameid in self.frameids:
            gt = self.gt[self.gt['frame_index'] == frameid]
            if 'confidence' in gt:
                gt = gt[gt['confidence'] == 1]
            gt_boxes = gt.loc[:, ['l', 't', 'r', 'b']].to_numpy()
            gt_ids = gt['track_id'].to_numpy()
            
            pred = self.get_pred_dataframe(frameid)
            pred_boxes = pred[pred['object_type'] == 1].loc[:, ['l', 't', 'r', 'b']].to_numpy()
            pred_ids = pred[pred['object_type'] == 1]['track_id'].to_numpy()
            
            distances = mm.distances.iou_matrix(
                box_xyxy_to_xywh(gt_boxes), 
                box_xyxy_to_xywh(pred_boxes), 
                max_iou=.5
            )
            self.acc.update(gt_ids, pred_ids, distances, frameid)
    
    def get_pred_dataframe(self, frameid: int):
        return self.pred_dataframe[self.pred_dataframe['frame_index'] == frameid]
    
    def get_frame_events(self, frameid: int) -> np.ndarray:
        if frameid in self.acc.mot_events.index:
            return self.acc.mot_events.loc[frameid].values
        return []

    def get_summary(self, start_frameid: int = None, end_frameid: int = None):
        if start_frameid is None:
            start_frameid = self.start_frameid
        if end_frameid is None:
            return self.mh.compute(self.acc.mot_events.loc[start_frameid:])
        else:
            return self.mh.compute(self.acc.mot_events.loc[start_frameid : end_frameid])
    
    def get_box(self, frameid, trackid, mode='gt', type=int, format='xyxy'):
        if np.isnan(trackid):
            return [None] * 4
        if mode == 'gt':
            dataframe = self.gt[self.gt['frame_index'] == frameid]
        elif mode == 'pred':
            dataframe = self.pred_dataframe[self.pred_dataframe['frame_index'] == frameid]
        else:
            raise ValueError(f'Unknown mode {mode}')
        
        l, t, r, b= dataframe[dataframe['track_id'] == trackid].loc[:, ['l', 't', 'r', 'b']].to_numpy().squeeze().astype(type)
        
        if format == 'xywh':
            return (l, t, r-l, b-t)
        elif format == 'xyxy':
            return (l, t, r, b)
        else:
            raise ValueError(f'not supported format {format}')
    
    def _get_motchallage_summary(self, summary):
        return mm.io.render_summary(
            summary, formatters=self.mh.formatters, namemap=mm.io.motchallenge_metric_names)

    def __str__(self):
        r"The motchallage summary of all frames"
        return self._get_motchallage_summary(self.get_summary())


class AutoIncreseId(object):
    def __init__(self, init_id=0):
        self._id = init_id
    
    @property
    def id(self):
        self._id += 1
        return self._id


def hungarian_match_iou(src, tgt, cost_threshold=0.9):
    cost_iou = 1 - box_iou(src, tgt).numpy()
    indexes = linear_sum_assignment(cost_iou)
    mask = [cost_iou[src_ind, tgt_ind] < cost_threshold for src_ind, tgt_ind in zip(*indexes)]
    
    return (indexes[0], indexes[1], mask)


def hungarian_match_iou_vis(src, tgt, frameid):
    from pathlib import Path
    from PIL import Image, ImageDraw
    
    file = Path('data/UA_DETRAC/train/Insight-MVT_Annotation/MVI_20011') / f'img{frameid:05}.jpg'
    img = Image.open(file)
    draw = ImageDraw.Draw(img, "RGBA")
    
    cost_iou = 1 - box_iou(src, tgt).numpy()
    indexes = linear_sum_assignment(cost_iou)
    
    for i, j in zip(*indexes):
        color = tuple(np.random.choice(range(256), size=3))
        draw.rectangle(src[i], outline=color)
        draw.rectangle(tgt[j], outline=color)
    
    img.show()
