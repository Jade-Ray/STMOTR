import warnings
from collections import OrderedDict
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_toolkits.mplot3d.art3d as art3d

import torch
from torch import Tensor
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
from scipy import interpolate
import cv2 as cv

import utils.logging as logging
from datasets.video_parse import SingleVideoParserBase
from datasets.transforms import NormalizeInverse
from utils.meters import MotValMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def plot_inputs_as_video(inputs: Tensor, boxes: List[Tensor], 
                         frameids: List[Tensor], referred: List[Tensor],
                         norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]) -> Tensor:
    """
    Normalize Inverse inputs tensor, and draw Boundingboxes on images, return video tensor of (B, T, C, H, W).
    """
    assert len({inputs.shape[0], len(boxes), len(frameids)}) == 1
    normalizeInverse = NormalizeInverse(norm_mean, norm_std)
    B, C, T, H, W,  = inputs.shape
    
    outputs = []
    for batch_img, batch_boxes, batch_frameids, batch_referred in zip(inputs, boxes, frameids, referred):
        batch_img, batch_boxes = normalizeInverse(batch_img.transpose(0, 1), 
                                                  boxes=batch_boxes.transpose(0, 1))
        for img, box_list, frameid, refer in zip(batch_img, batch_boxes.numpy(), batch_frameids.numpy(), batch_referred.transpose(0, 1).numpy()):
            mat = ImageConvert.to_mat_image(img)
            cv.putText(mat, f'{frameid}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
            for (l, t, r, b), ref in zip(box_list.astype(int), refer):
                if ref:
                    cv_rectangle(mat, (l,t), (r,b), (0,255,0), thickness=-1, alpha=0.2)
                else:
                    cv_rectangle(mat, (l,t), (r,b), (0,255,0), thickness=2, style='dashed')
            outputs.append(ImageConvert.to_tensor(mat))
    outputs = torch.stack(outputs, dim=0).view(B, T, C, H, W)
    return outputs


@torch.no_grad()
def plot_midresult_as_video(mot_meter: MotValMeter, batch_items: list) -> Tensor:
    """
    Plot MotValMeter midresult, return video tensor of (B, T, C, H, W).
    """
    outputs = []
    for item in batch_items:
        info = mot_meter.get_sequence_info(item)
        midresults = mot_meter.get_sequence_data(
            info['name'], info['item'])
        imgs, mate, frameids = info['imgs'], info['mate'], info['frame_ids']
        assert len(imgs) == len(frameids)
        
        for i, img in enumerate(info['imgs']):
            mat = ImageConvert.to_mat_image(img)
            cv.putText(mat, f'{frameids[i]}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
            
            if 'boxes' in mate:
                gt_boxes = mate['boxes'][:,i].astype(int)
                if 'confidences' in mate.keys():
                    gt_confidences = mate['confidences'][:, i].astype(bool)
                else:
                    gt_confidences = np.ones(gt_boxes.shape[0]).astype(bool)
                for confidence, (l, t, r, b) in zip(gt_confidences, gt_boxes):
                    if confidence:
                        cv_rectangle(mat, (l,t), (r,b), (0,255,0), thickness=-1, alpha=0.2)
                    else:
                        cv_rectangle(mat, (l,t), (r,b), (0,255,0), thickness=2, style='dashed')
            
            for boxes, scores in zip(midresults['boxes'], midresults['scores']):
                l, t, r, b = boxes[i].astype(int)
                text = f'{scores[i]:0.2f}'
                cv.putText(mat, text, (l,t), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv.LINE_AA)
                if scores[i] > 0.5:
                    cv_rectangle(mat, (l,t), (r,b), (0, 0, 255), thickness=2)
                else:
                    cv_rectangle(mat, (l,t), (r,b), (255, 153, 51), thickness=2)
            outputs.append(ImageConvert.to_tensor(mat))
    outputs = torch.stack(outputs, dim=0)
    B, (_, C, H, W) = len(batch_items), outputs.shape
    return outputs.view(B, -1, C, H, W)


@torch.no_grad()
def plot_pred_as_video(sequence_name, meter, base_ds,
                       save_video=False, output_dir='', vis_events=True,
                       plot_interval=(1, 100), resize=(960, 540)) -> Tensor:
    """
    Plot MotValMeter final predication, return video tensor of (B, T, C, H, W).
    """
    outputs = []
    parser = base_ds(sequence_name=sequence_name)
    plot_interval = range(*plot_interval)
    rcg = RCG()
    
    if save_video:
        writer = cv.VideoWriter(f'{output_dir}/{sequence_name}.avi', 
                                cv.VideoWriter_fourcc(*'MJPG'), 10, 
                                (parser.imWidth, parser.imHeight))
    else:
        writer = None
    
    pbar = tqdm(meter.frameids)
    for frameid in pbar:
        pbar.set_description(f'ploting: {sequence_name} {frameid}f')
        if frameid not in plot_interval and writer is None:
            continue
        mat = ImageConvert.to_mat_image(parser.get_image(frameid, opacity=0.75))
        cv.putText(mat, f'{frameid}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
        
        if vis_events:
            plot_mot_events(mat, meter, frameid, text_pos=(5,parser.imHeight-5))
        else:
            plot_track(mat, meter.get_pred_dataframe, frameid, rcg=rcg, show_id=True)

        if frameid in plot_interval:
            output = cv.resize(mat, resize)
            outputs.append(ImageConvert.to_tensor(output))
        if writer is not None:
            writer.write(mat)
    
    if writer is not None:
        writer.release()
        logger.info(f'Pred Video {output_dir}/{sequence_name}.avi saving done.')
        
    return torch.stack(outputs, dim=0)[None] if len(outputs) !=0 else None


def plot_mot_events(mat, meter, frameid, text_pos=(5, 10)):
    mot_events = meter.get_frame_events(frameid)
    summary = meter.get_summary(end_frameid=frameid)
    text = f"mota:{summary['mota'][0]:.2f} motp:{summary['motp'][0]:.2f} nfp:{summary['num_false_positives'][0]} nmis:{summary['num_misses'][0]} nsw:{summary['num_switches'][0]}"
    cv.putText(mat, text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv.LINE_AA)
            
    for Type, OId, HId, Dis in mot_events:
        o_l, o_t, o_r, o_b = meter.get_gt_box(frameid, OId, format='xyxy')
        h_l, h_t, h_r, h_b = meter.get_pred_box(frameid, HId, format='xyxy')
        if Type == 'MATCH':
            cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (0, 255, 0), 2)
        elif Type == 'FP':
            cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (0, 0, 255), 2)
        elif Type == 'MISS':
            cv_rectangle(mat, (o_l,o_t), (o_r,o_b), (0, 0, 255), 2, style='dashed')
        elif Type == 'SWITCH':
            last_events = meter.get_frame_events(frameid - 1)
            last_index = np.where(last_events[:, 1]==OId)[0]
            if last_index.size == 0:
                last_HId = HId
            else:
                last_HId = last_events[last_index, 2][0]
            if last_HId in last_events[:, 2] or last_HId in last_events[:, 1]:
                o_l, o_t, o_r, o_b = meter.get_pred_box(frameid - 1, last_HId, format='xyxy')
            cv_rectangle(mat, (o_l,o_t), (o_r,o_b), (255, 0, 0), -1, alpha=0.2)
            cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (255, 0, 0), -1, alpha=0.2)
            cv.arrowedLine(mat, (o_l,o_t), (h_l,h_t), (255, 0, 0), 1)
            cv.arrowedLine(mat, (o_r,o_b), (h_r,h_b), (255, 0, 0), 1)
            cv.putText(mat, f'{Dis:.2f}', (o_l, o_t), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv.LINE_AA)


def plot_track(mat, dataframe_func, frameid, rcg=None, show_id=False, show_ignore=False):
    for id, group in dataframe_func(frameid).groupby('track_id'):
        if rcg is not None:
            c = tuple(map(int, rcg(id)))
        else:
            c = (0, 255, 0)
        l, t, r, b = group[['l', 't', 'r', 'b']].values[0].astype(int)
        confidence = group['confidence'].values[0] > 0
        if confidence:
            cv_rectangle(mat, (l,t), (r,b), c, 2)
            if show_id:
                cv.putText(mat, f'{id}', (l,t), cv.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, cv.LINE_AA)
        elif show_ignore:
            cv_rectangle(mat, (l,t), (r,b), c, 2, style='dashed')


def plot_video_parser(parser: SingleVideoParserBase, output_dir=''):
    video_path = f'{output_dir}/{parser.sequence_name}_gt.avi'
    writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'MJPG'), 10, 
                            (parser.imWidth, parser.imHeight))
    rcg = RCG()
    pbar = tqdm(range(parser.start_frame_id, parser.end_frame_id+1))
    for frameid in pbar:
        pbar.set_description(f'ploting: {parser.sequence_name} {frameid}f')
        mat = ImageConvert.to_mat_image(parser.get_image(frameid, opacity=0.75))
        cv.putText(mat, f'{frameid}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
        plot_track(mat, parser.get_gt, frameid, rcg=rcg, show_id=True)
        writer.write(mat)
        
    writer.release()
    logger.info(f'Pred Video {video_path} saving done.')


def plot_table(cellText, figsize=(12, 7), rowLabels=None, colLabels=None, **kwargs):
    fig, ax =plt.subplots(figsize=figsize)
    
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    
    ax.table(cellText=cellText, 
             colLabels=colLabels, 
             rowLabels=rowLabels,
             loc='center',
             **kwargs)
    fig.tight_layout()
    return fig


def plot_multi_head_attention_weights(attn_weight: np.ndarray, pil_imgs: List[Image.Image], 
                                      boxes: np.ndarray, query_id: int, frame_ids: np.ndarray,
                                      scores: np.ndarray=None, score_threshold: float=.5):
    """Visualize DETR encoder-decoder multi-head attention weights."""
    assert len({len(pil_imgs), len(frame_ids), len(boxes)}) == 1
    fig, axs = plt.subplots(ncols=len(pil_imgs), nrows=2, figsize=(22, 7))
    for i, ax_i in enumerate(axs.T):
        l, t, r, b = boxes[i]
        w, h = r - l, b - t
        
        ax = ax_i[0]
        ax.imshow(attn_weight[i])
        ax.axis('off')
        ax.set_title(f'frame id: {frame_ids[i]}')
        ax = ax_i[1]
        ax.imshow(pil_imgs[i])
        if scores is not None and scores[i] < score_threshold:
            ax.add_patch(plt.Rectangle((l, t), w, h, fill=False, color='cyan', linestyle='--', linewidth=2))
        else:
            ax.add_patch(plt.Rectangle((l, t), w, h, fill=False, color='cyan', linewidth=2))
        ax.axis('off')
    fig.suptitle(f'query id: {query_id}')
    fig.tight_layout()
    return fig


def plot_deformable_attn_weights_3d(attn_weights: np.ndarray, attn_points: np.ndarray, 
                                    refer_points: np.ndarray, 
                                    pil_imgs: List[Image.Image], boxes: np.ndarray, 
                                    frame_ids: np.ndarray):
    """Visualize Deformable DETR encoder-decoder multi-head attention weights as 3d.
    
    Args:
        attn_weights: [num_frames, (num_levels * num_heads * num_points)]
        attn_points: [num_frames, (num_levels * num_heads * num_points), 2]
        refer_points: [num_frames, 2]
        pil_imgs: [num_frames]
        boxes: [num_frames, 4]
        frame_ids: [num_frames]
    """
    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(projection='3d')
    w, h = pil_imgs[0].size
    num_points = attn_weights.shape[1]
    num_frame = len(pil_imgs)
    colormap = LinearSegmentedColormap.from_list('mycamp', ['b', 'r'])
    
    idx = np.argsort(attn_weights, axis=1)
    ax.scatter3D(np.take_along_axis(np.repeat(frame_ids, num_points).reshape(num_frame, -1), idx, axis=1),
                 np.take_along_axis(attn_points[..., 0], idx, axis=1), 
                 np.take_along_axis(attn_points[..., 1], idx, axis=1),
                 c=np.take_along_axis(attn_weights, idx, axis=1),
                 s=10, cmap=colormap, vmin=0.0, vmax=1.0)
    ax.plot3D(frame_ids, refer_points[..., 0], refer_points[..., 1], 'g+', markersize=8)
    for i, (l, t, r, b) in zip(frame_ids, boxes):
        rec = plt.Rectangle((l, t), r -l, b - t, fill=False, color='g', linewidth=2)
        ax.add_patch(rec)
        art3d.pathpatch_2d_to_3d(rec, z=i, zdir="x")
    
    ax.set_xlabel('frame')
    ax.set_xlim((frame_ids[0], frame_ids[-1]))
    ax.set_ylabel('width')
    ax.set_ylim((0, w))
    ax.set_zlabel('height')
    ax.set_zlim((h, 0))
    return fig


def plot_deformable_attn_weights_2d(attn_weights: np.ndarray, attn_points: np.ndarray, 
                                    refer_points: np.ndarray, 
                                    pil_imgs: List[Image.Image], boxes: np.ndarray, 
                                    frame_ids: np.ndarray):
    """Visualize Deformable DETR encoder-decoder multi-head attention weights as 3d.
    
    Args:
        attn_weights: [num_frames, (num_levels * num_heads * num_points)]
        attn_points: [num_frames, (num_levels * num_heads * num_points), 2]
        refer_points: [num_frames, 2]
        pil_imgs: [num_frames]
        boxes: [num_frames, 4]
        frame_ids: [num_frames]
    """
    nrows, ncols = len(pil_imgs), 1
    colormap = LinearSegmentedColormap.from_list('mycamp', ['b', 'r'])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(9, 14), dpi=120, layout='constrained')
    for i, (img, (l, t, r, b), ax_row) in enumerate(zip(pil_imgs, boxes, axs)):
        # ax_row.imshow(img)
        # ax_row.add_patch(plt.Rectangle((l, t), r - l, b - t, fill=False, color='g', linewidth=2))
        # ax_row.plot(refer_points[i, 0], refer_points[i, 1], 'g+', markersize=8)
        
        data = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        img = Image.fromarray(data, 'RGBA') 
        ax_row.imshow(img)
        idx = np.argsort(attn_weights, axis=1)
        sca = ax_row.scatter(attn_points[i, idx, 0], attn_points[i, idx, 1],
                             c=attn_weights[i, idx], s=40, cmap=colormap, 
                             vmin=0.0, vmax=1.0)

        ax_row.set_xticks([])
        ax_row.xaxis.set_label_position('top')
        ax_row.set_xlim((0, img.width))
        ax_row.set_yticks([])
        ax_row.set_ylim((img.height, 0))
    return fig


def plot_deformable_lvl_attn_weights(attn_weights: np.ndarray, attn_points: np.ndarray, 
                                     refer_points: np.ndarray, 
                                     pil_imgs: List[Image.Image], boxes: np.ndarray, 
                                     frame_ids: np.ndarray):
    """Visualize Deformable DETR encoder-decoder multi-head attention weights as num_frames x num_levels subplots.
    
    Args:
        attn_weights: [num_frames, num_levels, (num_heads * num_points)]
        attn_points: [num_frames, num_levels, (num_heads * num_points), 2]
        refer_points: [num_frames, num_levels, 2]
        pil_imgs: [num_frames]
        boxes: [num_frames, 4]
        frame_ids: [num_frames]
    """
    nrows, ncols = len(pil_imgs), attn_weights.shape[1]
    colormap = LinearSegmentedColormap.from_list('mycamp', ['b', 'r'])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(11, 7.86), dpi=160, layout='constrained')
    for i, (img, (l, t, r, b), ax_row) in enumerate(zip(pil_imgs, boxes, axs)):
        for j, ax_col in enumerate(ax_row):
            ax_col.imshow(img)
            idx = np.argsort(attn_weights[i, j])
            sca = ax_col.scatter(attn_points[i, j, idx, 0], attn_points[i, j, idx, 1], 
                                 c=attn_weights[i, j, idx], s=20, cmap=colormap,
                                 vmin=0.0, vmax=1.0)
            ax_col.add_patch(plt.Rectangle((l, t), r -l, b - t, fill=False, color='g', linewidth=2))
            ax_col.plot(refer_points[i, j, 0], refer_points[i, j, 1], 'g+', markersize=8)
            ax_col.set_xticks([])
            ax_col.xaxis.set_label_position('top')
            ax_col.set_xlim([0, img.width])
            ax_col.set_yticks([])
            ax_col.set_ylim([img.height, 0])
            if ax_col.get_subplotspec().is_first_row():
                ax_col.set_xlabel(f'feature level {j+1}', fontdict={"fontsize": 19, 'fontfamily': '得意黑 斜体'})
            if ax_col.get_subplotspec().is_first_col():
                ax_col.set_ylabel(f'frame {frame_ids[i]}', fontdict={"fontsize": 19, 'fontfamily': '得意黑 斜体'})
    cbar = fig.colorbar(sca, ax=axs[0, -1], shrink=0.9, pad=0.01)
    cbar.set_ticks([0.02, 0.98])
    cbar.set_ticklabels(['low', 'high'], fontdict={"fontsize": 19, 'fontfamily': '得意黑 斜体'})
    cbar.ax.tick_params(labelsize=15)
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, hspace=0, wspace=0)
    return fig


def plot_object_queries(queries_points: np.ndarray, queries_weights: np.ndarray):
    """Visualize object queries as scatter plot.
    
    Args:
        queries_points: [num_queries, num_frame, num_data, 2]
        queries_weights: [num_queries, num_frame, num_data]
    """
    nrows, ncols = queries_points.shape[0], queries_points.shape[1]
    colormap = LinearSegmentedColormap.from_list('mycamp', ['b', 'g', 'r'])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(8, 10), dpi=160, layout='constrained')
    for i, ax_row in enumerate(axs):
        for j, ax_col in enumerate(ax_row):
            idx = np.argsort(queries_weights[i, j])
            ax_col.scatter(queries_points[i, j, idx, 0], queries_points[i, j, idx, 1],
                           c=queries_weights[i, j, idx], s=2, cmap=colormap, 
                           vmin=0.0, vmax=1.0)
            ax_col.set_xticks([])
            ax_col.set_yticks([])
            ax_col.set_xlim([0, 1])
            ax_col.set_ylim([1, 0])
            ax_col.set_aspect('equal')
            ax_col.set_adjustable('box')
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, hspace=0, wspace=0)
    return fig


def plot_track_queries(queries_points: np.ndarray, queries_weights: np.ndarray):
    """Visualize track queries as scatter plot.
    
    Args:
        queries_points: [num_queries, num_frame, num_data, 2]
        queries_weights: [num_queries, num_frame, num_data]
    """
    ncols = 5
    nrows = np.ceil(queries_points.shape[0] / ncols).astype(int)
    colormap = LinearSegmentedColormap.from_list('mycamp', ['b', 'g', 'r'])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(10, 4), dpi=160, layout='constrained')
    for ax, qu_points, qu_weights in zip(axs.flatten(), queries_points, queries_weights):
        track_weights = np.mean(qu_weights, axis=0)
        idx = np.argsort(track_weights)
        coefs = np.stack([np.polyfit(points[:, 0], points[:, 1], 1) 
                          for points in qu_points.transpose(1, 0, 2)])
        mv_coef_a = 1 / (1 + np.exp(coefs[idx, 0]))
        mv_coef_b = coefs[idx, 1]
        ax.scatter(mv_coef_a, mv_coef_b, c=track_weights[idx], 
                           s=2, cmap=colormap, vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, hspace=0, wspace=0)
    return fig


def plot_pr_curve(precisions: np.ndarray, recalls:np.ndarray, area:float,):
    fig, ax =plt.subplots()
    
    mask = ~(np.isnan(precisions) | np.isnan(recalls))
    recalls, precisions = recalls[mask], precisions[mask]
    ax.plot(recalls, precisions, '.--')
    ax.fill_between(recalls, precisions, alpha=0.4)
    
    ax.set_xlabel('recall')
    ax.set_xlim((0, 1))
    ax.set_ylabel('precision')
    ax.set_ylim((0, 1))
    ax.set_title(f'AP {area:.2f}')
    ax.grid()
    return fig


def plot_pr_mot_curve(precisions: np.ndarray, recalls:np.ndarray, mots: np.ndarray,
                      title:str = 'PR MOT'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    mask = ~(np.isnan(precisions) | np.isnan(recalls) | np.isnan(mots))
    recalls, precisions, mots = recalls[mask], precisions[mask], mots[mask]
    ax.plot3D(recalls, precisions, mots, '.--')
    
    ax.set_xlabel('recall')
    ax.set_xlim((0, 1))
    ax.set_ylabel('precision')
    ax.set_ylim((0, 1))
    ax.set_zlabel('mot_meter')
    ax.set_zlim((0, 1))
    ax.set_title(title)
    ax.view_init(18, 235)
    return fig


class ImageConvert(object):
    """A lot of image convert helper functions."""
    @staticmethod
    def to_pil_image(pic) -> Image.Image:
        if isinstance(pic, Tensor):
            return F.to_pil_image(pic)
        elif isinstance(pic, np.ndarray):
            return convert_mat2pil(pic)
        else:
            raise ValueError("Invalid type of pic to pil.")
    
    @staticmethod
    def to_mat_image(pic) -> np.ndarray:
        if isinstance(pic, Tensor):
            return convert_pil2mat(F.to_pil_image(pic))
        elif isinstance(pic, Image.Image):
            return convert_pil2mat(pic)
        else:
            raise ValueError("Invalid type of pic to mat.")
        
    @staticmethod
    def to_tensor(pic) -> Tensor:
        if isinstance(pic, Image.Image):
            return F.to_tensor(pic)
        elif isinstance(pic, np.ndarray):
            return F.to_tensor(convert_mat2pil(pic))
        else:
            raise ValueError("Invalid type of pic to tensor.")


def convert_pil2mat(img: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def convert_mat2pil(mat: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(mat, cv.COLOR_BGR2RGB))


def cv_rectangle(mat, pt1, pt2, color, thickness=1, style='', alpha=1.):
    """Extend opencv rectangle function.

    Args:
        mat (np.array): The Mat image.
        pt1 ([x, y]): The left-top corner point.
        pt2 ([x, y]): The right-bottom corner point.
        color (list): BGR Color of the rectangle.
        thickness (int, optional): Thickness of the rectangle. Defaults to 1.
        style (str, optional): Style of the rectangle with 3 options.`dashed` is draw dashed line of rectangle, `dotted` is draw dotted line of rectangle, `''` is norm rectangle. Defaults to ''.
        alpha (float, optional): Alpha of the rectangle. Defaults to `1.`.
    """
    if pt1[0] == pt2[0] or pt1[1] == pt2[1]:
        return
    overlay = mat.copy()
    if style in ('dashed', 'dotted'):
        pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
        drawpoly(overlay, pts, color, thickness, style)
    else:
        cv.rectangle(overlay, pt1, pt2, color, thickness)
    cv.addWeighted(overlay, alpha, mat, 1 - alpha, 0, mat)


def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv.line(img,s,e,color,thickness)
            i+=1


class RCG(object):
    """random color generator"""
    def __init__(self, max_len = 1000):
        self.color_map = OrderedDict()
        self.max_len = max_len
    
    def _random_generator(self, id):
        color = tuple(np.uint8(np.random.choice(range(256), size=3)))
        while len(self.color_map) >=  self.max_len:
            self.color_map.popitem(last=False)
        self.color_map[id] = color
    
    def __call__(self, id):
        if id not in self.color_map.keys():
            self._random_generator(id)
        return self.color_map[id]

