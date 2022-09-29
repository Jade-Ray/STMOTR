import warnings
from typing import List
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torchvision.transforms.functional as F
import numpy as np
from scipy import interpolate
import cv2 as cv

import utils.logging as logging
from datasets.transforms import NormalizeInverse
from utils.meters import MotValMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def plot_inputs_as_video(inputs: Tensor, boxes: List[Tensor], 
                         frameids: List[Tensor], 
                         norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]) -> Tensor:
    """
    Normalize Inverse inputs tensor, and draw Boundingboxes on images, return video tensor of (B, T, C, H, W).
    """
    assert len({inputs.shape[0], len(boxes), len(frameids)}) == 1
    normalizeInverse = NormalizeInverse(norm_mean, norm_std)
    B, C, T, H, W,  = inputs.shape
    
    outputs = []
    for batch_img, batch_boxes, batch_frameids in zip(inputs, boxes, frameids):
        batch_img, batch_boxes = normalizeInverse(batch_img.transpose(0, 1), 
                                                  boxes=batch_boxes.transpose(0, 1))
        for img, box_list, frameid in zip(batch_img, batch_boxes.numpy(), batch_frameids.numpy()):
            mat = ImageConvert.to_mat_image(img)
            cv.putText(mat, f'{frameid}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
            for l, t, r, b in box_list.astype(int):
                cv_rectangle(mat, (l,t), (r,b), (0, 255, 0), thickness=2)
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
                       save_video=False, output_dir='', 
                       plot_interval=(1, 100), resize=(960, 540)) -> Tensor:
    """
    Plot MotValMeter final predication, return video tensor of (B, T, C, H, W).
    """
    outputs = []
    parser = base_ds(sequence_name=sequence_name)
    plot_interval = range(*plot_interval)
    
    if save_video:
        writer = cv.VideoWriter(f'{output_dir}/{sequence_name}.avi', 
                                cv.VideoWriter_fourcc(*'MJPG'), 10, 
                                (parser.imWidth, parser.imHeight))
    else:
        writer = None
    
    for frameid in meter.frameids:
        if frameid not in plot_interval and writer is None:
            continue
        mat = ImageConvert.to_mat_image(parser.get_image(frameid))
        cv.putText(mat, f'{sequence_name}: {frameid}', (5,40), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
        mot_events = meter.get_frame_events(frameid)
        summary = meter.get_summary(end_frameid=frameid)
        text = f"mota:{summary['mota'][0]:.2f} motp:{summary['motp'][0]:.2f} nfp:{summary['num_false_positives'][0]} nmis:{summary['num_misses'][0]} nsw:{summary['num_switches'][0]}"
        cv.putText(mat, text, (5,parser.imHeight-5), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
            
        for Type, OId, HId, Dis in mot_events.values:
            o_l, o_t, o_r, o_b = meter.get_box(frameid, OId, mode='gt', format='xyxy')
            h_l, h_t, h_r, h_b = meter.get_box(frameid, HId, mode='pred', format='xyxy')
            if Type == 'MATCH':
                cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (0, 255, 0), 2)
            elif Type == 'FP':
                cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (0, 0, 255), 2, style='dashed')
            elif Type == 'MISS':
                cv_rectangle(mat, (o_l,o_t), (o_r,o_b), (0, 0, 255), 2)
            elif Type == 'SWITCH':
                cv_rectangle(mat, (o_l,o_t), (o_r,o_b), (255, 0, 0), -1, alpha=0.2)
                cv_rectangle(mat, (h_l,h_t), (h_r,h_b), (255, 0, 0), -1, alpha=0.2)
                cv.arrowedLine(mat, (o_l,o_t), (h_l,h_t), (255, 0, 0), 1)
                cv.arrowedLine(mat, (o_r,o_b), (h_r,h_b), (255, 0, 0), 1)

        if frameid in plot_interval:
            output = cv.resize(mat, resize)
            outputs.append(ImageConvert.to_tensor(output))
        if writer is not None:
            writer.write(mat)
    outputs = torch.stack(outputs, dim=0)
    
    if writer is not None:
        writer.release()
        logger.info(f'Pred Video {output_dir}/{sequence_name}.avi saving done.')

    return outputs[None]


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


def plot_deformable_attention_weights(attn_weights: np.ndarray, attn_points: np.ndarray,
                                      spatial_shapes: np.ndarray,
                                      refer_points: np.ndarray, expand_points: np.ndarray, 
                                      pil_imgs: List[Image.Image], boxes: np.ndarray, 
                                      query_id: int, frame_ids: np.ndarray):
    """Visualize Deformable DETR encoder-decoder multi-head attention weights."""
    assert len({len(pil_imgs), len(frame_ids), len(boxes)}) == 1
    assert len({len(attn_weights), len(attn_points), len(refer_points), 
                len(expand_points), len(spatial_shapes)}) == 1
    ncols, nrows = len(pil_imgs) + 1, len(attn_weights)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(25, 14))
    for nl, (ax_j, weight, point, shape, refer_point, expand_point) in enumerate(zip \
        (axs, attn_weights, attn_points, spatial_shapes, refer_points, expand_points)):
        ax = ax_j[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = interpolate.interp2d(point[:, 0], point[:, 1], weight)
            ax.imshow(f(np.arange(0, shape[0], 1), np.arange(0, shape[1], 1)))
        ax.plot(refer_point[0], refer_point[1], 'k+', markersize=8)
        ax.axis('off')
        ax.set_title(f'level: {nl+1}')
        
        for nf, (img, (l, t, r, b), f_point, frameid) in enumerate(zip \
            (pil_imgs, boxes, expand_point, frame_ids)):
            ax = ax_j[nf+1]
            ax.imshow(img)
            ax.add_patch(plt.Rectangle((l, t), r -l, b - t, fill=False, color='blue', linewidth=2))
            ax.scatter(f_point[:, 0], f_point[:, 1], s=[6]*len(weight), c=weight)
            ax.plot(l+(r-l)/2, t+(b-t)/2, 'g+', markersize=8)
            ax.axis('off')
            ax.set_title(f'frame id: {frameid} // query id: {query_id}')
    fig.tight_layout()
    return fig


def plot_pr_curve(recalls:np.ndarray, precisions: np.ndarray, area:float=None,
                  ori_recalls=None, ori_precisions=None):
    fig, ax =plt.subplots()
    ax.plot(recalls, precisions, '-')
    ax.fill_between(recalls, precisions, alpha=0.4, 
                    label=f'area: {area:.2f}' if area is not None else 'area')
    if ori_recalls is not None and ori_precisions is not None:
        ax.plot(ori_recalls, ori_precisions, 'o')
    
    ax.legend(loc='upper right')
    ax.set_xlabel('recall')
    ax.set_xlim((0, 1))
    ax.set_ylabel('precision')
    ax.set_ylim((0, 1))
    ax.grid()
    return fig


def plot_pr_mota_curve(recalls:np.ndarray, precisions: np.ndarray, motas: np.ndarray,
                       area:float=None,):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.plot(recalls, precisions, motas, label='PR_MOTA curve')
    ax.legend()
    ax.set_xlabel('recall')
    ax.set_xlim((0, 1))
    ax.set_ylabel('precision')
    ax.set_ylim((0, 1))
    ax.set_zlabel('mota')
    ax.set_zlim((-1, 1))
    
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
    if pt1 == pt2:
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
