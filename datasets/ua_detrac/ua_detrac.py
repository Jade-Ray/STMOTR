from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from datasets.video_parse import SingleVideoParserBase
import datasets.transforms as T
from utils.misc import nested_tensor_from_videos_list
from utils.box_ops import box_xywh_to_xyxy, box_iou
import utils.logging as logging

logger = logging.get_logger(__name__)

UA_CLASSES = {1: 'car', 2: 'van', 3: 'bus', 4: 'others'}


def get_object_type_num(type_name: str):
    for k, v in UA_CLASSES.items():
        if type_name == v:
            return k
    return 4 # default return others


class IgnoredRegion():
    """Record Ignored Region and some helpers functions.
    
    Args:
        ignored_regions (list[tuple]): The Ignored Region format is the list and a rectangle (x, y, w, h) in item.
        threshold (float): The threshold `(0.~1.)` of iou to detect weather obj in ignored_regions, 0 is overlap, 1 is non-overlap.
    """
    def __init__(self, ignored_regions: list[tuple], threshold: float = 0.1):
        self.ignored_regions = ignored_regions
        self.threshold = threshold
    
    @property
    def regions(self) -> np.ndarray:
        """Return a ignored region list with rectangle (x, y, w, h)"""
        return np.asfarray(self.ignored_regions)
    
    def in_ignored_region(self, objs):
        """Weather objs in ignored regions

        Args:
            objs (list[tuple] or tuple): The obj rectangles (x,y,w,h) in rows.

        Returns:
            result (list[bool] or bool): The list bool or single bool of weather objs in ignored regions 
        """
        if not isinstance(objs, list):
            objs = [objs]
        objs = np.asfarray(objs)
        if self.regions.size == 0 or objs.size == 0:
            result = [False for _ in objs]
        else:
            iou = box_iou(box_xywh_to_xyxy(objs), box_xywh_to_xyxy(self.regions), 
                           first_union=True).numpy()
            result = iou.sum(-1) > self.threshold
        return result[0] if len(objs) == 1 else result
    
    def plot_regions(self, pil_img: Image.Image, fill=(0,0,0), opacity=1.):
        imgcp = pil_img.copy()
        if len(fill) == 3:
            fill += (int(255 * opacity),)
        draw = ImageDraw.Draw(imgcp, "RGBA")
        for region in box_xywh_to_xyxy(self.regions.astype(int)):
            draw.rectangle(tuple(region), fill=fill)
        return imgcp


class SingleVideoParser(SingleVideoParserBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _read_seq_info(self, mot_file_path: Path):
        """Reading the mot sequence information from xml file"""
        
        self.sequence_name = mot_file_path.stem
        self.imDir = mot_file_path.parents[1] / 'Insight-MVT_Annotation' / self.sequence_name
        self.frameRate = 25.0
        self.seqLength = len(list(self.imDir.glob('*.jpg')))
        self.imWidth = 960
        self.imHeight = 540
        self.imExt = '.jpg'
        self.labelDir = mot_file_path
    
    def _read_mot_file(self):
        """Reading the xml mot_file. Returns a pd"""
        xtree = ET.parse(self.labelDir)
        xroot = xtree.getroot()
        
        ignored_regions = []
        ignored_region = xroot.find('ignored_region')
        for box in ignored_region.findall('box'):
            l, t, w, h = float(box.get("left")), float(box.get("top")), float(box.get("width")), float(box.get("height"))
            ignored_regions.append((l, t, w, h))
        self.ignored_region = IgnoredRegion(ignored_regions)
        
        columns = [
            "frame_index", "track_id", "l", "t", "r", "b", "w", "h",
            "confidence", "object_type", 'visibility']
        converted_data = []
        for frame in xroot.findall('frame'):
            frame_index = int(frame.get('num'))
            for target in frame.find('target_list').findall('target'):
                track_id = int(target.get("id"))
                l, t, w, h = float(target[0].get("left")), float(target[0].get("top")), float(target[0].get("width")), float(target[0].get("height"))
                r, b = l + w, t + h

                object_type = get_object_type_num(target[1].get("vehicle_type"))
                overlap_ratio = float(target[1].get("truncation_ratio"))
                in_ignored_region = self.ignored_region.in_ignored_region((l,t,w,h))

                converted_data += [[
                    frame_index, track_id, l, t, r, b, w, h,
                    1 - in_ignored_region, object_type, 1-overlap_ratio]]
        df = pd.DataFrame(converted_data, columns=columns)
        
        self.gt = df
    
    def get_image(self, frame_id: int, fill_ignored_region: bool = True, 
                  fill=(0,0,0), opacity=1.) -> Image.Image:
        """Return indicted image selected by frame_id."""
        pil_img = Image.open(self.imDir / f'img{frame_id:05}{self.imExt}')
        if fill_ignored_region:
            return self.ignored_region.plot_regions(pil_img, fill=fill, opacity=opacity)
        return pil_img
    
    def convert2mate(self, frame_ids) -> dict:
        df_gt = self.get_gt(frame_ids)
        video_mate = defaultdict(list)
        key = ['track_ids', 'labels', 'boxes', 'confidences']
        for track_id, track_group in df_gt.groupby('track_id'):
            video_mate['track_ids'].append(track_id)
            video_mate['labels'].append(int(track_group['object_type'].mode()))
            bboxes, confidences = [], [], []
            for i in frame_ids:
                if i in track_group['frame_index'].values:
                    bboxes.append(track_group.loc[track_group['frame_index'] == i, ['l', 't', 'r', 'b']].values[0])
                    confidences.append(track_group.loc[track_group['frame_index'] == i, 'confidence'].values[0])
                else:
                    bboxes.append(np.zeros(4, dtype=float))
                    confidences.append(0)
            video_mate['boxes'].append(np.array(bboxes))
            video_mate['confidences'].append(np.array(confidences))
        video_mate = {k: np.array(video_mate[k]) for k in key}
        video_mate['boxes'].reshape(-1, len(frame_ids), 4)
        video_mate['confidences'].reshape(-1, len(frame_ids))
        video_mate['frame_ids'] = frame_ids
        video_mate['orig_size'] = (self.imWidth, self.imHeight)
        video_mate['video_name'] = self.sequence_name
        return video_mate   

    def __getitem__(self, item: int):
        # create Object dict, containing frames of dim [frame_num, PIL.Image], 
        # bboxes with absolute coord of dim [track_num, frame_num, 4], 
        # visibilities with 0~1 of dim [track_num, frame_num], 
        # classification of targets of dim [track_num],
        # motions of dim [track_num, 4, num_degree]
        # orig_size of frames origin size of [w, h]
        # padding empty object if no obj in selected_frame
        frame_ids = self._get_sampling_frame_ids(item)
        images = self.get_images(frame_ids, opacity=0.)
        video_mate = self.convert2mate(frame_ids)
        
        return images, video_mate


class UADETRAC(Dataset):
    def __init__(self, subset_type: str = 'train', dataset_path: str ='./data/UA_DETRAC', 
                 sampling_num: int = 8, sampling_rate: int = 2, **kwargs):
        """UA_DETRAC Dataset"""
        super(UADETRAC, self).__init__()
        assert subset_type in ['train', 'test'], "error, unsupported dataset subset type. use 'train' or 'test'."
        self.subset_type = subset_type
        self.dataset_path = Path(dataset_path)
        self._load_data_from_sequence_list(sampling_num, sampling_rate)
        self.transform = UADETRACTransforms(subset_type, **kwargs)
        self.collator = Collator(subset_type)
        
    def _load_data_from_sequence_list(self, sampling_num, sampling_rate):
        sequence_file = Path(__file__).parent / f'sequence_list_{self.subset_type}.txt'
        assert sequence_file.exists()
        
        if self.subset_type == 'train':
            data_folder = self.dataset_path / 'train' / 'DETRAC-Annotations-XML'
        else:
            data_folder = self.dataset_path / 'test' / 'DETRAC-Annotations-XML'

        sequence_file_list = np.loadtxt(sequence_file, dtype=str)
        sequence_file_list = sequence_file_list if sequence_file_list.ndim > 0 else [sequence_file_list]
        
        files_path = data_folder.glob('MVI_[0-9][0-9][0-9][0-9][0-9].xml')
        files_selected_path = [file for file in files_path if file.stem in sequence_file_list]
        assert len(files_selected_path) > 0
        
        # load all the mot files
        self.data = []
        pbar = tqdm(files_selected_path)
        for file_path in pbar:
            pbar.set_description(f'reading: {file_path}')
            self.data += [SingleVideoParser(
                mot_file_path=file_path, subset_type=self.subset_type, 
                num_frames=sampling_num, sampling_rate=sampling_rate)]
            
        # Compute some basic information from data
        self.lens = [len(p) for p in self.data]
        start_index = np.cumsum([0] + self.lens)
        self.ranges = [(start_index[i], start_index[i+1]) for i in range(len(start_index)-1)]
    
    def get_parser_from_item(self, item):
        """Get video parser and its item from dataset item."""
        for i, (start, end) in enumerate(self.ranges):
            if item >= start and item < end:
                return self.data[i], item - start
    
    def get_parser_from_name(self, video_name: str):
        """Get video parser with name from all dataset video parsers"""
        for video_parser in self.data:
            if video_name == video_parser.sequence_name:
                return video_parser
        raise ValueError(f"{video_name} is not parser in Dataset.")
    
    def __len__(self):
        return np.sum(self.lens)
    
    def __getitem__(self, item):
        # locate the parser
        assert item < self.__len__(), f'the item of dataset must less than length {self.__len__()}, but get {item}'
        video_parser, video_parser_item = self.get_parser_from_item(item)
        imgs, video_mate = video_parser[video_parser_item]
        if self.transform is not None:
            imgs, video_mate = self.transform(imgs, video_mate)
        video_mate['item'] = torch.tensor(item)
        return imgs, video_mate
    

class UADETRACTransforms:
    def __init__(self, subset_type, color_jitter_aug=False, rand_crop_aug=False, **kwargs):
        
        normalize = T.Compose([
            T.MotToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        scales = [224, 240, 256, 272, 288, 304, 320, 336, 352, 368]
        
        if subset_type == 'train':
            if color_jitter_aug:
                logger.info('Training with RandomColorJitter.')
                color_transforms = [
                    T.RandomApply([T.MotColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0),]),
                ]
            if rand_crop_aug:
                logger.info('Training with RandomCrop.')
                scale_transforms = [
                    T.MotRandomFrozenTime(),
                    T.MotRandomHorizontalFlip(),
                    T.RandomSelect(
                        T.MotRandomResize(scales, max_size=655),
                        T.Compose([
                            T.MotRandomResize([140, 170, 200]),
                            T.FixedMotRandomCrop(112, 200),
                            T.MotRandomResize(scales, max_size=655),
                        ])
                    ),
                    T.MotConfidenceFilter(),
                    normalize,
                ]
            else:
                scale_transforms = [
                    T.MotRandomFrozenTime(),
                    T.MotRandomHorizontalFlip(),
                    T.MotRandomResize(scales, max_size=655),
                    T.MotConfidenceFilter(),
                    normalize,
                ]
            
            self.transforms = T.Compose(color_transforms + scale_transforms)
        
        elif subset_type == 'val' or subset_type == 'test':
            self.transforms = T.Compose([
                T.MotRandomResize([368], max_size=655),
                normalize,
            ])
        else:
            raise ValueError(f'Unknow {subset_type} transform strategy.')
    
    def __call__(self, imgs, video_mate):
        num_frame = len(imgs)
        targets = {
            'boxes': torch.tensor(video_mate['boxes']).view(-1, num_frame, 4),
            'referred': torch.tensor(video_mate['confidences']).view(-1, num_frame),
            'labels': torch.tensor(video_mate['labels']),
            'orig_size': torch.tensor(video_mate['orig_size']),
            'frame_indexes': torch.tensor(video_mate['frame_ids'])
        }
        return self.transforms(imgs, targets)


class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type
    
    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_videos_list(batch[0]) # [T, B, C, H, W]
        return tuple(batch)
