from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from datasets.video_parse import SingleVideoParserBase
import datasets.transforms as T
from utils.misc import nested_tensor_from_videos_list
import utils.logging as logging

logger = logging.get_logger(__name__)

MOT20_CLASSES = {
    1: 'Pedestrain', 
    2: 'Person on vehicle', 
    3: 'Car', 
    4: 'Bicycle',
    5: 'Motorbike', 
    6: 'None motorrized vehicle', 
    7: 'Static person',
    8: 'Distractor',
    9: 'Occluder',
    10: 'Occluder on the ground',
    11: 'Occluder full',
    12: 'reflection',
    13: 'Crowd',
}


class SingleVideoParser(SingleVideoParserBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _read_mot_file(self):
        if self.labelDir.exists():
            columns = [
                "frame_index", "track_id", "l", "t", "w", "h",
                "confidence", "object_type", 'visibility']
            df = pd.read_csv(self.labelDir, names=columns)
            
            def ltwh2rb(row):
                return row['l'] + row['w'], row['t'] + row['h']
            df['r'], df['b'] = zip(*df.apply(ltwh2rb, axis=1))
            
            self.gt = df
        else:
            self.gt = None
    
    def convert2mate(self, frame_ids) -> dict:
        video_mate = defaultdict(list)
        if self.gt is not None:
            df_gt = self.get_gt(frame_ids)
            key = ['track_ids', 'labels', 'boxes', 'confidences']
            for track_id, track_group in df_gt.groupby('track_id'):
                video_mate['track_ids'].append(track_id)
                video_mate['labels'].append(int(track_group['object_type'].mode()))
                bboxes, confidences = [], []
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
        images = self.get_images(frame_ids)
        video_mate = self.convert2mate(frame_ids)
        
        return images, video_mate


class MOT20(Dataset):
    def __init__(self, subset_type: str = 'train', dataset_path: str ='./data/MOT20', 
                 sampling_num: int = 8, sampling_rate: int = 2, **kwargs):
        super(MOT20, self).__init__()
        assert subset_type in ['train', 'val', 'test'], "error, unsupported dataset subset type. use 'train' or 'val' or 'test'."
        self.subset_type = subset_type
        self.dataset_path = Path(dataset_path)
        self._load_data_from_sequence_list(sampling_num, sampling_rate)
        self.transform = MOT20Transforms(subset_type, **kwargs)
        self.collator = Collator(subset_type)
        
    def _load_data_from_sequence_list(self, sampling_num, sampling_rate):
        sequence_file = Path(__file__).parent / f'sequence_list_{self.subset_type}.txt'
        assert sequence_file.exists()
        
        if self.subset_type == 'train' or self.subset_type == 'val':
            data_folder = self.dataset_path / 'train'
        else:
            data_folder = self.dataset_path / 'test'
        
        sequence_file_list = np.loadtxt(sequence_file, dtype=str)
        sequence_file_list = sequence_file_list if sequence_file_list.ndim > 0 else [sequence_file_list]
        
        files_path = data_folder.glob('MOT20-*')
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
    

class MOT20Transforms:
    def __init__(self, subset_type, color_jitter_aug=False, rand_crop_aug=False, **kwargs):
        normalize = T.Compose([
            T.MotToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        scales = [480, 512, 544, 576, 608, 640]
        
        if subset_type == 'train':
            color_transforms = []
            if color_jitter_aug:
                logger.info('Training with RandomColorJitter.')
                color_transforms.append(
                    T.RandomApply([T.MotColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0),]),
                )
            if rand_crop_aug:
                logger.info('Training with RandomCrop.')
                scale_transforms = [
                    T.MotRandomHorizontalFlip(),
                    T.RandomSelect(
                        T.MotRandomResize(scales, max_size=1333),
                        T.Compose([
                            T.MotRandomResize([333, 416, 500]),
                            T.FixedMotRandomCrop(320, 500),
                            T.MotRandomResize(scales, max_size=1333),
                        ]),
                    ),
                    T.MotConfidenceFilter(),
                    normalize,
                ]
            else:
                scale_transforms = [
                    T.MotRandomHorizontalFlip(),
                    T.MotRandomResize(scales, max_size=800),
                    T.MotConfidenceFilter(),
                    normalize,
                ]
            
            self.transforms = T.Compose(color_transforms + scale_transforms)
        
        elif subset_type == 'val' or subset_type == 'test':
            self.transforms = T.Compose([
                T.MotRandomResize([640], max_size=800),
                normalize,
            ])
        else:
            raise ValueError(f'Unknow {subset_type} transform strategy.')
    
    def __call__(self, imgs, video_mate):
        num_frame = len(imgs)
        targets = {
            'orig_size': torch.tensor(video_mate['orig_size']),
            'frame_indexes': torch.tensor(video_mate['frame_ids'])
        }
        if 'boxes' in video_mate:
            targets.update({
                'boxes': torch.tensor(video_mate['boxes']).view(-1, num_frame, 4),
                'referred': torch.tensor(video_mate['confidences']).view(-1, num_frame),
                'labels': torch.tensor(video_mate['labels']),
            })
        return self.transforms(imgs, targets)


class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type
    
    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_videos_list(batch[0]) # [T, B, C, H, W]
        return tuple(batch)
