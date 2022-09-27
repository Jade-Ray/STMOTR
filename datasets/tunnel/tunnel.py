from typing import List, Optional, Callable, Union
from collections import defaultdict
from pathlib import Path
import math
from configparser import ConfigParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import datasets.transforms as T
from utils.misc import nested_tensor_from_videos_list


class SingleVideoParser():
    def __init__(self, mot_file_path: Path, subset_type: str = 'train',
                 num_frames: int = 8, sampling_rate: int = 2, random_sampling: bool = False,
                 overlap_frame: int = 1):
        self.subset_type = subset_type
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.random_sampling = random_sampling
        self.overlap_frame = overlap_frame
        
        self._read_seq_info(mot_file_path)
        self._read_mot_file()
    
    @property
    def selecte_frame_scale(self) -> int:
        """The total number of frames sampled."""
        return self.num_frames * self.sampling_rate
    
    @property
    def selecte_frame_diff(self) -> int:
        """The difference value between the min&max frames sampled."""
        return self.selecte_frame_scale - 1
    
    @property
    def min_valid_node_num(self) -> int:
        """The min valid node number in one series of frames"""
        return 0
    
    @property
    def visible_thresh(self):
        """The visible threshold of object"""
        return 1 - 1.
    
    @property
    def start_frame_id(self) -> int:
        r"""Return the start frame id of video"""
        return 1
    
    @property
    def end_frame_id(self) -> int:
        r"""Return max frame index of parsed video data"""
        return self.gt['frame_index'].max()
    
    def _read_seq_info(self, mot_file_path: Path):
        """Reading the mot sequence information from seqinfo.ini"""
        mot_config = ConfigParser()
        mot_config.read(mot_file_path / 'seqinfo.ini')
        
        self.sequence_name = mot_config.get('Sequence', 'name')
        self.imDir = mot_file_path / mot_config.get('Sequence', 'imDir')
        self.frameRate = int(mot_config.get('Sequence', 'frameRate'))
        self.seqLength = int(mot_config.get('Sequence', 'seqLength'))
        self.imWidth = int(mot_config.get('Sequence', 'imWidth'))
        self.imHeight = int(mot_config.get('Sequence', 'imHeight'))
        self.imExt = mot_config.get('Sequence', 'imExt')
        self.labelDir = mot_file_path / 'gt/gt.txt'
    
    def _read_mot_file(self):
        columns = ["frame_index", "track_id", "object_type", "l", "t", "r", "b", 'visibility']
        df = pd.read_csv(self.labelDir, names=columns)
        
        def ltrb2wh(row):
            return row['l'] - row['r'], row['t'] - row['b']
        df['w'], df['h'] = zip(*df.apply(ltrb2wh, axis=1))
        
        self.gt = df
    
    def len(self):
        """The number of parsed video frames."""
        if self.subset_type == 'train':
            return self.seqLength - self.selecte_frame_diff
        else:
            # for val with config frame overlap
            return math.ceil((self.seqLength - self.overlap_frame) / 
                             (self.selecte_frame_scale - self.overlap_frame))
    
    def get_images(self, frame_ids: List[int]) -> List[Image.Image]:
        """Return indicted images list selected by frame_ids."""
        return [self.get_image(i) for i in frame_ids]
    
    def get_image(self, frame_id: int) -> Image.Image:
        """Return indicted image selected by frame_id."""
        return Image.open(self.imDir / f'{frame_id:06}{self.imExt}')
    
    def get_gt(self, frame_ids: List[int]) -> pd.DataFrame:
        df = self.gt[self.gt['frame_index'].isin(frame_ids)]
        
        # filited the number of track in selected frame less than threshold
        appeared_track_counts = df[df['visibility'] >= self.visible_thresh]['track_id'].value_counts()
        selected_track_id = appeared_track_counts[appeared_track_counts >= self.min_valid_node_num].index
        return df[df['track_id'].isin(selected_track_id)]
    
    def convert2mate(self, df_gt: pd.DataFrame) -> dict:
        frame_indexes = df_gt['frame_index'].unique()
        video_mate = defaultdict(list)
        video_mate['frame_ids'] = frame_indexes
        for track_id, track_group in df_gt.groupby('track_id'):
            video_mate['track_ids'].append(track_id)
            video_mate['labels'].append(int(track_group['object_type'].mode()))
            referred, bboxes, vises = [], [], []
            for i in frame_indexes:
                if i in track_group['frame_index'].values:
                    referred.append(True)
                    bboxes.append(track_group.loc[track_group['frame_index'] == i, ['l', 't', 'r', 'b']].values[0])
                    vises.append(track_group.loc[track_group['frame_index'] == i, 'visibility'].values[0])
                else:
                    referred.append(False)
                    bboxes.append(np.zeros(4, dtype=float))
                    vises.append(0.)
            video_mate['referred'].append(np.array(referred))
            video_mate['boxes'].append(np.array(bboxes))
            video_mate['visibilities'].append(np.array(vises))
        video_mate = {k: np.array(v) for k, v in video_mate.items()}
        video_mate['orig_size'] = (self.imWidth, self.imHeight)
        video_mate['video_name'] = self.sequence_name
        return video_mate
    
    def get_frame_ids(self, item:int):
        """get series frames indexes with item."""
        begin_index = self._get_begin_index(item)
        frame_ids = np.arange(begin_index, begin_index + self.selecte_frame_scale) # [strat, end)
        return frame_ids
    
    def _get_sampling_frame_ids(self, item:int):
        """get sampling series frames indexes with item."""
        frame_ids = self.get_frame_ids(item)
        if self.random_sampling and self.subset_type == 'train':
            return np.sort(np.random.choice(
                frame_ids, size=self.num_frames, replace=False))
        else:
            return frame_ids[::self.sampling_rate]
    
    def _get_begin_index(self, item):
        """Get series frames begin index from data item."""
        if self.subset_type == 'train':
            return item + self.start_frame_id
        else:
            begin_index = item * (self.selecte_frame_scale - self.overlap_frame) + self.start_frame_id
            # if end_index over max frame, fit it.
            if begin_index + self.selecte_frame_diff > self.end_frame_id:
                return self.end_frame_id - self.selecte_frame_diff
            else:
                return begin_index 
    
    def __len__(self):
        return self.len()       

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
        video_mate = self.convert2mate(self.get_gt(frame_ids))
        assert len({len(images), video_mate['boxes'].shape[1], video_mate['referred'].shape[1]}) == 1
        
        return images, video_mate


class Tunnel(Dataset):
    def __init__(self, subset_type: str = 'train', dataset_path: str ='./data/Tunnel', sampling_num: int = 8, sampling_rate: int = 2, **kwargs):
        super(Tunnel, self).__init__()
        assert subset_type in ['train', 'test'], "error, unsupported dataset subset type. use 'train' or 'test'."
        self.subset_type = subset_type
        self.dataset_path = Path(dataset_path)
        self._load_data_from_sequence_list(sampling_num, sampling_rate)
        self.transform = TunnelTransforms(subset_type, **kwargs)
        self.collator = Collator(subset_type)
        
    def _load_data_from_sequence_list(self, sampling_num, sampling_rate):
        sequence_file = Path(__file__).parent / f'sequence_list_{self.subset_type}.txt'
        data_folder = self.dataset_path / self.subset_type
        sequence_file_list = np.loadtxt(sequence_file, dtype=str)
        sequence_file_list = sequence_file_list if sequence_file_list.ndim > 0 else [sequence_file_list]
        
        files_path = data_folder.glob('K258-*')
        files_selected_path = [file for file in files_path if file.stem in sequence_file_list]
        
        # load all the mot files
        self.data = []
        pbar = tqdm(files_selected_path)
        for file_path in pbar:
            pbar.set_description(f'reading: {file_path}')
            self.data += [SingleVideoParser(
                file_path, subset_type=self.subset_type, 
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
    

class TunnelTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # size is slightly smaller than eval size below to fit in GPU memory
        transforms = []
        if horizontal_flip_augmentations and subset_type == 'train':
            transforms.append(T.RandomHorizontalFlip())
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'valid' or subset_type == 'test':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.transforms = T.Compose(transforms)
    
    def __call__(self, imgs, video_mate):
        num_frame = len(imgs)
        targets = {
            'boxes': torch.tensor(video_mate['boxes']).view(-1, num_frame, 4),
            'referred': torch.tensor(video_mate['referred']).view(-1, num_frame),
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
