from typing import List
from pathlib import Path
import math
from configparser import ConfigParser

import numpy as np
import pandas as pd
from PIL import Image


class SingleVideoParserBase(object):
    """Parse a single video file to a series of frames and targets. You should implement the func `_read_mot_file` and `convert2mate`.
    
    Args:
        mot_file_path (Path): The file path recording `multi-obj-track` info file path.
        subset_type (str): Options includes `train` or `test` mode. For the train set, select samples one by one until the max frame number of data. For the test set, select order samples with the overlap frame between successive samples. Default `train`.
        mun_frames (int): The number of clip frames to parser. Default `8`.
        sampling_rate (int): The video sampling rate of the input clip. Default `2`.
        random_sampling (bool): Whether to randomly sample the clip. Default `False`.
        overlap_frame (int): The overlap frame between continuous iters in `test` type. Default `1`.
        min_node_num (int): The minimum number of valid nodes in the clip. Default `1`.
        max_overlap (float): The maximum overlap rate threshold of objects. Default `1.0`.
        start_frame_id (int): The start frame id in video. Default `1`.
    """

    def __init__(self, mot_file_path: Path, subset_type: str = 'train',
                 num_frames: int = 8, sampling_rate: int = 2, 
                 random_sampling: bool = False,
                 overlap_frame: int = 1, min_node_num: int = 1,
                 max_overlap: float = 1.0, start_frame_id: int = 1):
        self.subset_type = subset_type
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.random_sampling = random_sampling
        self.overlap_frame = overlap_frame
        self._min_node_num = min_node_num
        self._max_overlap = max_overlap
        self._start_frame_id = start_frame_id
        
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
        return self._min_node_num
    
    @property
    def align_overlap_num(self) -> int:
        """The number of align overlap frames"""
        return self.overlap_frame * self.sampling_rate
    
    @property
    def visible_thresh(self):
        """The visible threshold of object"""
        return 1 - self._max_overlap
    
    @property
    def start_frame_id(self) -> int:
        r"""Return the start frame id of video"""
        return self._start_frame_id
    
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
        """Reading the xml mot_file. Returns a pd."""
        raise NotImplementedError
    
    def len(self):
        """The number of parsed video frames."""
        if self.subset_type == 'train':
            return self.seqLength - self.selecte_frame_diff
        else:
            # for val with config frame overlap
            return math.ceil((self.seqLength - self.align_overlap_num) / 
                             (self.selecte_frame_scale - self.align_overlap_num))
    
    def get_images(self, frame_ids: List[int], **kwargs) -> List[Image.Image]:
        """Return indicted images list selected by frame_ids."""
        return [self.get_image(i, **kwargs) for i in frame_ids]
    
    def get_image(self, frame_id: int, **kwargs) -> Image.Image:
        """Return indicted image selected by frame_id."""
        return Image.open(self.imDir / f'{frame_id:06}{self.imExt}')
    
    def get_gt(self, frame_ids: List[int]) -> pd.DataFrame:
        """Return indicted gt DataFrame selected by frame_ids."""
        df = self.gt[self.gt['frame_index'].isin(frame_ids)]
        
        # filited the number of track in selected frame less than threshold
        appeared_track_counts = df[df['visibility'] >= self.visible_thresh]['track_id'].value_counts()
        selected_track_id = appeared_track_counts[appeared_track_counts >= self.min_valid_node_num].index
        return df[df['track_id'].isin(selected_track_id)]
    
    def convert2mate(self, frame_ids: List[int]) -> dict:
        """Generate video mate data from frame_ids."""
        raise NotImplementedError
    
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
            begin_index = item * (self.selecte_frame_scale - self.align_overlap_num) + self.start_frame_id
            # if end_index over max frame, fit it.
            if begin_index + self.selecte_frame_diff > self.end_frame_id:
                return self.end_frame_id - self.selecte_frame_diff
            else:
                return begin_index 
    
    def __len__(self):
        return self.len()    