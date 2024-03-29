import warnings
import logging
import numpy as np

logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


class PolyfitData(object):
    """Fit a Least squares polynomial fit of degree deg to Data (x, y).
    
    Args:
        x (np.array): The frameid or time list.
        y (np.array): The data list, support 1 dimensional for normal shaped [num_frame], 2 dimensional for bounding boxes shaped [num_frame, 4].
        deg (int): The degree of the fitting polynomial.
        coefficients (Optional): The coefficients of the polynomial fit, if not None, don not calculate this from x and y.
    """
    def __init__(self, x: np.array, y: np.array, deg: int=2, coefficients: np.ndarray=None):
        if coefficients is None:
            assert x.shape[0] == y.shape[0]
            if y.ndim == 2:
                self.coefficients = np.stack([np.polyfit(x, sub_y, deg) for sub_y in y.transpose()])
            elif y.ndim == 1:
                self.coefficients = np.polyfit(x, y, deg)
            else:
                raise ValueError(f"Invalid input number of dimensions {x.ndim}")
        else:
            self.coefficients = coefficients
    
    def __call__(self, new_x: np.array):
        if self.coefficients.ndim == 1:
            return np.poly1d(self.coefficients)(new_x)
        elif self.coefficients.ndim == 2:
            return np.stack([np.poly1d(sub_coef)(new_x) for sub_coef in self.coefficients]).transpose()


class InterpolateTrack():
    """A lot of track interpolate functions, assume x is frame_ids, and y is track data.
    The y supported 1 dimensional for normal shaped [num_frame], 2 dimensional for bounding boxes shaped [num_frame, 4]."""
    
    @staticmethod
    def copy(frame_ids, y, new_frame_ids):
        """Just copy y data to fill in new frame ids"""
        assert frame_ids.shape[0] == y.shape[0], "frame_ids and track data lengthes should be same."
        assert frame_ids.shape[0] < new_frame_ids.shape[0], "frame_ids lengthes should be less new_frame_ids."
        shape = y.shape
        shape[0] = new_frame_ids.shape[0]
        new_y = np.zeros(shape)
        temp_y = None
        for id in new_frame_ids:
            if id in frame_ids:
                temp_y = y[id]
            if temp_y is not None:
                new_y[id] = temp_y.copy()
        return new_y
    
    @staticmethod
    def average(frame_ids, y, new_frame_ids):
        """Calculate the average of two continuous frame to fill in new frame ids"""
        assert frame_ids.shape[0] == y.shape[0], "frame_ids and track data lengthes should be same."
        assert frame_ids.shape[0] < new_frame_ids.shape[0], "frame_ids lengthes should be less new_frame_ids."
        assert frame_ids[0] == new_frame_ids[0], "first frame id should be same."
        ratio = int(new_frame_ids.shape[0] / y.shape[0])
        
        new_y = []
        for i in range(len(frame_ids)):
            start_y = y[i].copy()
            end_y = start_y.copy() if i==len(frame_ids)-1 else y[i + 1].copy()
            if isinstance(start_y, np.ndarray):
                temp = []
                for sy, ey in zip(start_y, end_y):
                    temp.append(np.linspace(sy, ey, ratio, endpoint=False))
                new_y.append(np.stack(temp).transpose())
            else:
                new_y.append(np.linspace(start_y, end_y, ratio, endpoint=False))
        new_y = np.concatenate(new_y, axis=0)
        return new_y
    
    @staticmethod
    def poly(frame_ids, y, new_frame_ids, deg: int=2):
        """Use a Least squares polynomial fit of degree deg to fill in new frame ids"""
        assert frame_ids.shape[0] == y.shape[0], "frame_ids and track data lengthes should be same."
        assert frame_ids.shape[0] < new_frame_ids.shape[0], "frame_ids lengthes should be less new_frame_ids."
        if y.ndim == 2:
            coefficients = np.stack(
                [np.polyfit(frame_ids, sub_y, deg) for sub_y in y.transpose()])
            new_y = [np.poly1d(sub_coef)(new_frame_ids) for sub_coef in coefficients]
            return np.stack(new_y).transpose()
        elif y.ndim == 1:
            coefficients = np.polyfit(frame_ids, y, deg)
            return np.poly1d(coefficients)(new_frame_ids)
        else:
            raise ValueError(f"Invalid input number of dimensions {y.ndim}")
            
