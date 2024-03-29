"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from utils.track_ops import area, wh_ratio


def crop(images, targets, region):
    cropped_images = [F.crop(image, *region) for image in images]

    targets = targets.copy()
    i, j, h, w = region
    
    fields = ["labels"]

    if "boxes" in targets:
        boxes = targets["boxes"]
        boxes_area = area(boxes)
        max_size = torch.as_tensor([w, h, w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes, max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        targets["boxes"] = cropped_boxes
        fields.append("boxes")

    # fix target elements for which the boxes have changed area
    if "boxes" in targets:
        cropped_boxes_area = area(targets["boxes"])
        cropped_boxes_wh_ratio = wh_ratio(targets["boxes"])
        # less than 1 piex area or w/h ratio not in [0.2, 5]
        mask = (cropped_boxes_area < 1) | (cropped_boxes_wh_ratio < 0.2) | (cropped_boxes_wh_ratio > 5)
        
        # reset zero area boxes referred is 0
        if "referred" in targets:
            targets["referred"][mask] = 0
            fields.append("referred")
        
        # scale visibilities with its box area scale ratio
        if "visibilities" in targets:
            targets["visibilities"] *= (cropped_boxes_area / boxes_area).nan_to_num()
            fields.append("visibilities")
        
        # remove track elements for which the boxes all have zero area
        keep = ~torch.all(mask, dim=1)
        for field in fields:
            targets[field] = targets[field][keep]

    return cropped_images, targets


def hflip(images, targets):
    flipped_images = [F.hflip(image) for image in images]

    w, h = images[0].size

    targets = targets.copy()
    if "boxes" in targets:
        boxes = targets["boxes"]
        boxes = boxes[..., [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        targets["boxes"] = boxes

    return flipped_images, targets


def resize(images, targets, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(images[0].size, size, max_size)
    rescaled_images = [F.resize(image, size) for image in images]

    if targets is None:
        return rescaled_images, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_images[0].size, images[0].size))
    ratio_width, ratio_height = ratios

    targets = targets.copy()
    if "boxes" in targets:
        boxes = targets["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        targets["boxes"] = scaled_boxes

    return rescaled_images, targets


def frozen_time(images, targets):
    frozen_images = [images[0] for _ in images]
    frame_num = len(frozen_images)
    
    targets = targets.copy()
    if "boxes" in targets:
        frozen_boxes = targets["boxes"][:, 0][:, None].repeat(1, frame_num, 1)
        targets["boxes"] = frozen_boxes
    if "visibilities" in targets:
        frozen_vis = targets["visibilities"][:, 0][:, None].repeat(1, frame_num)
        targets["visibilities"] = frozen_vis
    if "referred" in targets:
        frozen_referred = targets["referred"][:, 0][:, None].repeat(1, frame_num)
        targets["referred"] = frozen_referred
    
    return frozen_images, targets


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, targets):
        region = T.RandomCrop.get_params(imgs[0], self.size)
        return crop(imgs, targets, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs, targets: dict):
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        return crop(imgs, targets, region)


class FixedMotRandomCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, imgs, targets: dict):
        w, h = imgs[0].width, imgs[0].height
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        
        min_size = random.randint(self.min_size, min(min_original_size, self.max_size))
        max_size = int(round(min_size * max_original_size / min_original_size))
        
        if w > h:
            ow, oh = max_size, min_size
        else:
            ow, oh = min_size, max_size
        region = T.RandomCrop.get_params(imgs[0], [oh, ow])
        return crop(imgs, targets, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, targets):
        image_width, image_height = imgs[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(imgs, targets, (crop_top, crop_left, crop_height, crop_width))


class MotRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, targets):
        if random.random() < self.p:
            return hflip(imgs, targets)
        return imgs, targets


class MotRandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, imgs, targets=None):
        size = random.choice(self.sizes)
        return resize(imgs, targets, size, self.max_size)


class MotConfidenceFilter(object):
    """Filter all zero confidence track"""
    def __call__(self, imgs, targets):
        targets = targets.copy()
        fields = ["labels", "boxes", "referred"]
        if "boxes" in targets and "referred" in targets:
            keep = ~torch.all(targets["referred"] < 1, dim=1)
            
            if "visibilities" in targets:
                fields.append("visibilities")
            
            for field in fields:
                targets[field] = targets[field][keep]
        
        return imgs, targets


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, imgs, targets):
        if random.random() < self.p:
            return self.transforms1(imgs, targets)
        return self.transforms2(imgs, targets)


class MotRandomFrozenTime(object):
    """Likely crop operation, but work on time dim to get frozen time images and targets with probability p."""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, imgs, targets):
        if random.random() < self.p:
            return frozen_time(imgs, targets)
        return imgs, targets


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class MotToTensor(ToTensor):
    """Return Tensor images with `num frames` x `channel` x `height` x `width`"""
    def __call__(self, imgs, targets):
        return torch.stack([F.to_tensor(img) for img in imgs]), targets


class MotColorJitter(T.ColorJitter):
    def __call__(self, imgs, targets):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                imgs = [F.adjust_brightness(img, brightness_factor) for img in imgs]
            elif fn_id == 1 and contrast_factor is not None:
                imgs = [F.adjust_contrast(img, contrast_factor) for img in imgs]
            elif fn_id == 2 and saturation_factor is not None:
                imgs = [F.adjust_saturation(img, saturation_factor) for img in imgs]
            elif fn_id == 3 and hue_factor is not None:
                imgs = [F.adjust_hue(img, hue_factor) for img in imgs]
             
        return imgs, targets


class RandomApply(T.RandomApply):
    """Apply randomly a list of transformations with a given probability."""
    def __call__(self, images, targets):
        if self.p < torch.rand(1):
            return images, targets
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, targets=None):
        images = F.normalize(images, mean=self.mean, std=self.std)
        if targets is None:
            return images, None
        targets = targets.copy()
        h, w = images.shape[-2:]
        if "boxes" in targets:
            boxes = targets["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            targets["boxes"] = boxes
        return images, targets


class NormalizeInverse(T.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def _convert_boxes(self, boxes, h, w):
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        return boxes

    def __call__(self, images: torch.Tensor, targets=None, boxes=None):
        h, w = images.shape[-2:]
        images = super().__call__(images.clone())
        if boxes is not None:
            if isinstance(boxes, list):
                return images, [self._convert_boxes(box, h, w) for box in boxes]
            return images, self._convert_boxes(boxes, h, w)
        if targets is not None:
            targets = targets.copy()
            if "boxes" in targets:
                targets["boxes"] = self._convert_boxes(targets["boxes"], h, w)
            return images, targets
        return images, None


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, images, targets):
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n\t{t}"
        format_string += "\n)"
        return format_string
        