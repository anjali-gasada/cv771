import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

from utils import resize_image, load_image, get_largest_rotated_rect

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# Sample implementation. No need to modify
#################################################################################

class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    This class will crop a random region with in an input image. The target area
    / aspect ratio (=width/height) of the region is first sampled from a uniform
    distribution. A region satisfying the area / aspect ratio is then sampled
    and cropped. This crop is finally resized to a fixed given size. This is
    widely used as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            # compute the width and height
            # crop the image and resize to output size
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # crop the image
            if w <= img.shape[1] and h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)

                img = img[y1 : y1 + h, x1 : x1 + w]
                if isinstance(self.size, int):
                    img = resize_image(
                        img, (self.size, self.size), interpolation=interpolation
                    )
                else:
                    img = resize_image(img, self.size, interpolation=interpolation)

                return img

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            h, w = img.shape[0], img.shape[1]
            th, tw = self.size, self.size
            x1 = int(round((w - tw) / 2.0))
            y1 = int(round((h - th) / 2.0))
            img = img[y1 : y1 + th, x1 : x1 + tw]
            return img
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    This class will resize an input image based on its shortest side.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        # scale the image
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if h > w:
                # Width is the shorter edge
                new_w = self.size
                new_h = int(round(self.size * h / w))
            else:
                # Height is the shorter edge (or square image)
                new_h = self.size
                new_w = int(round(self.size * w / h))
            return resize_image(img, (new_w, new_h), interpolation=interpolation)
            #################################################################################
            # Fill in the code here
            #################################################################################
            
        else:
            #################################################################################
            # Fill in the code here
            #################################################################################
            return resize_image(img, self.size, interpolation=interpolation)

    def __repr__(self):
        if isinstance(self.size, int):
            return "Scale [Shortest Side ({:d})]".format(self.size)
        else:
            target_size = self.size
            return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomColor(object):
    """Perturb color channels of a given image.

    This class will apply random color perturbation to an input image. An alpha
    value is first sampled uniformly from the range of (-r, r). 1 + alpha is
    further multiply to a color channel. The sampling is done independently for
    each channel. An efficient implementation can be achieved using a LuT.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        is_float = np.issubdtype(img.dtype, np.floating)
        
        # 1. Convert float image [0.0, 1.0] to uint8 [0, 255] if necessary
        if is_float:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)

        num_channels = img_uint8.shape[2] if img_uint8.ndim == 3 else 1
        inputs = np.arange(256, dtype=np.float32)
        
        # 2. OpenCV requires shape (1, 256, num_channels) for multi-channel LUT
        lut = np.zeros((1, 256, num_channels), dtype=np.uint8)

        for c in range(num_channels):
            # Sample alpha ~ U(1 - r, 1 + r)
            alpha = np.random.uniform(1 - self.color_range, 1 + self.color_range)
            
            # Map [0..255] * alpha and clip to [0, 255]
            channel_lut = np.clip(inputs * alpha, 0, 255).astype(np.uint8)
            lut[0, :, c] = channel_lut

        # 3. Apply lookup table
        out = cv2.LUT(img_uint8, lut)

        # 4. Convert back to original float format if the input was float
        if is_float:
            out = out.astype(np.float32) / 255.0

        return out

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )

class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    This class will randomly rotate an image and further crop a local region with
    maximum area. A rotation angle is first sampled and then applied to the input.
    A region with maximum area and without any empty pixel is further determined
    and cropped.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Fill in the code here
        #################################################################################
        # get the rectangular with max area in the rotated image
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        interpolation = random.choice(self.interpolations)

        # 2. Get affine rotation matrix & apply transformation
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=interpolation)

        # 3. Compute largest valid crop dimensions
        crop_w, crop_h = get_largest_rotated_rect(w, h, degree)

        # 4. Extract center crop
        x1 = max(0, int(center[0] - crop_w / 2.0))
        y1 = max(0, int(center[1] - crop_h / 2.0))
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        return rotated_img[y1:y2, x1:x2]

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class Resize(object):
    """
    Resize an input image into a fixed resolution.
    """
    def __init__(self, size):
        assert (
            isinstance(size, int)
            or (isinstance(size, collections.Iterable) and len(size) == 2)
           )
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        img = resize_image(img, self.size, cv2.INTER_LINEAR)
        return img


class CenterCrop(object):
    """
    Crop the center patch of a given image (numpy array)

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
    """

    def __init__(
        self,
        size,
    ):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        if isinstance(self.size, int):
            th, tw = self.size, self.size
        else:
            th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        img = img[y1 : y1 + th, x1 : x1 + tw]
        return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Random Crop" + "[Size ({:d}, {:d})]".format(
            target_size[0],
            target_size[1],
        )


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader (from our tutorial)
    """
    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # list all files
        file_list = glob.glob(os.path.join(root_folder, '*.{:s}'.format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load img and label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = 1 if "cat" in os.path.basename(filename) else 0
        # apply data augmentation
        if self.transforms is not None:
            img  = self.transforms(img)
        return img, label


class Net(nn.Module):
    """
    A simple neural network modified from PyTorch tutorial
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
