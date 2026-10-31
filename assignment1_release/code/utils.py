import cv2
import numpy as np


def load_image(path):
    # load an image into RGB format
    img = cv2.imread(path)
    img = img[:, :, ::-1]  # BGR -> RGB
    return img


def save_image(path, img):
    # save an RGB image into a file
    img = img.copy()[:, :, ::-1]
    return cv2.imwrite(path, img)


def resize_image(img, new_size, interpolation):
    # resize an image into new_size (w * h) using specified interpolation
    # opencv has a weird rounding issue & this is a hacky fix
    # ref: https://github.com/opencv/opencv/issues/9096
    mapping_dict = {cv2.INTER_NEAREST: cv2.INTER_NEAREST_EXACT}
    if interpolation in mapping_dict:
        img = cv2.resize(img, new_size, interpolation=mapping_dict[interpolation])
    else:
        img = cv2.resize(img, new_size, interpolation=interpolation)
    return img

import math

def get_largest_rotated_rect(w, h, angle_deg):
    """Calculates max area inner rectangle inside rotated box of size w x h."""
    angle_rad = math.radians(angle_deg)
    # Map angle to [0, pi/2)
    quadrant = int(math.floor(angle_rad / (math.pi / 2))) & 3
    sign_alpha = angle_rad if ((quadrant & 1) == 0) else math.pi - angle_rad
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    # Ensure w <= h convention for aspect ratio check
    is_w_smaller = w < h
    side_short = w if is_w_smaller else h
    side_long = h if is_w_smaller else w

    # Max interior rectangle condition check
    if math.sin(2 * alpha) >= (side_short / side_long):
        # Touches 2 sides
        w_crop = side_short / math.sin(alpha) if math.sin(alpha) > 0 else side_short
        h_crop = side_short / math.cos(alpha) if math.cos(alpha) > 0 else side_short
        if not is_w_smaller:
            w_crop, h_crop = h_crop, w_crop
    else:
        # Touches all 4 sides
        cos_2a = math.cos(2 * alpha)
        if abs(cos_2a) < 1e-6:
            cos_2a = 1e-6
        w_crop = (w * math.cos(alpha) - h * math.sin(alpha)) / cos_2a
        h_crop = (h * math.cos(alpha) - w * math.sin(alpha)) / cos_2a

    return int(abs(w_crop)), int(abs(h_crop))