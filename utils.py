import glob
import os
from random import shuffle
from typing import Tuple, Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def parse_alog_file(alog_path):
    gps_and_image_info = []
    coord_n = None
    coord_e = None
    with open(alog_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "GPS" in line:
                parts = line.strip().split()[-1].split(',')
                parts_dict = {part.split('=')[0]: part.split('=')[1] for part in parts}
                coord_n = float(parts_dict.get('N', 0))
                coord_e = float(parts_dict.get('E', 0))
            if "LADYBUG_GRAB" in line:
                parts = line.strip().split()[-1].split(',')
                image_files = [part.split('=')[1].split('/')[-1] for part in parts if 'File' in part]
                if coord_n is not None and coord_e is not None:  # only append if we have valid GPS coordinates
                    gps_and_image_info.append(((coord_n, coord_e), image_files))
                coord_n = None
                coord_e = None
    return gps_and_image_info


def get_pairs_of_places(root_dir, max_distance, distance_threshold):
    data = []

    images_dir = os.path.join(root_dir, 'images')
    raw_logs_dir = os.path.join(root_dir, 'Raw_Logs')

    # Get a list of all .alog files recursively
    alog_files = glob.glob(os.path.join(raw_logs_dir, '**/*.alog'), recursive=True)

    gps_and_image_data = []
    for alog_file in alog_files:
        gps_and_image_data.extend(parse_alog_file(alog_file))
    shuffle(gps_and_image_data)

    # Iterate over image_data and create pairs of image sets
    for i in range(len(gps_and_image_data)):
        for j in range(i + 1, len(gps_and_image_data)):
            dist = euclidean_distance(gps_and_image_data[i][0], gps_and_image_data[j][0])

            if dist > max_distance:
                continue
            # TODO Sometimes we want to compare with places that are further than max_distance

            label = 1 if dist <= distance_threshold else 0

            # Get image paths for both sets of images
            place_1 = [os.path.join(images_dir, img_name) for img_name in gps_and_image_data[i][1]]
            place_2 = [os.path.join(images_dir, img_name) for img_name in gps_and_image_data[j][1]]

            data.append((place_1, place_2, label))
    return data


def load_image(image_path):
    return Image.open(image_path)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


class ResizeWithPad:
    """Resizes and pads an image to a target height and width without distortion."""
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if type(size) is int:
            self._target_height = self._target_width = size
        elif type(size) is tuple:
            self._target_height, self._target_width = size[0], size[1]
        else:
            raise TypeError("Incorrect target shape (should be int or tuple of 2 integers)")
        self._resize_transform = transforms.Resize(size=(self._target_height, self._target_width), antialias=False)
        # set antialias=False to avoid a warning message ( TODO check if this is the best way to do it )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        scaled_h, scaled_w = self.scaled_image_dims(image)
        image = transforms.Resize(size=(scaled_h, scaled_w), antialias=False)(image)
        vp = int((self._target_height - scaled_h) / 2)
        hp = int((self._target_width - scaled_w) / 2)
        padding = [hp, vp, hp + (scaled_w % 2) * (hp > 0), vp + (scaled_h % 2) * (vp > 0)]
        return F.pad(image, padding, 0, "constant")

    def resize_factor(self, image: torch.Tensor) -> int:
        height_ratio = image.shape[-2] / self._target_height
        width_ratio = image.shape[-1] / self._target_width
        return max(max(height_ratio, width_ratio), 1)

    def scaled_image_dims(self, image: torch.Tensor) -> Tuple[int, int]:
        r_factor = self.resize_factor(image)
        return int(image.shape[-2] / r_factor), int(image.shape[-1] / r_factor)
