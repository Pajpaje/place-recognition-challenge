import os
import re
import numpy as np
from PIL import Image


def parse_alog_file(alog_path):
    images_info = []
    coord_n = None
    coord_e = None
    with open(alog_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "GPS" in line:
                # TODO not every line should be parsed, but sometimes there are 2 lines with images in a row so a simople flag won't do the job
                parts = line.strip().split()[-1].split(',')
                coord_n = float(parts[0].split('=')[1])
                coord_e = float(parts[1].split('=')[1])
            if "LADYBUG_GRAB" in line:
                parts = line.strip().split()[-1].split(',')
                image_files = [part.split('=')[1] for part in parts if 'File' in part]
                images_info.append(((coord_n, coord_e), image_files))
    return images_info


def load_image(image_path):
    return np.array(Image.open(image_path))


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))