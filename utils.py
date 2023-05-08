import numpy as np
from PIL import Image


def parse_alog_file(alog_path):
    pass


def load_image(image_path):
    return np.array(Image.open(image_path))


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
