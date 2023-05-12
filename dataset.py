import os
import glob
from random import shuffle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from utils import parse_alog_file, load_image, euclidean_distance, ResizeWithPad


class PlaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, max_distance, distance_threshold):
        self.root_dir = root_dir
        self.max_distance = max_distance
        self.distance_threshold = distance_threshold
        self.transform = Compose([
            ToTensor(),
            ResizeWithPad(size=(256, 256))
        ])
        self.data = []

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

                if dist > self.max_distance:
                    continue
                # TODO Sometimes we want to compare with places that are further than max_distance

                label = 1 if dist <= self.distance_threshold else 0

                # Get image paths for both sets of images
                images_i = [os.path.join(images_dir, img_name) for img_name in gps_and_image_data[i][1]]
                images_j = [os.path.join(images_dir, img_name) for img_name in gps_and_image_data[j][1]]

                self.data.append((images_i, images_j, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_i, images_j, label = self.data[idx]
        images_i = [self.transform(load_image(img_path)).squeeze(0) for img_path in images_i]
        images_i = torch.stack(images_i, dim=0)
        images_j = [self.transform(load_image(img_path)).squeeze(0) for img_path in images_j]
        images_j = torch.stack(images_j, dim=0)
        images = torch.cat((images_i, images_j), dim=0)
        label = torch.tensor(label, dtype=torch.float32)
        return images, label
