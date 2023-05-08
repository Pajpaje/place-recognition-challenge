import os
import glob
import torch
from torch.utils.data import Dataset
from utils import parse_alog_file, load_image, euclidean_distance


class PlaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, max_distance):
        self.root_dir = root_dir
        self.max_distance = max_distance
        self.data = []

        images_dir = os.path.join(root_dir, 'images')
        raw_logs_dir = os.path.join(root_dir, 'Raw_Logs')

        # Get a list of all .alog files recursively
        alog_files = glob.glob(os.path.join(raw_logs_dir, '**/*.alog'), recursive=True)

        for alog_file in alog_files:
            gps_data, image_data = parse_alog_file(alog_file)

            # Iterate over image_data and create pairs of image sets
            for i in range(len(image_data)):
                for j in range(i + 1, len(image_data)):
                    dist = euclidean_distance(gps_data[i], gps_data[j])
                    label = 1 if dist <= max_distance else 0

                    # Get image paths for both sets of images
                    images_i = [os.path.join(images_dir, img_name) for img_name in image_data[i]]
                    images_j = [os.path.join(images_dir, img_name) for img_name in image_data[j]]

                    self.data.append((images_i, images_j, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_i, images_j, label = self.data[idx]

        # Load and concatenate images
        images_i = [load_image(img_path) for img_path in images_i]
        images_j = [load_image(img_path) for img_path in images_j]

        images = torch.cat(images_i + images_j, dim=0)
        label = torch.tensor(label, dtype=torch.float32)

        return images, label
