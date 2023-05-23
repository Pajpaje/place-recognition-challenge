import os

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_image, ResizeWithPad, get_pairs_of_places, get_pairs_of_preprocessed_places


class SiameseDataset(Dataset):
    def __init__(self, root_dir, max_distance, distance_threshold):
        self.root_dir = root_dir
        self.max_distance = max_distance
        self.distance_threshold = distance_threshold
        self.data = get_pairs_of_preprocessed_places(self.root_dir, self.max_distance, self.distance_threshold)
        # TODO dvc pipelines and other solutions from the template may be useful l8r
        # https://github.com/ashleve/lightning-hydra-template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(idx)
        place_i, place_j, label = self.data[idx]
        place_i = torch.load(place_i)
        place_j = torch.load(place_j)
        label = torch.tensor(label, dtype=torch.float32)
        return place_i, place_j, label


class SiameseConcatenationDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_val_test_ratio=(0.8, 0.1, 0.1), batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

    def setup(self, stage=None):
        # Calculate the lengths for train, validation, and test sets
        total_length = len(self.dataset)
        train_length = int(total_length * self.train_val_test_ratio[0])
        val_length = int(total_length * self.train_val_test_ratio[1])
        test_length = total_length - train_length - val_length

        # Split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_length, val_length, test_length]
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
