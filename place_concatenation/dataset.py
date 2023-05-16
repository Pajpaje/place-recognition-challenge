import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor
import pytorch_lightning as pl
from utils import load_image, ResizeWithPad, get_pairs_of_places


class PlaceConcatenationDataset(Dataset):
    def __init__(self, root_dir, max_distance, distance_threshold):
        self.root_dir = root_dir
        self.max_distance = max_distance
        self.distance_threshold = distance_threshold
        self.transform = Compose([
            ToTensor(),
            ResizeWithPad(size=(256, 256))
        ])
        self.data = get_pairs_of_places(self.root_dir, self.max_distance, self.distance_threshold)

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


class PlaceConcatenationDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_val_test_ratio=(0.8, 0.1, 0.1), batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.batch_size = batch_size
        self.num_workers = min(2, os.cpu_count())

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
