import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from place_concatenation.dataset import PlaceConcatenationDataModule
from utils import load_image, ResizeWithPad, get_pairs_of_places


class SiameseDataset(Dataset):
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
        label = torch.tensor(label, dtype=torch.float32)
        return images_i, images_j, label


SiameseDataModule = PlaceConcatenationDataModule
