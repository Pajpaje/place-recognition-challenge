import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import PlaceRecognitionDataset
from model import PlaceRecognitionModel


def main():
    data_root = 'Eynsham'
    max_distance = 5.0
    batch_size = 16
    num_epochs = 50

    dataset = PlaceRecognitionDataset(data_root, max_distance)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlaceRecognitionModel()

    # Set up the trainer and train the model
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=int(torch.cuda.is_available()), progress_bar_refresh_rate=20)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
