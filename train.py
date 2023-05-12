from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import PlaceRecognitionDataset
from model import PlaceRecognitionModel
from pytorch_lightning.tuner.tuning import Tuner


def main():
    data_root = 'Eynsham'
    max_distance = 1000
    distance_threshold = 200
    batch_size = 2
    max_epochs = 2

    dataset = PlaceRecognitionDataset(data_root, max_distance, distance_threshold)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PlaceRecognitionModel()
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=max_epochs)

    tun = Tuner(trainer)
    lr_finder = tun.lr_find(model, dataloader, attr_name="lr")
    fig = lr_finder.plot(suggest=True)
    fig.show()
    model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
