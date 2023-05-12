import pytorch_lightning as pl
from dataset import PlaceRecognitionDataset, PlaceRecognitionDataModule
from model import PlaceRecognitionModel
from pytorch_lightning.tuner.tuning import Tuner


def main():
    # Configuration
    data_root = 'Eynsham'
    max_distance = 1000
    distance_threshold = 200
    batch_size = 16
    max_epochs = 25

    dataset = PlaceRecognitionDataset(data_root, max_distance, distance_threshold)
    datamodule = PlaceRecognitionDataModule(dataset, batch_size=batch_size)
    model = PlaceRecognitionModel()
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=max_epochs)

    # Find hyperparameters automatically
    tuner = Tuner(trainer)

    # Find maximum batch size
    tuner.scale_batch_size(model, datamodule)

    # Find good learning rate
    lr_finder = tuner.lr_find(model, datamodule)

    # Get the suggestion
    suggestion = lr_finder.suggestion()

    # Plot and print the suggestion
    fig = lr_finder.plot(suggest=True)
    fig.show()
    print(f"Suggestion = {suggestion}")

    # Pick point based on plot, or get suggestion
    model.hparams.lr = suggestion

    trainer.fit(model, datamodule)

    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
