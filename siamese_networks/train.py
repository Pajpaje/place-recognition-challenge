import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import SiameseNetwork
from pytorch_lightning.tuner.tuning import Tuner

from siamese_networks.dataset import SiameseDataset, SiameseConcatenationDataModule


def main():
    # Configuration
    data_root = '..\\Eynsham'
    max_distance = 1000
    distance_threshold = 200
    batch_size = 16
    max_epochs = 25

    print("Dataset loaded.")
    datamodule = SiameseConcatenationDataModule(data_root, max_distance, distance_threshold, batch_size=batch_size)

    print("Loading model...")
    model = SiameseNetwork()

    logger = WandbLogger(
        entity="spytaj-chata",
        project="samochodowy-rozpoznawacz",
        name="default",
    )

    print("Model loaded.")
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=max_epochs, logger=logger)

    # Find hyperparameters automatically
    # tuner = Tuner(trainer)

    # Find maximum batch size
    # tuner.scale_batch_size(model, datamodule)
    #
    # Find good learning rate
    # lr_finder = tuner.lr_find(model, datamodule)

    # Get the suggestion
    # suggestion = lr_finder.suggestion()

    # Plot and print the suggestion
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # print(f"Suggestion = {suggestion}")

    # Pick point based on plot, or get suggestion
    # model.hparams.lr = suggestion

    trainer.fit(model, datamodule)

    print(trainer.callback_metrics)

    trainer.test(model, datamodule)

    print(trainer.callback_metrics)
    # TODO test wandb logging


if __name__ == '__main__':
    main()
