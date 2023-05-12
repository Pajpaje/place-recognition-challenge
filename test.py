import pytorch_lightning as pl
from dataset import PlaceRecognitionDataset, PlaceRecognitionDataModule
from model import PlaceRecognitionModel


def main():
    # Configuration
    data_root = 'Eynsham'
    max_distance = 1000
    distance_threshold = 200
    batch_size = 16

    checkpoint_path = ""
    hparams_file = ""

    dataset = PlaceRecognitionDataset(data_root, max_distance, distance_threshold)
    datamodule = PlaceRecognitionDataModule(dataset, batch_size=batch_size)

    # Load the model from the checkpoint
    model = PlaceRecognitionModel.load_from_checkpoint(checkpoint_path, hparams_file=hparams_file)

    trainer = pl.Trainer(accelerator="gpu")

    # Run testing
    trainer.test(model, datamodule)
    # Show stats such as accuracy, precision, recall
    print(trainer.callback_metrics)

    # TODO show stats such as accuracy precision recall

if __name__ == "__main__":
    main()
