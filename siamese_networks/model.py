import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl


class SiameseNetwork(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load pre-trained SqueezeNet
        original_feature_extractor = models.squeezenet1_1(pretrained=True).features

        # Create new first layer to accommodate 5-channel input
        new_first_layer = torch.nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(2, 2))

        # Create new feature extractor with the new first layer and the rest of the original layers
        self.feature_extractor = nn.Sequential(
            new_first_layer,
            *original_feature_extractor[1:]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, input1, input2):
        output1 = self.feature_extractor(input1)
        output2 = self.feature_extractor(input2)
        output1 = self.global_avg_pool(output1)
        output2 = self.global_avg_pool(output2)
        output1, output2 = output1.view(output1.size(0), -1), output2.view(output2.size(0), -1)
        return output1, output2

    def training_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        distance = F.pairwise_distance(output1, output2)
        similarity = torch.sigmoid(-distance)
        loss = F.binary_cross_entropy(similarity, labels.float())

        self.log('train/loss', loss)
        self.train_accuracy(similarity, labels.float())
        self.log('train/accuracy', self.train_accuracy)
        self.train_f1(similarity, labels.float())
        self.log('train/f1', self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        distance = F.pairwise_distance(output1, output2)
        similarity = torch.sigmoid(-distance)
        loss = F.binary_cross_entropy(similarity, labels.float())

        self.log('val/loss', loss)
        self.val_accuracy(similarity, labels)
        self.log('val/accuracy', self.val_accuracy)
        self.val_f1(similarity, labels)
        self.log('val/f1', self.val_f1)

        return loss

    def test_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        distance = F.pairwise_distance(output1, output2)
        similarity = torch.sigmoid(-distance)
        loss = F.binary_cross_entropy(similarity, labels.float())

        self.log('test/loss', loss)
        self.test_accuracy(similarity, labels)
        self.log('test/accuracy', self.test_accuracy)
        self.test_f1(similarity, labels)
        self.log('test/f1', self.test_f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
