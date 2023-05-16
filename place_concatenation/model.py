import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PlaceRecognitionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(PlaceRecognitionModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')

        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')

        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.binary_cross_entropy(outputs, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_accuracy(outputs, labels)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_precision(outputs, labels)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_recall(outputs, labels)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.binary_cross_entropy(outputs, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(outputs, labels)
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_precision(outputs, labels)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_recall(outputs, labels)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.binary_cross_entropy(outputs, labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_accuracy(outputs, labels)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_precision(outputs, labels)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_recall(outputs, labels)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
