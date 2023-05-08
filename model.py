import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class PlaceRecognitionModel(pl.LightningModule):
    def __init__(self):
        super(PlaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(30, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 96 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.binary_cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
