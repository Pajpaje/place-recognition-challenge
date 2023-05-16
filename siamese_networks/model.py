import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from place_concatenation.model import DepthwiseSeparableConv


class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.base_network = nn.Sequential(
            DepthwiseSeparableConv(5, 64, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        input1, input2, labels = batch
        output1, output2 = self.forward(input1, input2)
        distance = F.pairwise_distance(output1, output2)
        loss = F.mse_loss(distance, labels.float())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
