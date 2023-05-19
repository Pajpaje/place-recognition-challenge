import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl


class SiameseNetwork(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        # Load pre-trained SqueezeNet
        self.feature_extractor = models.squeezenet1_1(pretrained=True)

        # Replace the first layer to accommodate 5-channel input
        self.feature_extractor.features[0] = torch.nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(2, 2))

        # Freeze all layers except the first one
        for name, param in self.feature_extractor.named_parameters():
            if not name.startswith('features.0'):  # The name of first layer usually starts with 'features.0'
                param.requires_grad = False

    def forward(self, input1, input2):
        output1 = self.feature_extractor(input1)
        output2 = self.feature_extractor(input2)
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
