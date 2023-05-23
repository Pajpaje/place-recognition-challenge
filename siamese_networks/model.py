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
        original_feature_extractor = models.squeezenet1_1(pretrained=True).features

        # Create new first layer to accommodate 5-channel input
        new_first_layer = torch.nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(2, 2))

        # Create new feature extractor with the new first layer and the rest of the original layers
        self.feature_extractor = nn.Sequential(
            new_first_layer,
            *original_feature_extractor[1:]
        )

        # Freeze all layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze the first layer and the last layer of the last module
        next(self.feature_extractor.parameters()).requires_grad = True
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        loss = F.mse_loss(similarity, labels.float())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
