#!/usr/bin/python

import torch
from torch.nn import functional as F
from torch import optim

from transformers import ViTForImageClassification
import torchmetrics


import lightning as L

class VisionTransformerPretrained(L.LightningModule):
    '''
    Wrapper for the torchvision pretrained Vision Transformers

    Args:
      model (str)       : specifies which flavor of ViT to use
      num_classes (int) : number of output classes
      learning_rate (float) : optimizer learning rate

    '''

    def __init__(self, model, num_classes, learning_rate):

        super().__init__()

        if model == 'vit_b_16':
            backbone = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            raise ValueError(model)

        self.backbone = backbone

        # metrics
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)

        # other
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.backbone(x)

    def step(self, batch):
       '''
       Any step processes batch to return loss and predictions
       '''

       x, y = batch
       prediction = self.backbone(x)
       y_hat = torch.argmax(prediction.logits, dim=-1)

       loss = F.cross_entropy(prediction.logits, y)
       acc = self.acc(y_hat, y)
       
       return loss, acc, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', acc, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
