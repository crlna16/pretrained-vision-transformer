#!/usr/bin/env python

import torch

from torchvision.datasets import Country211
from torchvision.transforms import v2

import lightning as L
from torch.utils.data import DataLoader

class Country211DataModule(L.LightningDataModule):
    '''
    Lightning datamodule for the Country211 dataset

    '''

    def __init__(self, data_root, batch_size):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size

        self.num_workers = 8

    @property
    def num_classes(self):
        return 200 # TODO check

    def prepare_data(self):
        '''
        Download the data
        '''
        Country211(self.data_root, split='train', download=True)
        Country211(self.data_root, split='valid', download=True)
        Country211(self.data_root, split='test', download=True)

    def setup(self, split):
        '''
        Setup the dataset

        Arguments:
          split : choice of train, valid, test
        '''

        # define the transforms
        transforms = v2.Compose([v2.ToImage(),
                                 v2.RandomResizedCrop(size=(224,224), antialias=True),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
                                
        self.train_data = Country211(self.data_root, split='train', transform=transforms)
        self.valid_data = Country211(self.data_root, split='valid', transform=transforms)
        self.test_data = Country211(self.data_root, split='test', transform=transforms)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
