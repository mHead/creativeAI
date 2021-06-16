from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools import utils as u
from ..DatasetMusic2emotion.tools.utils import format_timestamp
from ..DatasetMusic2emotion.tools import va2emotion as va2em

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
from torch.nn import Module
import torch.cuda as cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPTDataset, emoMusicPTSubset, emoMusicPTDataLoader

CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8
}




class TorchModel(Module):
    def __init__(self, dataset: emoMusicPTDataset, train_dl: emoMusicPTDataLoader, test_dl: emoMusicPTDataLoader,
                 save_dir_root, version=None,
                 **kwargs):
        super(TorchModel, self).__init__()
        """
        :param train_dl -> emoMusicPTDataLoader
        :param test_dl -> emoMusicPTDataLoader
        :param save_dir_root -> where to store results
        :param kwargs: optionals
        """
        self.version = version
        if version == 'v0':
            self.name = "CNN_1conv1D"
        elif version == 'v1':
            self.name = "CNN_2conv1D"
        elif version == 'v3':
            self.name = "CNN_3conv1D"

        self.save_dir = save_dir_root
        self.emoMusicPTDataset = dataset
        self.train_dataloader = train_dl
        self.test_dataloader = test_dl
        self.labelsDict = va2em.EMOTIONS
        self.labelsDict.__delitem__(8)
        self.num_classes = len(self.labelsDict)
        self.example_0, self.ex0_songid, self.ex0_filename, self.ex0_label, self.ex0_label_coords = train_dl.dataset[0]
        self.input_shape = self.example_0.shape
        print(f'self.input_shape {type(self.example_0)}')

        self.kernel_features_maps = CNNHyperParams.get('kernel_features_maps')
        self.kernel_size = CNNHyperParams.get('kernel_size')
        self.kernel_shift = int(CNNHyperParams.get('kernel_shift'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1d_L_output = ((self.input_shape[1] - 1 * (self.kernel_size - 1) - 1) // self.kernel_shift) + 1
        self.conv1d_output_dim = self.conv1d_L_output * self.kernel_features_maps

        print(f'[TorchModel.py]{self.name} will run on the following device: {self.device}')

        # Network definition
        self.conv1 = conv1DBlock(in_channels=1, out_channels=self.kernel_features_maps, kernel_size=self.kernel_size, kernel_stride=self.kernel_shift)

        #if version == 'v1':
           #self.conv2 =  tmb.conv1DBlock()

        self.flatten = nn.Flatten()

        self.clf_fc = nn.Linear(self.conv1d_output_dim, self.num_classes, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):

        x = self.conv1(x)
        if self.version == 'v1':
            x = self.conv2(x)

        flatten = self.flatten(x)
        logits = self.clf_fc(flatten)

        return logits, flatten


class conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_stride):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._kernel_stride = kernel_stride

        self._block = nn.Sequential(
            nn.Conv1d(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size=self._kernel_size,
                      stride=self._kernel_stride),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm1d(num_features=out_channels),
            # nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.kernel_shift),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        for layer in self._block:
            x = layer(x)
        return x
