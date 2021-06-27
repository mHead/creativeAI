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

# Baseline_v0: 30% accuracy on test set
'''
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8,
    "groups": 1,
    "batch_size": 1
}
'''
# Baseline_v0.1: Test Loss: 93.2152, Test Acc: 25.9470%
'''
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8 * 16,
    "groups": 1,
    "batch_size": 32
}
'''
# Baseline_v1: 34,6% accuracy on test set <------- Best until 18 giu 2021 18:50
'''
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8 * 32,
    "groups": 1,
    "batch_size": 1
}
'''
# Baseline_v1.1: 34,21% accuracy on test set
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8 * 32,
    "groups": 1,
    "batch_size": 4
}
# Baseline_v2: 32% accuracy on test set
'''
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8 * 64,
    "groups": 1,
    "batch_size": 1
}

'''
# Baseline_v3: best is 27,08%
'''
CNNHyperParams = {
    "kernel_size": 22050,
    "kernel_shift": 11025,
    "kernel_features_maps": 8 * 32,
    "groups": 1,
    "batch_size": 1
}
'''


class TorchModel(Module):
    """
    Baseline with one conv1D layer
    """
    def __init__(self, dataset: emoMusicPTDataset, train_dl: emoMusicPTDataLoader, test_dl: emoMusicPTDataLoader,
                 val_dl: emoMusicPTDataLoader,
                 save_dir_root, version=None, n_gru=None,
                 **kwargs):
        super(TorchModel, self).__init__()
        """
        :param train_dl -> emoMusicPTDataLoader
        :param test_dl -> emoMusicPTDataLoader
        :param save_dir_root -> where to store results
        :param kwargs: optionals
        """
        self.version = version
        self.name = 'music2emoCNN_' + self.version

        self.save_dir = save_dir_root
        self.emoMusicPTDataset = dataset
        self.train_dataloader = train_dl
        self.val_dataloader = val_dl
        self.test_dataloader = test_dl
        self.labelsDict = va2em.EMOTIONS_
        self.num_classes = len(self.labelsDict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setting up input shape
        self.__version__ = version
        self.example_0, self.ex0_songid, self.ex0_filename, self.ex0_label, self.ex0_label_coords = train_dl.dataset[0]
        self.input_shape = self.example_0.shape
        print(f'Setting up input shape for the Model train_dl __getitem__:'
              f'\n\twaveform: {self.example_0}'
              f'\n\t\tself.input_shape {self.input_shape}'
              f'\n\t\ttype: {type(self.example_0)}'
              f'\n\tsong_id: {self.ex0_songid}'
              f'\n\tfilename: {self.ex0_filename}'
              f'\n\tlabel: {self.ex0_label}'
              f'\n\tcoordinates in dataframe: {self.ex0_label_coords}')

        self.kernel_features_maps = CNNHyperParams.get('kernel_features_maps')
        self.kernel_size = CNNHyperParams.get('kernel_size')
        self.kernel_shift = int(CNNHyperParams.get('kernel_shift'))
        print(f'\n\n{self.name} will run on {self.device}')

        self._nGRU_Layers = n_gru

        # Network definition
        self.conv1 = conv1DBlock(in_channels=1, out_channels=self.kernel_features_maps, kernel_size=self.kernel_size,
                                 kernel_stride=self.kernel_shift)
        self.conv1._L_output = ((self.input_shape[1] - 1 * (self.kernel_size - 1) - 1) // self.kernel_shift) + 1
        self.conv1._output_dim = self.conv1._L_output * self.kernel_features_maps

        if version == 'v1':
            self.conv2 = conv1DBlock(in_channels=self.kernel_features_maps, out_channels=8,
                                     kernel_size=self.kernel_size, kernel_stride=self.kernel_shift)
            self.conv2._L_output = ((self.conv1._L_output - 1 * (self.kernel_size - 1) - 1) // self.kernel_shift) + 1
            self.conv2._output_dim = self.conv2._L_output * 8
            self.flatten = nn.Flatten()
            self.clf_fc = nn.Linear(self.conv2._output_dim, self.num_classes, bias=False)
        elif version == 'v2' and self._nGRU_Layers is not None:
            self.conv2 = conv1DBlock(in_channels=self.kernel_features_maps, out_channels=8,
                                     kernel_size=self.kernel_size, kernel_stride=self.kernel_shift)
            self.conv2._L_output = ((self.conv1._L_output - 1 * (self.kernel_size - 1) - 1) // self.kernel_shift) + 1
            self.conv2._output_dim = self.conv2._L_output * 8
            self.gruLayers = GRUBlock(input_size=self.conv2._L_output, hidden_size=self.conv2._L_output,
                                      num_layers=self._nGRU_Layers,
                                      batch_first=True, dropout=0, bidirectional=False)
            self.flatten = nn.Flatten()
            self.clf_fc = nn.Linear(self.gruLayers._OUTPUT_DIM, self.num_classes)

        else:
            self.flatten = nn.Flatten()
            self.clf_fc = nn.Linear(self.conv1._output_dim, self.num_classes, bias=False)

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

        elif self.version == 'v2':
            x = self.conv2(x)
            x = self.gruLayers(x)

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
                      stride=self._kernel_stride, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm1d(num_features=out_channels),
            # nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.kernel_shift),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        for layer in self._block:
            x = layer(x)
        return x


class GRUBlock(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=False, batch_first=True, dropout=None,
                 bidirectional=False):
        super().__init__()
        """
        :param input_size: if batch_first = True -> input_size (batch, seq, feature) instead of (seq, batch, feature)
        :param hidden_size: #features in the hidden state
        :param num_layers:
        :param batch_first:
        :param dropout: if non-zero introduces a Dropout layer on the outputs of each GRU layer except the last.
        :param bidirectional:
        """
        self._block = nn.Sequential()

        for i in range(0, num_layers):
            name = 'GRULayer'+str(i+1)
            self._block.add_module(name=name, module=nn.GRU(input_size, hidden_size,
                                    num_layers, bias, batch_first, dropout, bidirectional))


        self._OUTPUT_DIM = hidden_size

    def forward(self, x):
        for layer in self._block:
            x = layer(x)
        return x
