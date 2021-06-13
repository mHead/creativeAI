from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools import utils as u
from ..DatasetMusic2emotion.tools.utils import format_timestamp

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
                 save_dir_root, n_classes,
                 **kwargs):
        super(TorchModel, self).__init__()
        """
        :param train_dl -> emoMusicPTDataLoader
        :param test_dl -> emoMusicPTDataLoader
        :param save_dir_root -> where to store results
        :param kwargs: optionals
        """
        self.name = "CNN_1conv1D"
        self.save_dir = save_dir_root
        self.emoMusicPTDataset = dataset
        self.train_dataloader = train_dl
        self.test_dataloader = test_dl
        self.num_classes = n_classes
        self.labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
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
        self.first_conv1d = nn.Conv1d(in_channels=1, out_channels=self.kernel_features_maps,
                                      kernel_size=self.kernel_size, stride=self.kernel_shift,
                                      bias=False)
        self.ReLU = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(CNNHyperParams.get('kernel_features_maps'))
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        self.clf_head = nn.Linear(self.conv1d_output_dim, self.num_classes, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.first_conv1d(x)
        x = self.ReLU(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        flatten = self.flatten(x)
        logits = self.clf_head(flatten)

        return logits, flatten
