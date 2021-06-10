from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools.utils import format_timestamp

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
from torch.nn import Module
import torch.cuda as cuda

# cuda.init()
'''
Initialize PyTorch’s CUDA state. 
You may need to call this explicitly if you are interacting with PyTorch via its C API, 
as Python bindings for CUDA functionality will not be available until this initialization takes place.
Ordinary users should not need this, as all of PyTorch’s CUDA methods automatically initialize CUDA state on-demand.

Does nothing if the CUDA state is already initialized.
'''
# some print informations
print(f'****\tTorchModel.py imported****\t****\n')
print(f'Using torch version: {torch.__version__}')

if cuda.is_available():
    print(f'\t- GPUs available: {cuda.device_count()}')
    print(f'\t- Current device index: {cuda.current_device()}')
else:
    print(f'\t- GPUs available: {cuda.device_count()}')
    print(f'\t- Cuda is NOT available\n')

import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPTDataset, emoMusicPTSubset, emoMusicPTDataLoader

TrainingSettings = {
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.0001,
    "weight_decay": 1e-6,
    "momentum": 0.9
}

CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8
}


class TorchModel(Module):
    def __init__(self, train_dl: emoMusicPTDataLoader, test_dl: emoMusicPTDataLoader, save_dir_root, n_classes,
                 **kwargs):
        super(TorchModel, self).__init__()
        """
        :param train_dl -> emoMusicPTDataLoader
        :param test_dl -> emoMusicPTDataLoader
        :param save_dir_root -> where to store results
        :param kwargs: optionals
        """
        self.name = "CNN_conv1D"
        self.save_dir = save_dir_root
        self.train_dataloader = train_dl
        self.test_dataloader = test_dl
        self.num_classes = n_classes
        self.example_0, self.ex0_songid, self.ex0_filename, self.ex0_label, self.ex0_label_coords = train_dl.dataset[0]
        self.input_shape = self.example_0.shape
        print(f'self.input_shape {type(self.example_0)}')
        self.kernel_features_maps = CNNHyperParams.get('kernel_features_maps')
        self.kernel_size = CNNHyperParams.get('kernel_size')
        self.kernel_shift = int(CNNHyperParams.get('kernel_shift'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.name} will run on the following device: {self.device}')

        # Network definition
        self.first_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.kernel_features_maps,
                      kernel_size=(1, self.kernel_size), stride=(1, self.kernel_shift),
                      bias=True),
            nn.BatchNorm1d(CNNHyperParams.get('kernel_features_maps')),
            nn.Dropout(0.25))

        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(CNNHyperParams.get('kernel_features_maps'), self.num_classes))



    def forward(self, x):
        return x

    def train(self):
        """
        each epoch: forward and backward pass of ALL training samples
        batch_size = number of training samples in one forward & backward pass
        n_iterations = number of passes, each pass using [batch_size] number of samples
        e.g. 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch

        :return:
        """
        model = TorchModel()
        return

    def test(self):
        return
