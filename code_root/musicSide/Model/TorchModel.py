from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools.utils import format_timestamp

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
import torch.cuda as cuda

#cuda.init()
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
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPT


TrainingSettings = {
    "batch_size": 32,
    "epochs": 200,
}

CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8,
    "learning_rate": 0.0001,
    "weight_decay": 1e-6,
    "momentum": 0.9
}

class TorchModel(nn.Module):
    def __init__(self, dataset: emoMusicPT, save_dir_root, **kwargs):
        super().__init__()
        """
        :param dataset: emoMusicPT object which overrides torch.utils.data.Dataset
        :param save_dir: root of outputs
        :param kwargs: optionals
        """
        self.name = "CNN_conv1D"
        self.save_dir = save_dir_root
        self.dataset = dataset
        self.num_classes = self.dataset.num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.name} will run on the following device: {self.device}')
        self.epochs = TrainingSettings.get('epochs')
        self.batch_size = TrainingSettings.get('batch_size')
        # self.net = net() #TODO: make the class for the different networks

    def train(self):
        """
        each epoch: forward and backward pass of ALL training samples
        batch_size = number of training samples in one forward & backward pass
        n_iterations = number of passes, each pass using [batch_size] number of samples
        e.g. 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch

        :return:
        """

        return
    def test(self):
        return

