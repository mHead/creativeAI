from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools.utils import format_timestamp

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
print(f'Using torch version: {torch.__version__}')
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPT

__DEVICE = 'cuda'
__N_CLASSES = 8


class TorchModel:
    def __init__(self, dataset: emoMusicPT, save_dir_root, **kwargs):
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
        # self.net = net() #TODO: make the class for the different networks

    def train(self):
        return
    def test(self):
        return

