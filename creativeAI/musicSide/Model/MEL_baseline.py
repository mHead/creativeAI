import os
import sys

import torch
import torchvision
from torchvision import models
import torch.nn as nn
from ..DatasetMusic2emotion.tools import va2emotion as va2emo


class MEL_baseline:
    """
    This is a wrapper class for the resnet18 to perform a multi-class task: classify spectrograms based on emotion labels
    """

    def __init__(self, verbose=False, hyperparams=None):
        self.model = models.resnet18(pretrained=True)
        self.name = 'MEL-spectrogram-resnet18'
        self.hyperparams = hyperparams
        self.name = self.name + f'_weight_mode_{self.hyperparams.get("criterion_weight_mode")}'
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.labelsDict = va2emo.EMOTIONS_

        # properties
        self._save_dir = r''
        self.example_0 = None
        self.ex0_songid = None
        self.ex0_filename = None
        self.ex0_label = None
        self.slice_no = None

        self.num_classes = self.hyperparams.get('n_output')
        self.drop_out = self.hyperparams.get("dropout")
        self.drop_out_p = self.hyperparams.get("dropout_p")

        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size[0], stride=self.model.conv1.stride[0], padding=self.model.conv1.padding[0])
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(*[nn.Dropout(p=self.hyperparams['dropout_p']), nn.Linear(self.num_features, self.hyperparams['n_output'])])


    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, path):
        self._save_dir = os.path.join(path, self.name)
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

    @property
    def ex0(self):
        return self.example_0

    @ex0.setter
    def ex0(self, ex0):
        self.example_0 = ex0

    @property
    def ex0_sid(self):
        return self.ex0_songid

    @ex0_sid.setter
    def ex0_sid(self, sid):
        self.ex0_songid = sid

    @property
    def ex0_fn(self):
        return self.ex0_filename

    @ex0_fn.setter
    def ex0_fn(self, fn):
        self.ex0_filename = fn

    @property
    def ex0_lbl(self):
        return self.ex0_label

    @ex0_lbl.setter
    def ex0_lbl(self, lbl):
        self.ex0_label = lbl

    @property
    def ex0_sn(self):
        return self.slice_no

    @ex0_sn.setter
    def ex0_sn(self, sn):
        self.slice_no = sn