import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from ..DatasetMusic2emotion.tools import va2emotion as va2emo


class MFCC_baseline:
    """
     wrapper for MFCC task model

    """
    _slice_mode = False

    def __init__(self, verbose=False, hyperparams=None):
        self.name = 'MFCC-baseline'
        self.hyperparams = hyperparams
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.labelsDict = va2emo.EMOTIONS_

        # properties
        self._save_dir = r''
        self.mfcc_features_dict_ex0 = None
        self.example_0 = self.mfcc_features_dict_ex0['waveform']
        self.ex0_songid = None
        self.ex0_filename = None
        self.ex0_label = None
        self.slice_no = None

        self.hyperparams = hyperparams
        self.set_slice_mode(self.hyperparams['slice_mode'])
        self.num_classes = self.hyperparams.get('n_output')
        self.drop_out = self.hyperparams.get("dropout")
        self.drop_out_p = self.hyperparams.get("dropout_p")
        # TODO which model for MFCC?
        self.model = models.resnet18(pretrained=True)

    @classmethod
    def set_slice_mode(cls, sl):
        cls._slice_mode = sl

    @classmethod
    def slice_mode(cls):
        return cls._slice_mode

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