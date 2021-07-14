import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from ..DatasetMusic2emotion.tools import va2emotion as va2emo


class TorchM11(nn.Module):
    """
    The following architecture is modeled after the M5 network architecture described in https://arxiv.org/pdf/1610.00087.pdf
    """
    _slice_mode = False

    def __init__(self, verbose=False, hyperparams=None):
        super(TorchM11, self).__init__()
        self.name = 'TorchM11_music2emoCNN_criterion_version'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labelsDict = va2emo.EMOTIONS_

        # properties
        self._save_dir = r''
        self.example_0 = None
        self.ex0_songid = None
        self.ex0_filename = None
        self.ex0_label = None
        self.slice_no = None

        self.hyperparams = hyperparams
        self.set_slice_mode(self.hyperparams['slice_mode'])
        self.num_classes = self.hyperparams.get('n_output')
        self.drop_out = self.hyperparams.get("dropout")
        self.drop_out_p = self.hyperparams.get("dropout_p")

        self.weights_list = []
        self.biases = []

        # kernel setup
        self.n_channel = self.hyperparams.get("kernel_features_maps")
        self.kernel_features_maps = self.n_channel  # redundant but called during print_training_stats, common to models
        self.n_input = self.hyperparams.get("n_input")
        self.n_output = self.hyperparams.get("n_output")
        self.kernel_size = self.hyperparams.get("kernel_size")
        self.kernel_shift = self.hyperparams.get("kernel_shift")

        if verbose:
            self.print_model_info()

        # Network architecture
        if self.hyperparams is None:
            print('!!!!backbone not created!!!!')
            return
        else:
            self.conv1 = nn.Conv1d(self.n_input, self.n_channel, kernel_size=self.hyperparams.get("kernel_size"),
                                   stride=self.hyperparams.get("kernel_shift"))
            self.bn1 = nn.BatchNorm1d(self.n_channel)
            self.pool1 = nn.MaxPool1d(4)
            if self.drop_out:
                self.dropout1 = nn.Dropout(self.drop_out_p)

            self.conv2 = nn.Conv1d(self.n_channel, self.n_channel, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(self.n_channel)
            self.conv3 = nn.Conv1d(self.n_channel, self.n_channel, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(self.n_channel)
            self.pool2 = nn.MaxPool1d(4)
            if self.drop_out:
                self.dropout2 = nn.Dropout(self.drop_out_p)

            self.conv4 = nn.Conv1d(self.n_channel, 2 * self.n_channel, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm1d(2 * self.n_channel)
            self.conv5 = nn.Conv1d(2 * self.n_channel, 2 * self.n_channel, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm1d(2 * self.n_channel)
            if self.slice_mode():
                self.pool3 = nn.MaxPool1d(2)
            else:
                self.pool3 = nn.MaxPool1d(4)
            if self.drop_out:
                self.dropout3 = nn.Dropout(self.drop_out_p)

            self.conv6 = nn.Conv1d(2 * self.n_channel, 4 * self.n_channel, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm1d(4 * self.n_channel)
            self.conv7 = nn.Conv1d(4 * self.n_channel, 4 * self.n_channel, kernel_size=3, padding=1)
            self.bn7 = nn.BatchNorm1d(4 * self.n_channel)
            self.conv8 = nn.Conv1d(4 * self.n_channel, 4 * self.n_channel, kernel_size=3, padding=1)
            self.bn8 = nn.BatchNorm1d(4 * self.n_channel)
            if self.slice_mode():
                self.pool4 = nn.MaxPool1d(2)
            else:
                self.pool4 = nn.MaxPool1d(4)
            if self.drop_out:
                self.dropout4 = nn.Dropout(self.drop_out_p)

            self.conv9 = nn.Conv1d(4 * self.n_channel, 8 * self.n_channel, kernel_size=3, padding=1)
            self.bn9 = nn.BatchNorm1d(8 * self.n_channel)
            self.conv10 = nn.Conv1d(8 * self.n_channel, 8 * self.n_channel, kernel_size=3, padding=1)
            self.bn10 = nn.BatchNorm1d(8 * self.n_channel)
            if self.drop_out:
                self.dropout5 = nn.Dropout(self.drop_out_p)

            if self.name == "TorchM11_music2emoCNN_criterion_version":
                self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(8 * self.n_channel, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        if self.drop_out:
            x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool2(x)
        if self.drop_out:
            x = self.dropout2(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool3(x)
        if self.drop_out:
            x = self.dropout3(x)

        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.pool4(x)
        if self.drop_out:
            x = self.dropout4(x)

        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        x = self.conv10(x)
        x = F.relu(self.bn10(x))
        if self.drop_out:
            x = self.dropout5(x)

        ks = x.shape[-1]
        if isinstance(ks, torch.Tensor):
            if self.device.type == 'cpu':
                ks = ks.item()
            else:
                ks = ks.item()  # stackoverflow try ks[0].item() is wrong
                print(f'ks: {ks}, type: {type(ks)}')

        x = F.avg_pool1d(x, kernel_size=ks)

        if self.name == 'TochM5_music2emoCNN':
            # do as Documentation
            x = x.permute(0, 2, 1)
            x = self.fc1(x)
            return F.log_softmax(x, dim=2)
        else:
            # train with criterion = CategoricalCrossEntropyLoss
            flatten = self.flatten(x)
            logits = self.fc1(flatten)
            return logits, flatten

    def print_model_info(self):
        print(f'Setting up input shape for the Model train_dl __getitem__:'
              f'\n\twaveform: {self.example_0}'
              f'\n\t\tself.input_shape {self.input_shape}'
              f'\n\t\ttype: {type(self.example_0)}'
              f'\n\tsong_id: {self.ex0_songid}'
              f'\n\tfilename: {self.ex0_filename}'
              f'\n\tlabel: {self.ex0_label}')
        if not self.slice_mode:
            print(f'\n\tslice_no: False')
        else:
            print(f'\n\tslice_no: {self.slice_no}')

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
