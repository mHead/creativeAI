import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from ..DatasetMusic2emotion.tools import va2emotion as va2emo


class TorchM5(nn.Module):
    """
    The following architecture is modeled after the M5 network architecture described in https://arxiv.org/pdf/1610.00087.pdf
    """
    def __init__(self, dataset, train_dl, test_dl, hyperparams):
        super(TorchM5, self).__init__()
        self.emoMusicPTDataset = dataset
        self.train_dataloader = train_dl
        self.test_dataloader = test_dl
        self.name = 'TorchM5_music2emoCNN_criterion_version'
        save_dir_m5 = os.path.join(self.emoMusicPTDataset._SAVE_DIR_ROOT, self.name)
        if not os.path.exists(save_dir_m5):
            os.mkdir(save_dir_m5)
        self.save_dir = save_dir_m5
        self.labelsDict = va2emo.EMOTIONS_
        self.num_classes = hyperparams.get('n_output')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # kernel setup
        self.n_channel = hyperparams.get("kernel_features_maps")
        self.kernel_features_maps = self.n_channel  # redundant but called during print_training_stats, common to models
        self.n_input = hyperparams.get("n_input")
        self.n_output = hyperparams.get("n_output")
        self.kernel_size = hyperparams.get("kernel_size")
        self.kernel_shift = hyperparams.get("kernel_shift")

        # setting up input shape
        self.example_0, self.ex0_songid, self.ex0_filename, self.ex0_label, slice_no = self.train_dataloader.dataset[0]
        self.input_shape = self.example_0.shape
        print(f'Setting up input shape for the Model train_dl __getitem__:'
              f'\n\twaveform: {self.example_0}'
              f'\n\t\tself.input_shape {self.input_shape}'
              f'\n\t\ttype: {type(self.example_0)}'
              f'\n\tsong_id: {self.ex0_songid}'
              f'\n\tfilename: {self.ex0_filename}'
              f'\n\tlabel: {self.ex0_label}')
        if not self.emoMusicPTDataset.slice_mode:
            print(f'\n\tslice_no: False')
        else:
            print(f'\n\tslice_no: {slice_no}')

        # Network architecture
        self.conv1 = nn.Conv1d(self.n_input, self.n_channel, kernel_size=hyperparams.get("kernel_size"),
                               stride=hyperparams.get("kernel_shift"))
        self.bn1 = nn.BatchNorm1d(self.n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(self.n_channel, self.n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(self.n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(self.n_channel, 2 * self.n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * self.n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * self.n_channel, 2 * self.n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * self.n_channel)
        self.pool4 = nn.MaxPool1d(4)
        if self.name == "TorchM5_music2emoCNN_criterion_version":
            self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * self.n_channel, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        ks = x.shape[-1]
        if isinstance(ks, torch.Tensor):
            ks = ks.item()

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