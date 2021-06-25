import os, sys
from ..Model.TorchModel import TorchModel
from ..DatasetMusic2emotion.tools import utils as u
import datetime
from .Benchmark import Benchmark
import torch
import torch.cuda as cuda
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..Model.TorchModel import TorchModel
import matplotlib.pyplot as plt
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPTDataLoader, emoMusicPTSubset, emoMusicPTDataset

PlotColors = {
    'train_loss': '#30A666',  # verde scuro
    'test_loss': '#38F58F',  # verde chiaro
    'train_acc': '#A8251E',  # rosso scuro
    'test_acc': '#F55750',  # rosso chiaro
    'eval_loss': '#213E46',  # blu scuro
    'eval_acc': '#B1E2F0'  # azzurro
}
TrainingSettings = {
    "batch_size": 4,
    "epochs": 40,
    "learning_rate": 0.00001,
    "stopping_rate": 1e-7,
    "weight_decay": 1e-6,
    "momentum": 0.8
}

TrainingPolicies = {
    "monitor": 'val_loss',
    "mode": 'min',
    "factor": 0.9,
    "patience": 20,
    "min_lr": 0.000001,
    "verbose": 1
}

TrainSavingsPolicies = {
    "plot_save_dir": 'pytorch_outs/plots',
    "save_directory": 'pytorch_outs/best_models',
    "tensorboard_outs": 'pytorch_outs/tb_outputs',
    "monitor": 'val_categorical_accuracy',
    "quiet": 0,
    "verbose": 1
}

TestingSettings = {}
TestPolicies = {}
TestSavingPolicies = {}


class Runner(object):
    """
    This is the class which handle the model lifecycle
    :param _model:
    """

    def __init__(self, _model: TorchModel):
        self.SUCCESS = 0
        self.FAILURE = -1
        self.model = _model
        self.device = self.model.device
        self.settings = TrainingSettings
        self.train_policies = TrainingPolicies
        self.save_policies = TrainSavingsPolicies
        self.learning_rate = self.settings.get('learning_rate')
        self.stopping_rate = self.settings.get('stopping_rate')
        self.tensorboard_outs_path = os.path.join(self.model.save_dir, TrainSavingsPolicies.get('tensorboard_outs'))
        self.models_save_dir = os.path.join(self.model.save_dir, TrainSavingsPolicies.get('save_directory'))
        # Write the graph to be read on Tensorboard
        SummaryWriter()
        self.writer = SummaryWriter(self.tensorboard_outs_path, filename_suffix=self.model.name)
        example = self.model.emoMusicPTDataset[0]
        if self.model.emoMusicPTDataset.slice_mode:
            self.writer.add_graph(self.model, example[0].reshape(1, 1, 22050))
        else:
            self.writer.add_graph(self.model, example[0].reshape(1, 1, 1345050))
        self.writer.close()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.settings.get('weight_decay'))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=self.train_policies.get('mode'),
                                           factor=self.train_policies.get('factor'),
                                           patience=self.train_policies.get('patience'), verbose=True)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                                   reduction='mean')

    def run(self, dataloader, current_epoch, mode='train', slice_mode=False):
        """
        Called one time for each epoch. It has to parse the whole dataset each time.
        :param current_epoch:
        :param dataloader:
        :param mode: train or eval
        :param slice_mode: True or False
        :return:
        """
        self.model.train() if mode == 'train' else self.model.eval()

        print(f'[Runner.run()] call for epoch: {current_epoch} / {self.settings.get("epochs")}')
        if slice_mode:
            '''
            input of conv1D is (1, 1, 22050) take prediction for every slice (61 in one song)
            when in slice_mode dataset expects indices ranging from 0 to 45357 such that:
            - pass the song_index then convert that index in slice_index and loop 61 times
            '''
            epoch_loss = 0.0
            epoch_acc = 0.0

            model = u.set_device(self.model, self.device)
            train_indices = dataloader.sampler.data_source.indices
            train_indices.sort()
            s = dataloader.dataset[54]
            for song_idx in train_indices:
                start_idx, end_idx = dataloader.song_idx_to_slices_range(song_idx)
                for i in range(start_idx, end_idx):
                    audio_segment, song_id, filename, emotion_label, coords = dataloader.dataset[i]

            for batch, (audio_segment, song_id, filename, label, coords) in enumerate(dataloader):

                # score is pure logits, since I'm using CrossEntropyLoss it will do the log_softmax of the logits
                score, flatten = self.model(audio_segment)

                loss = self.criterion(score, label)
                acc = self.accuracy(score, label)

                self.print_prediction(current_epoch, song_id, filename, label, score)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += score.size(0) * loss.item()
                epoch_acc += score.size(0) * acc

            epoch_loss = epoch_loss / len(dataloader.dataset)
            epoch_acc = epoch_acc / len(dataloader.dataset)
        else:
            '''
            the input of conv1d is (1, 1, 1345050).
            When slice_mode is off dataset expects indices ranging from 0 to 743 such that:
            - pass the song_index and retrieve the whole song
            '''
            epoch_loss = 0.0
            epoch_acc_running_corrects = 0.0

            # model = u.set_device(self.model, self.device)

            for batch, (song_data, song_id, filename, dominant_label, coords) in enumerate(dataloader):
                if cuda.is_available() and cuda.device_count() > 0:
                    song_data = song_data.to('cuda')
                    dominant_label = dominant_label.to('cuda')
                    self.model = self.model.to('cuda')

                    # clear gradients
                if mode == 'train':
                    self.optimizer.zero_grad()

                # score is pure logits, since I'm using CrossEntropyLoss it will do the log_softmax of the logits
                # compute outputs
                score, flatten = self.model(song_data)

                # print prediction info about first, second and every 50 epochs
                self.print_prediction(current_epoch, song_id, filename, dominant_label, score)

                loss = self.criterion(score, dominant_label)
                epoch_acc_running_corrects += self.accuracy(score, dominant_label)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                    # self.optimizer.zero_grad()

                epoch_loss += score.size(0) * loss.item()

            epoch_loss = epoch_loss / float(len(dataloader))
            epoch_acc = epoch_acc_running_corrects / float(len(dataloader))

        return epoch_loss, epoch_acc

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

    def accuracy(self, source, target):
        _, preds = torch.max(source, 1)
        target = target.long().cpu()
        correct = torch.sum(preds == target).data.item()
        return correct / float(source.size(0))

    def train(self):
        train_done = False
        self.settings = TrainingSettings
        self.train_policies = TrainingPolicies
        self.save_policies = TrainSavingsPolicies

        t = Benchmark("[Runner] train call")
        print(f'Starting training loop for {self.settings.get("epochs")} epochs')
        t.start_timer()

        train_losses = np.zeros(self.settings.get('epochs'))
        train_accuracies = np.zeros(self.settings.get('epochs'))

        if self.model.emoMusicPTDataset.slice_mode:
            for epoch in range(self.settings.get('epochs')):
                print(f'[Runner.run(train_dl, {epoch + 1}, slice_mode=True)] called by Runner.train()')
                train_loss, train_acc = self.run(self.model.train_dataloader, epoch + 1, mode='train', slice_mode=True)
                # Store epoch stats
                train_losses[epoch] = train_loss
                train_accuracies[epoch] = train_acc

                print(f'[Runner.train()]Epoch: {epoch + 1}/{self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {(100 * train_acc):.4f} %')

                if self.early_stop(train_loss, epoch + 1):
                    break
            train_done = True

        else:
            for epoch in range(self.settings.get('epochs')):
                print(f'[Runner.run(train_dl, {epoch + 1}, slice_mode=False)] called by Runner.train()')
                train_loss, train_acc = self.run(self.model.train_dataloader, epoch + 1, 'train', slice_mode=False)
                # Store epoch stats
                train_losses[epoch] = train_loss
                train_accuracies[epoch] = train_acc

                print(f'[Runner.train()] Epoch: {epoch + 1}/{self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {(100 * train_acc):.4f} %')

                if self.early_stop(train_loss, epoch + 1):
                    break
            train_done = True

        if train_done:
            print(f'[Runner: {self}] Training finished. Going to test the network')
            # Print training statistics
            self.plot_scatter_training_stats(train_losses, train_accuracies, self.settings.get('epochs'), mode='train')
            best_acc, at_epoch = [np.amax(train_accuracies), np.where(train_accuracies == np.amax(train_accuracies))[0]]
            print(f'[Runner.train() -> train_done!]\n\tBest accuracy: {best_acc} at epoch {at_epoch}\n')
            t.end_timer()
            return self.SUCCESS
        else:
            t.end_timer()
            return self.FAILURE

    def eval(self):
        eval_done = False
        self.settings = TrainingSettings
        self.train_policies = TrainingPolicies
        self.save_policies = TrainSavingsPolicies

        t = Benchmark("[Runner] eval call")
        print(f'Starting evaluation for 1 epoch')
        t.start_timer()

        if self.model.emoMusicPTDataset.slice_mode:
            print(f'[Runner.run(test_dl, 1, slice_mode=True)] called by Runner.eval()')
            test_loss, test_acc = self.run(self.model.test_dataloader, 1, mode='eval', slice_mode=True)
            eval_done = True

        else:
            print(f'[Runner.run(test_dl, 1, slice_mode=False)] called by Runner.eval()')
            test_loss, test_acc = self.run(self.model.test_dataloader, 1, 'eval', slice_mode=False)
            eval_done = True

        if eval_done:
            print(f'[Runner.eval(): {self}] Evaluation on test set finished.')
            t.end_timer()
            print(f'[Runner.eval()]Epoch: 1/1\n'
                  f'\tTest Loss: {test_loss:.4f}\n\tTest Acc: {(100 * test_acc):.4f} %')
            return self.SUCCESS
        else:
            return self.FAILURE

    def plot_scatter_training_stats(self, losses, accuracies, epochs, mode=None):
        n_rows = 1
        n_cols = 2
        epochs_axes = np.arange(1, epochs + 1, 1)
        fig, axes = plt.subplots(1, 2)  # 1 row 2 columns -> 2 plots in a row
        if mode == 'train':
            fig.suptitle("Training curves")
        elif mode == 'eval':
            fig.suptitle("Testing curves")
        else:
            print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
            sys.exit(self.FAILURE)
        measures = ['Loss', 'Accuracy']
        for col in range(n_cols):
            ax = axes[col]
            if col == 0:
                if mode == 'train':
                    ax.plot(epochs_axes, losses, PlotColors.get('train_loss'), label=measures[col])
                    ax.set_title(f'Train Loss')
                elif mode == 'eval':
                    ax.plot(epochs_axes, losses, PlotColors.get('test_loss'), label=measures[col])
                    ax.set_title(f'Test Loss')
                else:
                    print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
                    sys.exit(self.FAILURE)
                ax.set_xlabel('epochs')
                ax.set_ylabel('Loss value')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)
            else:
                if mode == 'train':
                    ax.plot(epochs_axes, accuracies * 100, PlotColors.get('train_acc'), label=measures[col])
                    ax.set_title(f'Train Accuracies')
                elif mode == 'eval':
                    ax.plot(epochs_axes, accuracies * 100, PlotColors.get('test_acc'), label=measures[col])
                    ax.set_title(f'Test Accuracies')
                else:
                    print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
                    sys.exit(self.FAILURE)

                ax.set_xlabel('epochs')
                ax.set_ylabel('Accuracies (%)')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)
        plt.tight_layout()
        save_path = os.path.join(self.model.save_dir, TrainSavingsPolicies.get('plot_save_dir'))
        if mode == 'train':
            print(f'Saving..... Train Curves\tto {save_path + "_timestamp_train_curve.png"}\n')
            d = datetime.datetime.now()
            plt.savefig(
                save_path + f"{self.model.name}_{self.model.kernel_features_maps}_{u.format_timestamp(d)}_train_curves.png")
        elif mode == 'eval':
            print(f'Saving..... Test Curves\tto {save_path + "_timestamp_test_curve.png"}\n')
            d = datetime.datetime.now()
            plt.savefig(
                save_path + f"{self.model.name}_{self.model.kernel_features_maps}_{u.format_timestamp(d)}_test_curves.png")
        else:
            print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
            sys.exit(self.FAILURE)

        plt.show()

    def print_prediction(self, current_epoch, song_id, filename, label, score):
        if current_epoch - 1 == 0 or (current_epoch - 1) % 20 == 0:
            print(f'[Runner.run()] Epoch: {current_epoch}\n\tPrediction for song_id: {song_id} filename: {filename}')
            _, preds = torch.max(score, 1)
            print(f'\tGround Truth label: {label}\n\tPredicted:{preds}')
