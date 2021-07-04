import os
import sys
import datetime
import torch
import torch.cuda as cuda
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from .Benchmark import Benchmark
from ..DatasetMusic2emotion.tools import utils as u
from ..Model.TorchM5 import TorchM5

PlotColors = {
    'train_loss': '#30A666',  # verde scuro
    'test_loss': '#38F58F',  # verde chiaro
    'train_acc': '#A8251E',  # rosso scuro
    'test_acc': '#F55750',  # rosso chiaro
    'eval_loss': '#213E46',  # blu scuro
    'eval_acc': '#B1E2F0'  # azzurro
}


class Runner(object):
    """
    This is the class which handle the model lifecycle
    :param _model:
    :param _bundle dict containing 3 dictionaries: TrainingSetting, TrainingPolicies, TrainSavingsPolicies
    """

    def __init__(self, _model: TorchM5, _bundle):
        self.SUCCESS = 0
        self.FAILURE = -1

        self.model = _model
        self.device = self.model.device

        self.settings = _bundle

        self.learning_rate = self.settings.get('learning_rate')
        self.stopping_rate = self.settings.get('stopping_rate')
        # Save paths
        self.best_model_to_save_path = self.set_saves_path( _bundle.get("best_models_save_dir"))
        self.plots_save_path = self.set_saves_path(_bundle.get("plots_save_dir"))
        self.tensorboard_outs_path = self.set_saves_path(_bundle.get("tensorboard_outs"))

        # %%Write the graph to be read on Tensorboard
        SummaryWriter()
        self.writer = SummaryWriter(self.tensorboard_outs_path, filename_suffix=self.model.name)
        example = self.model.emoMusicPTDataset[0]
        if self.model.emoMusicPTDataset.slice_mode:
            self.writer.add_graph(self.model, example[0].reshape(1, 1, 22050))
        else:
            self.writer.add_graph(self.model, example[0].reshape(1, 1, 1345050))
        self.writer.close()
        # %%

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.settings.get('weight_decay'))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=self.settings.get('mode'),
                                           factor=self.settings.get('factor'),
                                           patience=self.settings.get('patience'), verbose=True)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                                   reduction='mean')

    def run(self, current_epoch, mode='train'):
        """
        Called one time for each epoch. It has to parse the whole dataset each time.
        :param current_epoch:
        :param dataloader:
        :param mode: train or eval
        :param slice_mode: True or False
        :return:

        if not slice_mode:
            the input of conv1d is (1, 1, 1345050).
            When slice_mode is off dataset expects indices ranging from 0 to 743 such that:
            - pass the song_index and retrieve the whole song
        else:
            input of conv1D is  ttc (1, 1, 22050) take prediction for every slice (61 in one song)
            when in slice_mode dataset expects indices ranging from 0 to 45357 such that:
            - pass the index and retrieve slice, labels and metadata
        """
        self.model.train() if mode == 'train' else self.model.eval()
        print(f'[Runner.run()] call for epoch: {current_epoch} / {self.settings.get("epochs")}')

        epoch_loss = 0.0
        epoch_acc_running_corrects = 0.0

        # model = u.set_device(self.model, self.device)
        # if slice_mode=False, slice_no is always 0
        for batch, (song_data, song_id, filename, dominant_label, slice_no) in enumerate(self.model.train_dataloader):
            if cuda.is_available() and cuda.device_count() > 0:
                song_data = song_data.to('cuda')
                dominant_label = dominant_label.to('cuda')
                self.model = self.model.to('cuda')

                # clear gradients
            if mode == 'train':
                self.optimizer.zero_grad()

            # score is pure logits, since I'm using CrossEntropyLoss it will do the log_softmax of the logits
            # compute outputs
            if self.model.name == 'TorchM5_music2emoCNN':
                output = self.model(song_data)
                loss = F.nll_loss(output.squeeze(), dominant_label)
            else:
                score, flatten = self.model(song_data)
                loss = self.criterion(score, dominant_label)

            # print first prediction plus every 10
            if mode == 'train':
                self.print_prediction(current_epoch, song_id, slice_no, filename, dominant_label, score)

            pred = self.get_likely_index(score)
            epoch_acc_running_corrects += self.number_of_correct(pred, dominant_label)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            epoch_loss += score.size(0) * loss.item()

        epoch_loss = epoch_loss / float(len(self.model.train_dataloader))
        epoch_acc = epoch_acc_running_corrects / float(len(self.model.train_dataloader))
        return epoch_loss, epoch_acc

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

    def get_likely_index(self, tensor):
        return tensor.argmax(dim=-1)

    def number_of_correct(self, pred, target):
        return pred.squeeze().eq(target).sum().item() / float(pred.size(0))

    def accuracy(self, source, target):
        _, preds = torch.max(source, 1)
        target = target.long().cpu()
        correct = torch.sum(preds == target).data.item()

        return correct / float(source.size(0))

    def count_parameters(self, _model):
        return sum(p.numel() for p in _model.parameters() if p.requires_grad)

    def train(self):
        train_done = False

        t = Benchmark("[Runner] train call")
        print(f'Starting training loop of {self.model.name} for {self.settings.get("epochs")} epochs')
        print(f'\n\t- The model has {self.count_parameters(self.model)} parameters')
        print(f'\n\t- Slice mode is turned on: {self.model.emoMusicPTDataset.slice_mode}')
        print(f'\n\t- Dropout: {self.model.drop_out}\n')
        t.start_timer()

        train_losses = np.zeros(self.settings.get('epochs'))
        train_accuracies = np.zeros(self.settings.get('epochs'))

        for epoch in range(self.settings.get('epochs')):
            print(f'[Runner.run(train_dl, {epoch + 1}, {self.model.emoMusicPTDataset.slice_mode})] called by Runner.train()')
            train_loss, train_acc = self.run(epoch + 1, 'train')
            # Store epoch stats
            train_losses[epoch] = train_loss
            train_accuracies[epoch] = train_acc

            print(f'[Runner.train()] Epoch: {epoch + 1}/{self.settings.get("epochs")}\n'
                  f'\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {(100 * train_acc):.4f} %')

            if self.early_stop(train_loss, epoch + 1):
                self.save_model(epoch=epoch, early_stop=True)
                break
        train_done = True

        if train_done:
            print(f'[Runner: {self}] Training finished.')
            # Print training statistics
            self.plot_scatter_training_stats(train_losses, train_accuracies, self.settings.get('epochs'), mode='train')
            best_acc, at_epoch = [np.amax(train_accuracies), np.where(train_accuracies == np.amax(train_accuracies))[0]]
            print(f'[Runner.train() -> train_done!]\n\tBest accuracy: {best_acc} at epoch {at_epoch}\n')

            self.save_model(epoch, early_stop=False)

            t.end_timer()
            return self.SUCCESS
        else:
            t.end_timer()
            return self.FAILURE

    def eval(self):
        eval_done = False

        t = Benchmark("[Runner] eval call")
        print(f'Starting evaluation for 1 epoch')
        t.start_timer()

        print(f'[Runner.run(test_dl, 1, slice_mode=False)] called by Runner.eval()')
        test_loss, test_acc = self.run(1, 'eval')
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

        if mode == 'train':
            print(f'Saving..... Train Curves\tto {self.plots_save_path + "_timestamp_train_curve.png"}\n')
            d = datetime.datetime.now()
            plt.savefig(os.path.join(self.plots_save_path, f"{self.model.name}_kfm={self.model.kernel_features_maps}"
                                                           f'_ksz={self.model.kernel_size}'
                                                           f'_ksf={self.model.kernel_shift}'
                                                           f'_sm={self.model.emoMusicPTDataset.slice_mode}'
                                                           f'_bs={self.settings.get("batch_size")}'
                                                           f'_ep={self.settings.get("epochs")}'
                                                           f"_{u.format_timestamp(d)}_train_curves.png"))
        elif mode == 'eval':
            print(f'Saving..... Test Curves\tto {self.plots_save_path + "_timestamp_test_curve.png"}\n')
            d = datetime.datetime.now()
            plt.savefig(os.path.join(self.plots_save_path, f"{self.model.name}"
                                                           f"_kfm={self.model.kernel_features_maps}"
                                                           f'_ksz={self.model.kernel_size}'
                                                           f'_ksf={self.model.kernel_shift}'
                                                           f'_sm={self.model.emoMusicPTDataset.slice_mode}'
                                                           f'_bs={self.settings.get("batch_size")}'
                                                           f'_ep={self.settings.get("epochs")}'
                                                           f"_{u.format_timestamp(d)}_test_curves.png"))
        else:
            print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
            sys.exit(self.FAILURE)

        plt.show()

    def print_prediction(self, current_epoch, song_id, slice_no, filename, label, score):
        #if current_epoch - 1 == 0 or (current_epoch - 1) % self.settings.get("print_preds_every") == 0:
        if (current_epoch - 1) % self.settings.get("print_preds_every") == 0:
            if not self.model.emoMusicPTDataset.slice_mode:
                print(f'[Runner.run()] Epoch: {current_epoch}\n\tPrediction for song_id: {song_id}')
                _, preds = torch.max(score, 1)
                print(f'\tGround Truth label: {label}\n\tPredicted:{preds}\n\n')
            else:
                print(f'[Runner.run()] Epoch: {current_epoch}\n\tPrediction for song_id+slice_no: {song_id}+{slice_no}')
                _, preds = torch.max(score, 1)
                print(f'\tGround Truth label: {label}\n\tPredicted:{preds}\n\n')

    def save_model(self, epoch, early_stop=False):
        d = datetime.datetime.now()
        path = self.best_model_to_save_path

        if early_stop:
            print(f'Saving checkpoint model...')
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            }
            # remember: n_channel = kernel features maps
            path = os.path.join(path, f'{self.model.name}_kfm={self.model.n_channel}'
                                      f'_ksz={self.model.kernel_size}'
                                      f'_ksf={self.model.kernel_shift}'
                                      f'_sm={self.model.emoMusicPTDataset.slice_mode}'
                                      f'_bs={self.settings.get("batch_size")}'
                                      f'_ep={self.settings.get("epochs")}'
                                      f'_ts={u.format_timestamp(d)}_checkpoint_model.pth')
            torch.save(checkpoint, path)
            # to load it
            # loaded_checkpoint = torch.load(path)
            # epoch = loaded_checkpoint['epoch']
            # model = TorchM5(...)
            # optimizer = define optim with lr=0
            # model.load_State_dict(loaded_checkpoint['model_state']
            # optimizer.load_state_dict(loaded_checkpoint['optim_state']
            # .. continue training
            # print(optimizer.state_dict())
        else:
            print(f'Saving best model...')
            path = os.path.join(path, f'{self.model.name}_kfm={self.model.n_channel}'
                                      f'_ksz={self.model.kernel_size}'
                                      f'_ksf={self.model.kernel_shift}'
                                      f'_sm={self.model.emoMusicPTDataset.slice_mode}'
                                      f'_bs={self.settings.get("batch_size")}'
                                      f'_ep={self.settings.get("epochs")}'
                                      f'_ts={u.format_timestamp(d)}_best_model.pth')
            torch.save(self.model.state_dict(), path)
            # in order to load the model we have to
            # loaded_model = TorchM5(...)
            # loaded_model.load_state_dict(torch.load(path))
            # loaded_model.eval()
            # for p in loaded_model.parameters():
            #   print(p)

    def set_saves_path(self, path):
        _path = os.path.join(self.model.save_dir, path)

        if not os.path.exists(_path):
            os.mkdir(_path)

        return _path







