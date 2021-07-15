import os
import sys
import datetime
import torch
import torch.cuda as cuda
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from .Benchmark import Benchmark
from ..DatasetMusic2emotion.tools import utils as u
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPTDataLoader
from ..Model.TorchM5 import TorchM5
from ..Model.MFCC_baseline import MFCC_baseline
from ..Model.MEL_baseline import MEL_baseline

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

    def __init__(self, _model: TorchM5, _train_dl: emoMusicPTDataLoader, _test_dl: emoMusicPTDataLoader, _bundle, task):
        self.SUCCESS = 0
        self.FAILURE = -1
        self.TASK = task

        self.model = _model
        self.device = self.model.device

        self.settings = _bundle
        self.train_dl = _train_dl
        self.test_dl = _test_dl
        self.learning_rate = self.settings.get('learning_rate')
        self.stopping_rate = self.settings.get('stopping_rate')
        # Save paths
        self.best_model_to_save_path = self.set_saves_path(_bundle.get("best_models_save_dir"))
        self.plots_save_path = self.set_saves_path(_bundle.get("plots_save_dir"))
        self.tensorboard_outs_path = self.set_saves_path(_bundle.get("tensorboard_outs"))

        # %%Write the graph to be read on Tensorboard
        if self.settings.get('run_config') != 'legion' and isinstance(self.model, TorchM5):
            SummaryWriter()
            self.writer = SummaryWriter(self.tensorboard_outs_path, filename_suffix=self.model.name)
            example = self.model.example_0
            if self.model.slice_mode():
                self.writer.add_graph(self.model, example.reshape(1, 1, 22050))
            else:
                self.writer.add_graph(self.model, example.reshape(1, 1, 1345050))
            self.writer.close()
        # %%

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                        weight_decay=self.settings.get('weight_decay'))

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=self.settings.get('mode'),
                                           factor=self.settings.get('factor'),
                                           patience=self.settings.get('patience'), verbose=True)

        self.criterion_weights = torch.Tensor(self.settings.get('criterion_weights'))
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.criterion_weights, size_average=None, ignore_index=-100, reduce=None,
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
        if mode == 'train':
            # print(f'Setting up train dataloader...')
            dataloader = self.train_dl
            # print(f'Entering training phase...\n')
        else:
            # print(f'Setting up test dataloader...')
            preds = []
            ground_truth = []
            dataloader = self.test_dl
            # print(f'Entering evaluation phase...\n')

        for batch, (song_data, song_id, filename, dominant_label, slice_no) in enumerate(dataloader):
            if cuda.is_available() and cuda.device_count() > 0:
                # print(f'cuda is available: moving src, target to cuda and perform forward')
                song_data = song_data.to('cuda')
                dominant_label = dominant_label.to('cuda')
                self.model = self.model.to('cuda')
                self.criterion = self.criterion.to('cuda')

                # clear gradients
            if mode == 'train':
                self.optimizer.zero_grad()

            # score is pure logits, since I'm using CrossEntropyLoss it will do the log_softmax of the logits
            # compute outputs
            if not self.model.name.__contains__('criterion_version'):
                output = self.model(song_data)
                loss = F.nll_loss(output.squeeze(), dominant_label)
            else:
                score, flatten = self.model(song_data)
                if cuda.is_available() and cuda.device_count() > 0:
                    score = score.to('cuda')
                loss = self.criterion(score, dominant_label)

            # print first prediction plus every 'print_preds_every'
            # do not print all predictions, but the ones every 50 batches
            # if mode == 'train' and batch % 10 == 0:
                # self.print_prediction(batch, current_epoch, song_id, slice_no, filename, dominant_label, score)

            pred = self.get_likely_index(score)
            epoch_acc_running_corrects += self.number_of_correct(pred, dominant_label)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
            else:
                preds.append(pred)
                ground_truth.append(dominant_label)

            epoch_loss += score.size(0) * loss.item()

        epoch_loss = epoch_loss / float(len(dataloader))
        epoch_acc = epoch_acc_running_corrects / float(len(dataloader))
        if mode == 'train':
            return epoch_loss, epoch_acc
        else:
            return epoch_loss, epoch_acc, preds, ground_truth

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
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def create_classification_report(self, preds, gt):
        preds_list = [a.squeeze().tolist() for a in preds]
        gt_list = [a.squeeze().tolist() for a in gt]

        preds_list_flattened = []
        gt_list_flattened = []
        for i in preds_list:
            if isinstance(i, list):
                for j in i:
                    preds_list_flattened.append(j)
            elif isinstance(i, int):
                preds_list_flattened.append(i)

        for i in gt_list:
            if isinstance(i, list):
                for j in i:
                    gt_list_flattened.append(j)
            elif isinstance(i, int):
                gt_list_flattened.append(i)

        confusion_matrix_df = pd.DataFrame(confusion_matrix(preds_list_flattened, gt_list_flattened))
        plt.figure()
        fig = sns.heatmap(confusion_matrix_df, annot=True)
        plt.tight_layout()
        ts = datetime.datetime.now()

        plt.savefig(os.path.join(self.plots_save_path,
                                 f'{self.model.name}_conf={self.model.hyperparams["__CONFIG__"]}'
                                 f'_kfm={self.model.kernel_features_maps}'
                                 f'_ksz={self.model.kernel_size}'
                                 f'_ksf={self.model.kernel_shift}'
                                 f'_sm={self.model.slice_mode()}'
                                 f'_bs={self.settings.get("batch_size")}'
                                 f'_ep={self.settings.get("epochs")}'
                                 f"_{u.format_timestamp(ts)}_confusion_matrix.png"))

        # plt.show()
        print(classification_report(gt_list_flattened, preds_list_flattened))
        with open(os.path.join(self.plots_save_path,
                               f'{self.model.name}_{u.format_timestamp(ts)}_eval_report.txt'), 'w') as f:
            f.write(f'Classification report for: {self.model.name}'
                    f'\n\n___________________________________\n\n'
                    f'kernel features maps:\t{self.model.kernel_features_maps}\n'
                    f'kernel size:\t\t{self.model.kernel_size}\n'
                    f'kernel shift:\t\t{self.model.kernel_shift}\n'
                    f'slice mode:\t\t{self.model.slice_mode()}\n'
                    f'lr:\t\t{self.settings.get("learning_rate")}\n'
                    f'bs:\t\t{self.settings.get("batch_size")}\n'
                    f'ep:\t\t{self.settings.get("epochs")}\n'
                    f'\n___________________________________\n')
            f.write(classification_report(gt_list_flattened, preds_list_flattened))

    def train(self):
        train_done = False

        t = Benchmark("[Runner] train call")
        print(f'Starting training loop of {self.model.name} for {self.settings.get("epochs")} epochs')
        print(f'\n\t- The model has {self.count_parameters(self.model)} parameters')
        print(f'\n\t- Slice mode is turned on: {self.model.slice_mode()}')
        print(f'\n\t- Dropout: {self.model.drop_out}\n')
        t.start_timer()

        train_losses = np.zeros(self.settings.get('epochs'))
        train_accuracies = np.zeros(self.settings.get('epochs'))

        for epoch in range(self.settings.get('epochs')):
            # print(f'\n\n[Runner.run(train_dl, {epoch + 1}, {self.model.slice_mode()})] called by Runner.train()\n')
            train_loss, train_acc = self.run(epoch + 1, 'train')
            # Store epoch stats
            train_losses[epoch] = train_loss
            train_accuracies[epoch] = train_acc

            # Store epoch weights and biases
            self.store_weights_and_biases(epoch)

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
        test_loss, test_acc, preds, ground_truth = self.run(1, 'eval')
        eval_done = True

        if eval_done:
            print(f'[Runner.eval(): {self}] Evaluation on test set finished.')
            t.end_timer()

            self.create_classification_report(preds, ground_truth)

            print(f'[Runner.eval()]Epoch: 1/1\n'
                  f'\tTest Loss: {test_loss:.4f}\n\tTest Acc: {(100 * test_acc):.4f} %')
            return self.SUCCESS
        else:
            return self.FAILURE

    def store_weights_and_biases(self, epoch):
        model_children = list(self.model.children())
        epoch_weights = {epoch: []}
        epoch_biases = {epoch: []}

        for i in range(len(model_children)):
            if isinstance(model_children[i], torch.nn.Conv1d) or isinstance(model_children[i], torch.nn.BatchNorm1d) or isinstance(model_children[i], torch.nn.Linear):
                epoch_weights.get(epoch).append(model_children[i].weight)
                epoch_biases.get(epoch).append(model_children[i].bias)

        self.model.weights_list.append(epoch_weights)
        self.model.biases.append(epoch_biases)

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

            plt.savefig(os.path.join(self.plots_save_path,
                                     f'{self.model.name}_{self.model.hyperparams["__CONFIG__"]}'
                                     f'_kfm={self.model.kernel_features_maps}'
                                     f'_ksz={self.model.kernel_size}'
                                     f'_ksf={self.model.kernel_shift}'
                                     f'_sm={self.model.slice_mode()}'
                                     f'_bs={self.settings.get("batch_size")}'
                                     f'_ep={self.settings.get("epochs")}'
                                     f"_{u.format_timestamp(d)}_train_curves.png"))
        elif mode == 'eval':
            print(f'Saving..... Test Curves\tto {self.plots_save_path + "_timestamp_test_curve.png"}\n')
            d = datetime.datetime.now()
            plt.savefig(os.path.join(self.plots_save_path, f'{self.model.name}'
                                                           f'_{self.model.hyperparams["__CONFIG__"]}'
                                                           f'_kfm={self.model.kernel_features_maps}'
                                                           f'_ksz={self.model.kernel_size}'
                                                           f'_ksf={self.model.kernel_shift}'
                                                           f'_sm={self.model.slice_mode()}'
                                                           f'_bs={self.settings.get("batch_size")}'
                                                           f'_ep={self.settings.get("epochs")}'
                                                           f"_{u.format_timestamp(d)}_test_curves.png"))
        else:
            print(f'[Runner.plot_scatter_training_stats() mode error: {mode}]')
            sys.exit(self.FAILURE)

        # plt.show()

    def print_prediction(self, current_batch, current_epoch, song_id, slice_no, filename, label, score):
        # if current_epoch - 1 == 0 or (current_epoch - 1) % self.settings.get("print_preds_every") == 0:
        if (current_epoch - 1) % self.settings.get("print_preds_every") == 0:
            if not self.model.slice_mode():
                print(f'[Runner.run()] Epoch: {current_epoch} - Batch: {current_batch}\n\tPrediction for song_id: {song_id}')
                _, preds = torch.max(score, 1)
                print(f'\tGround Truth label: {label}\n\tPredicted:{preds}\n\n')
            else:
                print(f'[Runner.run()] Epoch: {current_epoch} - Batch: {current_batch}\n\tPrediction for song_id: '
                      f'{song_id} | slice_no: {slice_no}')
                _, preds = torch.max(score, 1)
                print(f'\tGround Truth label: {label}\n\tPredicted:{preds}\n\n')

    def save_model(self, epoch, early_stop=False):
        d = datetime.datetime.now()
        path = self.best_model_to_save_path
        model_type = ''
        if early_stop:
            print(f'Saving checkpoint model...')
            model_type = 'checkpoint'
        else:
            print('Saving best model...')
            model_type = 'best_model'

        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "hyperparams": self.model.hyperparams,
            'model_weights': self.model.weights_list,  # TODO populate list over epochs or just at the end?
            'biases': self.model.biases,
            'runner_settings': self.settings,
            'metadata': {
                'timestamp': u.format_timestamp(d),
                'model_name': self.model.name,
                'model_type': model_type,
                'save_dir': self.model.save_dir,
                'ex0': self.model.ex0,
                'ex0_filename': self.model.ex0_fn,
                'ex0_songid': self.model.ex0_sid,
                'ex0_slice_no': self.model.ex0_sn,
                'ex0_label': self.model.ex0_lbl
            }
        }

        # remember: n_channel = kernel features maps
        path = os.path.join(path, f'{self.model.name}_{self.model.hyperparams["__CONFIG__"]}_kfm={self.model.n_channel}'
                                  f'_ksz={self.model.kernel_size}'
                                  f'_ksf={self.model.kernel_shift}'
                                  f'_sm={self.model.slice_mode()}'
                                  f'_bs={self.settings.get("batch_size")}'
                                  f'_ep={self.settings.get("epochs")}'
                                  f'_ts={u.format_timestamp(d)}_{model_type}.pth')

        torch.save(checkpoint, path)

    def set_saves_path(self, path):
        _path = os.path.join(self.model.save_dir, path)

        if not os.path.exists(_path):
            os.mkdir(_path)

        return _path

    def load_model(self, path):
        loaded_checkpoint = torch.load(path)  # Load the dictionary

        hyperparams = loaded_checkpoint['hyperparams']
        model = TorchM5(hyperparams=hyperparams)  # create a new model with same hyperparams
        model_state_dict = loaded_checkpoint['model']  # load state_dict
        model.load_state_dict(model_state_dict)
        model_weights = loaded_checkpoint['model_weights']  # TODO populate list over epochs or just at the end?
        biases = loaded_checkpoint['biases']  # TODO as well

        model.weights_list = model_weights
        model.biases = biases

        model.eval()
        print('\nModel Parameters:\n')
        for p in model.parameters():
            print(p)

        epoch = loaded_checkpoint['epoch']

        optim = loaded_checkpoint['optim']
        runner_settings = loaded_checkpoint['runner_settings']

        metadata = loaded_checkpoint['metadata']
        optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=runner_settings['weight_decay'])
        optimizer.load_state_dict(optim)

        # .. continue training or use for evaluation
        # print(optimizer.state_dict())

        return model, optim, runner_settings, loaded_checkpoint['metadata']
