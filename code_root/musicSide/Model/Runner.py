import os, sys
from ..Model.TorchModel import TorchModel
from ..DatasetMusic2emotion.tools import utils as u
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..Model.TorchModel import TorchModel
from ..DatasetMusic2emotion.emoMusicPT import emoMusicPTDataLoader, emoMusicPTSubset, emoMusicPTDataset


TrainingSettings = {
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.0001,
    "stopping_rate": 1e-5,
    "weight_decay": 1e-6,
    "momentum": 0.9
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
        self.writer = SummaryWriter(self.tensorboard_outs_path)
        example = self.model.emoMusicPTDataset[0]
        self.writer.add_graph(self.model, example[0].reshape(1, 1, 22050))
        self.writer.close()
        # Defining aspects of the model lifecycle
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.settings.get('weight_decay'))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=self.train_policies.get('mode'),
                                           factor=self.train_policies.get('factor'),
                                           patience=self.train_policies.get('patience'), verbose=True)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None,
                                                   reduction='mean')

    def run(self, dataloader, mode='train', slice_mode=False):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0.0
        epoch_acc = 0.0

        model = u.set_device(self.model, self.device)

        for batch, (audio_segment, song_id, filename, label, coords) in enumerate(dataloader):

            # score is pure logits, since I'm using CrossEntropyLoss it will do the log_softmax of the logits
            score, flatten = self.model(audio_segment)

            loss = self.criterion(score, label)
            acc = self.accuracy(score, label)

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += score.size(0) * loss.item()
            epoch_acc += score.size(0) * acc

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_acc = epoch_acc / len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.long().cpu()
        correct = (source == target).sum().item()
        return correct / float(source.size(0))

    def train(self):
        train_done = False
        self.settings = TrainingSettings
        self.train_policies = TrainingPolicies
        self.save_policies = TrainSavingsPolicies

        if self.model.emoMusicPTDataset.slice_mode:

            for epoch in range(self.settings.get('epochs')):
                train_loss, train_acc = self.run(self.model.train_dataloader, mode='train', slice_mode=True)

                print(f'Epoch: {epoch + 1 / self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {train_acc:.4f}')
                train_done = True
                if self.early_stop(train_loss, epoch + 1):
                    train_done = True
                    break

        else:
            for epoch in range(self.setting.get('epochs')):
                train_loss, train_acc = self.run(self.model.train_dataloader, 'train', slice_mode=False)

                print(f'Epoch: {epoch + 1 / self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {train_loss:.4f}\n\tTrain Acc: {train_acc:.4f}')
                train_done = True
                if self.early_stop(train_loss, epoch + 1):
                    train_done = True
                    break

        if train_done:
            print(f'[Runner: {self}] Training finished. Going to test the network')
            test_loss, test_acc = self.run(self.model.test_dataloader, 'eval')
            print(f'Test Accuracy: {(100 * test_acc):.4f}\nTest Loss: {test_loss:.4f}')
            return self.SUCCESS
        else:
            return self.FAILURE

    def eval(self):
        eval_done = False
        self.settings = TrainingSettings
        self.train_policies = TrainingPolicies
        self.save_policies = TrainSavingsPolicies

        if self.model.emoMusicPTDataset.slice_mode:

            for epoch in range(self.settings.get('epochs')):
                test_loss, test_acc = self.run(self.model.train_dataloader, mode='eval', slice_mode=True)

                print(f'Epoch: {epoch + 1 / self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {test_loss:.4f}\n\tTrain Acc: {test_acc:.4f}')
                eval_done = True
                if self.early_stop(test_loss, epoch + 1):
                    eval_done = True
                    break

        else:
            for epoch in range(self.setting.get('epochs')):
                test_loss, test_acc = self.run(self.model.train_dataloader, 'eval', slice_mode=False)

                print(f'Epoch: {epoch + 1 / self.settings.get("epochs")}\n'
                      f'\tTrain Loss: {test_loss:.4f}\n\tTrain Acc: {test_loss:.4f}')
                eval_done = True
                print(f'Test Accuracy: {(100 * test_acc):.4f}\nTest Loss: {test_acc:.4f}')
                if self.early_stop(test_loss, epoch + 1):
                    eval_done = True
                    break

        if eval_done:
            print(f'[Runner: {self}] Training finished. Going to test the network')
            return self.SUCCESS
        else:
            return self.FAILURE

