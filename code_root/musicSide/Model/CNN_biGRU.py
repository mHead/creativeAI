from ..DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from ..DatasetMusic2emotion.tools.utils import format_timestamp

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import Dense, GRU
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv1D, TimeDistributed, Bidirectional

from keras import optimizers

# import keras.backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# Rest
# from scipy.fftpack import fft
# from scipy import signal
# from scipy.io import wavfile


# %% start global variables
CNNHyperParams = {
    "kernel_size": 220,
    "kernel_shift": 110,
    "kernel_features_maps": 8,
    "learning_rate": 0.0001,
    "weight_decay": 1e-6,
    "momentum": 0.9
}

BiGRUParams = {
    "n_units": 8
}

TrainingSettings = {
    "batch_size": 32,
    "epochs": 1000,

}

TrainingPolicies = {
    "monitor": 'val_loss',
    "factor": 0.9,
    "patience": 20,
    "min_lr": 0.000001,
}

SavingsPolicies = {
    "save_directory": 'best_models',
    "monitor": 'val_categorical_accuracy',
    "mode": 'max',
    "quiet": 0,
    "verbose": 1
}


# %% end global variables

# %% 0. Model definition
class CNN_BiGRU:
    def __init__(self, emo_music_dataset_object: DatasetMusic2emotion, save_dir, do_train=False,
                 do_test=False, load_model=False, load_model_path=(None, None), **kwargs):

        self.save_dir = save_dir
        self.dataset = emo_music_dataset_object
        self.num_classes = len(np.unique(np.argmax(self.dataset.Y_train, 1)))

        # TODO: check the X_train = np.array(X_train) and then reshape. It is just to be sure it is an array?
        #  If yes,
        #  I'm ok
        self.X_train = self.dataset.X_train.reshape(self.dataset.X_train.shape[0],
                                                    self.dataset.X_train.shape[1] * self.dataset.X_train.shape[2], 1)
        self.X_test = self.dataset.X_test.reshape(self.dataset.X_test.shape[0],
                                                  self.dataset.X_test.shape[1] * self.dataset.X_test.shape[2], 1)
        self.Y_train = self.dataset.Y_train.reshape(self.dataset.Y_train.shape[0] * self.dataset.Y_train.shape[1], 1)
        self.Y_test = self.dataset.Y_test.reshape(self.dataset.Y_test.shape[0] * self.dataset.Y_test.shape[1], 1)

        self.input_shape = self.X_train.shape[1:]
        self.kernel_features_maps = CNNHyperParams.get('kernel_features_maps')
        self.kernel_size = CNNHyperParams.get('kernel_size')
        self.kernel_shift = CNNHyperParams.get('kernel_shift')
        self.batch_size = CNNHyperParams.get('batch_size')
        self.epochs = TrainingSettings.get('epochs')
        self.learning_rate = CNNHyperParams.get('learning_rate')
        self.weight_decay = CNNHyperParams.get('weight_decay')
        self.momentum = CNNHyperParams.get('momentum')
        self.gru_units = BiGRUParams.get('n_units')

        if load_model:
            if load_model_path[0] is not None and load_model_path[1] is not None:
                self.isTrained, self.model = self.load_model_and_weights(load_model_path)
        else:
            self.model = self.create_model('CNN_BiGRU')
            print(f'Model created!\ntype:{type(self.model)}\n')
            print(f'Model summary:\n{self.model.summary()}')
            self.compile_model(do_train, do_test)

        if do_train:
            # prepare for training
            self.callbacks_lrr, self.callbacks_mcp = self.create_callbacks(True, True)

            self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                          validation_data=(self.X_test, self.Y_test), callbacks=[self.callbacks_lrr,
                                                                                                 self.callbacks_mcp])
        if do_test:
            # prepare for testing
            self.compile_model(do_train, do_test)
            self.best_score = self.evaluate_model()
            self.predictions = self.make_prediction()
        return

    # %% 0. end Model definition

    # %% 1. Model methods
    def create_model(self, _name):
        model = Sequential(name=_name)

        # CNN
        model.add(Conv1D(filters=self.kernel_features_maps, kernel_size=self.kernel_size, strides=self.kernel_shift,
                         padding='same', data_format='channels_last', input_shape=self.input_shape,
                         activation=tf.nn.relu))

        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Dense(16, activation=tf.nn.relu)))

        # Recurrent
        model.add(Bidirectional(GRU(self.gru_units, activation='tanh'), merge_mode='concat'))
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def compile_model(self, train, test):

        _optimizer = keras.optimizers.adam_v2.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.99, amsgrad=False)

        if test:
            self.model.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            print(
                f'Model compiled for evaluation on Test Set with loss: categorical_crossentropy and metrics: accuracy')
            print(f'Model summary:\n{self.model.summary()}')
        elif train:
            self.model.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            print(f'Model compiled for training with loss: categorical_crossentropy and metrics: categorical_accuracy')
            print(f'Model summary:\n{self.model.summary()}')
        else:
            print(f'The model did not compiled due to ambiguous train/test intent')

    def make_prediction(self):
        _preds = self.model.predict(self.X_test, batch_size=TrainingSettings.get('batch_size'), verbose=1)

        preds_argmax = _preds.argmax(axis=1)
        abc = preds_argmax.astype(int).flatten()
        print(f'{abc}')
        return _preds

    def evaluate_model(self, verbose=0):
        best_score = self.model.evaluate(self.X_test, self.Y_test, verbose)
        print(f'The accuracy of the model is {self.model.metrics_names[1], best_score[1] * 100}')
        return best_score

    def create_callbacks(self, lr_reduce=True, model_checkpoint=True):
        lr_reduce_ = 0
        save_model = 0
        if lr_reduce:
            lr_reduce_ = ReduceLROnPlateau(monitor=TrainingPolicies.get('monitor'),
                                           factor=TrainingPolicies.get('factor'),
                                           patience=TrainingPolicies.get('patience'),
                                           min_lr=TrainingPolicies.get('min_lr'), verbose=TrainingPolicies.get('quiet'))

        if model_checkpoint:
            save_model = ModelCheckpoint(os.path.join(self.save_dir, 'best_models/conv1D_BN_D_TDFC_biGRU_FC.h5'),
                                         save_best_only=True, monitor=SavingsPolicies.get('monitor'),
                                         mode=SavingsPolicies.get('max'))

        return lr_reduce_, save_model

    # %% end Model methods

    # %% 2. Print methods
    def print_training_history(self):
        print(f'{max(self.history.history["val_categorical_accuracy"])}')
        self.print_loss()
        self.print_accuracy()
        return

    def print_accuracy(self):
        plt.plot(self.history.history['categorical_accuracy'])
        plt.plot(self.history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return

    def print_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return

    # %% end Evaluations methods

    # %% 3. IO
    def save_to_json(self):
        model_json = self.model.to_json()
        with open(os.path.join(self.save_dir, 'best_models/' + format_timestamp(datetime.now())), 'w') as json_file:
            json_file.write(model_json)
        return

    def load_model_and_weights(self, load_model_path):
        json_file = open(load_model_path[0], 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)

        model.load_weigths(load_model_path[1])
        is_trained = True

        return is_trained, model
    # %% end utilities
