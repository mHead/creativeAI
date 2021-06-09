import os
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from ..DatasetMusic2emotion.tools import utils as u

print('****\t DatasetMusic2emotion imported\t****')


class DatasetMusic2emotion:
    """This is the dataset wrapper class
        params: data_root, train_fraction
        kwargs: {"run_config: string", "preprocess" : bool}
    """
    def __init__(self, data_root, train_frac, **kwargs):
        self.run_configuration = kwargs.get('run_config')
        print(f'Creating an object DatasetMusic2emotion')
        # Settings and paths
        self.splits_done = False
        self.music_data_root = data_root
        self.wav_dir_relative = r'MusicEmo_dataset_raw_wav/clips_45seconds_wav'
        self.wav_dir_relative_preprocessed = r'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed'
        self.emotions_csv_path_relative = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'
        self.emotions_csv_path = os.path.join(self.music_data_root, self.emotions_csv_path_relative)
        self.train_fraction = train_frac
        self.test_fraction = 1 - self.train_fraction
        # Labels
        self.Y, self.song_ids, self.single_label_array = self.extract_labels(self.emotions_csv_path)
        self.num_classes = np.max(self.Y) + 1
        print(f'classes : {self.num_classes} : {self.single_label_array}')
        print(f'For songs id: {self.song_ids}')
        # Data
        self.X, self.example_in_sample_length, self.sample_rate, self.slices_per_song, self.window_500ms_size = self.load_X_dataset(preprocess=kwargs.get('preprocess'))
        self.print(self.splits_done, False)
        # Splits
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.train_song_labels, self.test_song_labels = self.make_splits()
        self.splits_done = True
        self.print(splits_done=self.splits_done, paths_info=False), #

    def print(self, splits_done, paths_info):
        if paths_info:
            print(f'The following project is working with the followings paths:'
                  f'music_data_root is: {self.music_data_root}\n'
                  f'emotions_labels csv path: {self.emotions_csv_path}\n'
                  f'audio source wav relative path: {self.wav_dir_relative}'
                  f'')

        if not splits_done:
            print(f'*****DONE!\t DatasetMusic2emotion.py created!*****')

            print(f'Y labels set:\n\tlen: {len(self.Y)};  shape: {self.Y.shape}\n'
                  f'X examples set\n\tlen: {len(self.X)}; shape: {self.X.shape}\n'
                  f'each example is composed by {self.X.shape[2] * self.X.shape[1]} audio samples\n'
                  f'each input_example lasts 500ms and is composed by {self.X.shape[2]} audio samples\n'
                  f'{self.X.shape[1]} input_examples read in time direction define one among {self.X.shape[0]} songs\n'
                  f'sample rate is {self.sample_rate}\n'
                  f'')
        else:
            print(f'Splits done!\n'
                  f'Training set:\n-X_train:\n\tlen: {len(self.X_train)} type: {type(self.X_train)} shape: {self.X_train.shape}\n'
                  f'-Y_train\n\tlen: {len(self.Y_train)} type: {type(self.Y_train)} shape: {self.Y_train.shape}\n'
                  f'Test set:\n-X_test:\n\tlen: {len(self.X_test)} type: {type(self.X_test)} shape: {self.X_test.shape}\n'
                  f'-Y_test:\n\tlen: {len(self.Y_test)} type: {type(self.Y_test)} shape: {self.Y_test.shape}\n'
                  f'')

        return

    def extract_labels(self, path):
        return u.extract_labels(path)

    def make_splits(self):
        print(f'You are going to split the dataset with followings percentages:\n'
              f'Train-Test splits: {int(self.train_fraction * 100)}-{int(self.test_fraction * 100)}')
        training_length = int(self.X.shape[0] * self.train_fraction)
        test_length = self.X.shape[0] - training_length

        assert training_length + test_length == self.X.shape[0]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_fraction, random_state=0)
        splits = splitter.split(self.X, self.single_label_array)

        for train_index, test_index in splits:
            print(f'TRAIN INDEX: {train_index}')
            print(f'TEST INDEX: {test_index}')

            # song-split level
            x_train = self.X[train_index]
            x_test = self.X[test_index]
            y_train = self.Y[train_index]
            y_test = self.Y[test_index]
            # song level
            train_labels = self.single_label_array[train_index]
            test_labels = self.single_label_array[test_index]
        print(f'X_train.shape {x_train.shape}')
        print(f'X_test.shape: {x_test.shape}')
        print(f'Y_train.shape {y_train.shape}')
        print(f'Y_test.shape: {y_test.shape}')
        print(f'train_labels.shape {train_labels.shape}')
        print(f'test_labels.shape: {test_labels.shape}')



        #check the emotion distribution between train and test (be sure test contains all emotions)
        # check if test_labels contains all elements in train_labels
        assert all(label in test_labels for label in train_labels)


        return x_train, y_train, x_test, y_test, train_labels, test_labels

    # create an array with the same size of the dataset, containing 0 if the sample has to be picked for Train, 1 for Test
    def generate_splits_indexes(self, train_len, test_len):
        indexes = np.zeros(train_len + test_len)
        ratio = int(round((train_len + test_len) / test_len))

        n_training_samples = 0
        n_test_samples = 0
        for i in range(train_len + test_len):
            if i % ratio == 0:
                # pick for test
                indexes[i] = 1
                n_test_samples += 1
            else:
                # pick for train
                indexes[i] = 0
                n_training_samples += 1
        assert n_training_samples == train_len and n_test_samples == test_len

        return indexes

    '''
    returns X (744, 61, 22050), example_in_sample_length: 22050*61, sample_rate, slices_per_song = 61, window_size = 500ms as 22050 samples
    if preprocess = True the preprocessing pipeline is started, otherwise it was executed and read_wavs() just have to load preprocessed wavs into
    MusicEmo_dataset_preprocessed_wav/ directory, under musicSide_root_data
    NOTE: when preprocess is active and we want to save the preprocessed wav, the pipeline ends at trim, so the folder will holds n_slices*n_song files with names
    "song_id+slice_no"
    '''
    def load_X_dataset(self, preprocess):
        if preprocess:
            return u.read_wavs(os.path.join(self.music_data_root, self.wav_dir_relative), preprocess=preprocess)
        else:
            return u.read_preprocessed_wavs(os.path.join(self.music_data_root, self.wav_dir_relative_preprocessed))

    def get_shaped_dataset(self):
        # originally into CNN_BiGRU.__init__()
        x_train = self.X_train.reshape(self.X_train.shape[0] * self.X_train.shape[1], self.X_train.shape[2], 1)
        x_test = self.X_test.reshape(self.X_test.shape[0] * self.X_test.shape[1], self.X_test.shape[2], 1)

        y_train = self.Y_train.reshape(self.Y_train.shape[0] * self.Y_train.shape[1], 1)
        y_test = self.Y_test.reshape(self.Y_test.shape[0] * self.Y_test.shape[1], 1)

        # from [7, 7, 7 ..] to [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1]]
        y_train_one_hot = u.to_one_hot(y_train)
        y_test_one_hot = u.to_one_hot(y_test)

        return x_train, x_test, y_train_one_hot, y_test_one_hot



