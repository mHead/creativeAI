import os
import numpy as np
from ..DatasetMusic2emotion.tools import utils as u


class DatasetMusic2emotion:
    """This is the dataset wrapper class
        params: data_root, train_fraction
        kwargs: {"run_config: string", "preprocess" : bool}
    """
    def __init__(self, data_root, train_frac, **kwargs):
        print(f'Creating an object DatasetMusic2emotion')
        self.run_configuration = kwargs.get('run_config')

        self.splits_done = False
        self.music_data_root = data_root
        self.wav_dir_relative = r'MusicEmo_dataset_raw_wav/clips_45seconds_wav'
        self.wav_dir_relative_preprocessed = r'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed'
        self.emotions_csv_path_relative = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'

        self.emotions_csv_path = os.path.join(self.music_data_root, self.emotions_csv_path_relative)
        self.emotions_label_df = u.read_labels(self.emotions_csv_path)

        self.Y, self.song_ids = self.extract_labels()
        self.num_classes = np.max(self.Y) + 1

        self.X, self.example_in_sample_length, self.sample_rate, self.slices_per_song, self.window_500ms_size = self.load_X_dataset(preprocess=kwargs.get('preprocess'))

        self.print(self.splits_done, False)

        self.train_fraction = train_frac
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.train_test_indexes = self.make_splits()

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

    def extract_labels(self):
        return u.extract_labels(self.emotions_label_df)

    def make_splits(self):
        print(f'You are going to split the dataset with followings percentages:\n'
              f'Train-Test splits: {int(self.train_fraction * 100)}-{int(100 - self.train_fraction*100)}')
        training_length = int(self.X.shape[0] * self.train_fraction)
        test_length = self.X.shape[0] - training_length

        assert training_length + test_length == self.X.shape[0]

        splits_indexes = self.generate_splits_indexes(training_length, test_length)
        assert len(splits_indexes) == training_length + test_length
        x_train = []
        x_test = []
        y_train = []
        y_test = []

        for i in range(len(splits_indexes)):
            if splits_indexes[i] != 0:  # test
                x_test.append(self.X[i])
                y_test.append(self.Y[i])
            else:  # train
                x_train.append(self.X[i])
                y_train.append(self.Y[i])

        # reshaping
        x_train = np.asarray(x_train).reshape(len(x_train), self.X.shape[1], self.X.shape[2])
        x_test = np.asarray(x_test).reshape(len(x_test), self.X.shape[1], self.X.shape[2])
        y_train = np.asarray(y_train).reshape(len(y_train), self.X.shape[1])
        y_test = np.asarray(y_test).reshape(len(y_test), self.X.shape[1])

        return x_train, y_train, x_test, y_test, splits_indexes

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
        x_train = self.X_train.reshape(self.X_train.shape[0] * self.X_train.shape[1], 1, self.X_train.shape[2])
        x_test = self.X_test.reshape(self.X_test.shape[0] * self.X_test.shape[1], 1, self.X_test.shape[2])

        y_train = self.Y_train.reshape(self.Y_train.shape[0] * self.Y_train.shape[1], 1)
        y_test = self.Y_test.reshape(self.Y_test.shape[0] * self.Y_test.shape[1], 1)

        # from [7, 7, 7 ..] to [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1]]
        y_train_one_hot = u.to_one_hot(y_train)
        y_test_one_hot = u.to_one_hot(y_test)

        return x_train, x_test, y_train_one_hot, y_test_one_hot
