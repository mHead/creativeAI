import os
from ..DatasetMusic2emotion.tools import utils as u


class DatasetMusic2emotion:

    def __init__(self, _music_data_root, train_frac, **kwargs):
        print(f'Creating an object DatasetMusic2emotion')
        self.splits_done = False
        self.music_data_root = _music_data_root
        self.wav_dir_relative = r'MusicEmo_dataset_raw_wav/clips_45seconds_wav'
        self.emotions_csv_path_relative = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'

        self.emotions_csv_path = os.path.join(self.music_data_root, self.emotions_csv_path_relative)
        self.emotions_label_df = u.read_labels(self.emotions_csv_path)

        self.print_path_info()
        self.Y = self.extract_labels()

        self.X, self.example_in_sample_length, self.sample_rate, self.slices_per_song, self.window_500ms_size = u.read_wavs(
            os.path.join(self.music_data_root, self.wav_dir_relative), preprocess=True)

        self.print_data_info(self.splits_done)

        self.train_fraction = train_frac
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.make_splits()

        self.splits_done = True
        self.print_data_info(splits_done=self.splits_done)


    def print_path_info(self):
        print(f'The following project is working with the followings paths:'
              f'music_data_root is: {self.music_data_root}\n'
              f'emotions_labels csv path: {self.emotions_csv_path}\n'
              f'audio source wav relative path: {self.wav_dir_relative}'
              f'')
        return

    def print_data_info(self, splits_done):
        if not splits_done:
            print(f'*****DONE!\t DatasetMusic2emotion.py created!*****')

            print(f'Y labels set:\n\tlen: {len(self.Y)};  shape: {self.Y.shape}\n'
                  f'X examples set\n\tlen: {len(self.X)}; shape: {self.X.shape}\n'
                  f'each example is composed by {self.X.shape[2]*self.X.shape[1]} audio samples\n'
                  f'each input_example lasts 500ms and is composed by {self.X.shape[2]} audio samples\n'
                  f'{self.X.shape[1]} input_examples read in time direction define one among {self.X.shape[0]} songs\n'
                  f'sample rate is {self.sample_rate}\n'
                  f'')
        else:
            print(f'Splits done!'
                  f'Training set: X_train:\n\tlen: {len(self.X_train)} type: {type(self.X_train)} shape: {self.X_train.shape}\n'
                  f'Y_train\n\tlen: {len(self.Y_train)} type: {type(self.Y_train)} shape: {self.Y_train.shape}\n'
                  f'Test set: X_test:\n\tlen: {len(self.X_test)} type: {type(self.X_test)} shape: {self.X_test.shape}\n'
                  f'Y_test:\n\tlen: {len(self.Y_test)} type: {type(self.Y_test)} shape: {self.Y_test.shape}\n'
                  f'')

        return

    def extract_labels(self):
        return u.extract_labels(self.emotions_label_df)

    def make_splits(self):
        print(f'You are going to split the dataset with followings percentages:\n'
              f'Train-Test splits: {self.train_fraction * 10}-{(1-self.train_fraction) * 10}')
        training_length = int(self.X.shape[0] * self.train_fraction)
        test_length = self.X.shape[0] - training_length

        assert training_length + test_length == self.X.shape[0]




        return X_train, Y_train, X_test, Y_test