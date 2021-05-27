import os
from ..DatasetMusic2emotion.tools import utils as u


class DatasetMusic2emotion:

    def __init__(self, _music_data_root, **kwargs):
        print(f'Creating an object DatasetMusic2emotion')
        self.music_data_root = _music_data_root
        self.wav_dir_relative = r'MusicEmo_dataset_raw_wav/clips_45seconds_wav'
        self.emotions_csv_path_relative = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'

        self.emotions_csv_path = os.path.join(self.music_data_root, self.emotions_csv_path_relative)
        self.emotions_label_df = u.read_labels(self.emotions_csv_path)

        self.print_path_info()
        self.Y = self.extract_labels()

        self.X, self.example_in_sample_length, self.sample_rate, self.slices_per_song, self.window_500ms_size = u.read_wavs(
            os.path.join(self.music_data_root, self.wav_dir_relative), preprocess=True)

        self.print_data_info()

    def print_path_info(self):
        print(f'The following project is working with the followings paths:'
              f'music_data_root is: {self.music_data_root}\n'
              f'emotions_labels csv path: {self.emotions_csv_path}\n'
              f'audio source wav relative path: {self.wav_dir_relative}'
              f'')
        return

    def print_data_info(self):
        print(f'*****DONE!\t DatasetMusic2emotion.py created!*****')

        print(f'Y labels set:\n\tlen: {len(self.Y)};  shape: {self.Y.shape}\n'
              f'X examples set\n\tlen: {len(self.X)}; shape: {self.X.shape}\n'
              f'each example is composed by {self.X.shape[2]*self.X.shape[1]} audio samples\n'
              f'each input_example lasts 500ms and is composed by {self.X.shape[2]} audio samples\n'
              f'{self.X.shape[1]} input_examples read in time direction define one among {self.X.shape[0]} songs\n'
              f'sample rate is {self.sample_rate}\n'
              f'')
        return

    def extract_labels(self):
        return u.extract_labels(self.emotions_label_df)
