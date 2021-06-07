import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import re
import os
import pandas as pd
legion = False
verbose = True

_MULTI_LABELS_PER_SONG_CSV = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'
_SINGLE_LABEL_PER_SONG_CSV = r'[labels]emotion_average_dataset_csv/music_single_emotion_labels.csv'

if legion:
    _REPO_ROOT = r'/home/mtesta/creativeAI'
else:
    _REPO_ROOT = r'/Users/head/Documents/GitHub/creativeAI'

_MUSIC_DATA_ROOT = r'musicSide_root_data'
_WAV_DIR_RELATIVE = r'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG'

_N_SLICES_PER_SONG = 61
_SLICE_SIZE = 22050
_SONG_SAMPLES_TOTAL_DURATION = _SLICE_SIZE * _N_SLICES_PER_SONG
_N_CLASSES = 8


class emoMusicPT(Dataset):
    # get csv data
    def __init__(self, dataset_root):
        super().__init__()
        self.name = 'Dataset for PyTorch'
        self.music_data_root = os.path.join(_REPO_ROOT, _MUSIC_DATA_ROOT)
        print(f'**** Creating {self.name}\n\trepo root: {_REPO_ROOT}\n\tmusic data root: {self.music_data_root} ****')
        # PATHS for 2 csv + audio folder
        self.song_id_dominant_emotion_csv_path = os.path.join(self.music_data_root, _SINGLE_LABEL_PER_SONG_CSV)
        self.song_id_emotions_csv_path = os.path.join(self.music_data_root, _MULTI_LABELS_PER_SONG_CSV)
        self.audio_path = os.path.join(self.music_data_root, _WAV_DIR_RELATIVE)
        self.song_id_emotions_labels_frame = pd.read_csv(self.song_id_emotions_csv_path)
        self.num_classes = _N_CLASSES

        wav_filenames = [f for f in os.listdir(self.audio_path) if not f.startswith('.')]
        wav_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.wav_filenames = wav_filenames
        self.single_label_per_song_frame = pd.read_csv(self.song_id_dominant_emotion_csv_path)
        self.song_level_labels = self.single_label_per_song_frame.loc[:, 'emotion_label']



    def print_metadata(self, _metadata, src=None):
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        print(" - sample_rate:", _metadata.sample_rate)
        print(" - num_channels:", _metadata.num_channels)
        print(" - num_frames:", _metadata.num_frames)
        print(" - bits_per_sample:", _metadata.bits_per_sample)
        print(" - encoding:", _metadata.encoding)
        print()

    def __len__(self):
        nsongs = os.listdir(self.audio_path)
        self.len = nsongs * _N_SLICES_PER_SONG
        return self.len

    def __getitem__(self, n):
        """
        loads the single sample (considered as a slice of 500ms)
        returns wav_slice, song_id, emotion_int, dictionary {'song_id', 'slice_no', 'sample_index'}

        the dictionary will holds row_id, col_id as the coordinates to navigate the big dataframe containing
        col:    0           1                       61
        row: song_id | sample_1500ms | ... | sample_45000ms
        0       2           7           ..          7
        1       3           7           ..          7
        ..      ..          ..          ..          ..

        wav_slice is the (row_id, col_id) corresponding torch.Tensor with shape (1, 22050).

        """
        assert len(self.song_id_emotions_labels_frame) * _N_SLICES_PER_SONG > n >= 0

        row_id = n // _N_SLICES_PER_SONG                # song_id_idx
        col_id = n - row_id * _N_SLICES_PER_SONG        # slice_no

        # calculate start - end sample
        start_offset = col_id * _SLICE_SIZE
        end_offset = start_offset + _SLICE_SIZE
        # read labels
        song_id = self.song_id_emotions_labels_frame.iloc[row_id, 0]
        emotion_label = self.song_id_emotions_labels_frame.iloc[row_id, col_id]
        if verbose:
            print(f'type song_id {type(song_id)} : {song_id} | type emotion_label : {type(emotion_label)} : {emotion_label}')

        # now read audio associated
        filename = str(song_id)+'.wav'
        filename_path = os.path.join(self.audio_path, filename)
        _wav_slice = None
        if os.path.exists(filename_path):
            metadata = torchaudio.info(filename_path)
            if verbose:
                self.print_metadata(_metadata=metadata, src=filename_path)
            waveform, sample_rate = torchaudio.load(filename_path)

            print(f'BEFORE SAMPLE EXTRACTION: type of waveform: {type(waveform)} {waveform.shape}')
            waveform_array = waveform.detach().numpy()
            wave_trimmed = waveform_array[0][start_offset:end_offset]
            _waveform = torch.from_numpy(wave_trimmed)
            _wav_slice = torch.reshape(_waveform, shape=(1, _SLICE_SIZE))
            print(f'AFTER SAMPLE EXTRACTION: type of waveform: {type(_wav_slice)} {_wav_slice.shape}')
        else:
            print(f'The file at {filename_path} does not exist')

        del filename, waveform, waveform_array, wave_trimmed, _waveform, filename_path, start_offset, end_offset
        return _wav_slice, song_id, emotion_label, {'song_id': row_id, 'slice_no': col_id, 'sample_index': n}

    def stratified_song_level_split(self, test_fraction):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=0)
        splits = splitter.split(self.wav_filenames, self.song_level_labels)

        for train_index, test_index in splits:
            print(f'TRAIN INDEX: {train_index} type: {type(train_index)} shape: {train_index.shape}')
            print(f'TEST INDEX: {test_index} type: {type(test_index)} shape: {test_index.shape}')

        return train_index, test_index


