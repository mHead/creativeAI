import numpy as np
import sys
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from ..DatasetMusic2emotion.tools.utils import int_to_one_hot
from sklearn.model_selection import StratifiedShuffleSplit

import re
import os
import pandas as pd

legion = False
verbose = True

if legion:
    _REPO_ROOT = r'/home/mtesta/creativeAI'
else:
    _REPO_ROOT = r'/Users/head/Documents/GitHub/creativeAI'



_MULTI_LABELS_PER_SONG_CSV = r'[labels]emotion_average_dataset_csv/music_emotions_labels.csv'
_SINGLE_LABEL_PER_SONG_CSV = r'[labels]emotion_average_dataset_csv/music_single_emotion_labels.csv'
_MUSIC_DATA_ROOT = r'musicSide_root_data'
_WAV_DIR_RELATIVE = r'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG'

_SAMPLE_RATE = 44100
_N_SLICES_PER_SONG = 61
_SLICE_SIZE = 22050
_SONG_SAMPLES_TOTAL_DURATION = _SLICE_SIZE * _N_SLICES_PER_SONG
_N_CLASSES = 8


class emoMusicPTDataset(Dataset):
    """

    """
    def __init__(self, dataset_root, slice_mode=False):
        """
        This is a map-style dataset because it implements the __getitem__ and __len__ protocols
        it can be accessed as dataset[n] which read the idx-th sample (the song chunk) and its corresponding label.

        slice_mode:
            if True the __getitem__(self, n)
                returns the slice of the song thinking n in (0, 744x61) with its label (the slice's one)
            else
                returns the whole song as a torch.Tensor([1, 1345050])

        """
        super().__init__()
        self.name = 'emoMusicPT: Dataset wrapper for PyTorch'
        self.slice_mode = slice_mode
        self.music_data_root = os.path.join(_REPO_ROOT, _MUSIC_DATA_ROOT)
        print(f'**** Creating {self.name}\n\trepo root: {_REPO_ROOT}\n\tmusic data root: {self.music_data_root}\n\tisInSliceMode? {self.slice_mode} ****')


        # PATHS for 2 csv + audio folder
        self.song_id_dominant_emotion_csv_path = os.path.join(self.music_data_root, _SINGLE_LABEL_PER_SONG_CSV)
        self.song_id_emotions_csv_path = os.path.join(self.music_data_root, _MULTI_LABELS_PER_SONG_CSV)
        self.audio_path = os.path.join(self.music_data_root, _WAV_DIR_RELATIVE)
        self.num_classes = _N_CLASSES
        # Read from filesystem audio filenames + 2 csv
        self.song_id_emotions_labels_frame = pd.read_csv(self.song_id_emotions_csv_path)
        self.single_label_per_song_frame = pd.read_csv(self.song_id_dominant_emotion_csv_path)
        wav_filenames = [f for f in os.listdir(self.audio_path) if not f.startswith('.')]
        wav_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # Actual data to refer
        self.wav_filenames = wav_filenames
        self.labels_song_level = self.single_label_per_song_frame.loc[:, 'emotion_label']
        self.labels_slice_levels = self.song_id_emotions_labels_frame.loc[:, self.song_id_emotions_labels_frame.columns != 'song_id']

        self.print_info()

    def __len__(self):
        if self.slice_mode:
            return len(self.wav_filenames) * _N_SLICES_PER_SONG     # 744 x 61
        else:
            return len(self.wav_filenames)      # 744

    def __getitem__(self, n):
        """
        given n 0 <= n <= len(emoMusicPT)
        if slice_mode on
            len : 744*61 = 45,384 slices
        else
            len : 744 songs

        if slice_mode on:
            loads the single slice as a sample (considered as a slice of 500ms : 22050 real time-series samples)
            returns wav_slice, song_id, emotion_int, dictionary {'row_id', 'col_id'}

            wav_slice: the actual raw data : 22050 timeseries objects torch.Tensor([1, 22050])
            song_id: integer number
            filename: string
            emotion_int: integer number
            dictionary: holds the coordinates for the big csv, to locate the label of the requested sample

            the dictionary will holds row_id, col_id as the coordinates to navigate the big dataframe containing
            col:    0           1                       61
            row: song_id | sample_1500ms | ... | sample_45000ms
            0       2           7           ..          7
            1       3           7           ..          7
            ..      ..          ..          ..          ..

            wav_slice is the (row_id, col_id) corresponding torch.Tensor with shape (1, 22050).

        elif slice_mode off:
            loads the whole song as a sample
            returns whole_song, song_id, emotion_int, row_id

            whole_song: torch.Tensor([1, 1345050])
            song_id: integer number
            filename: string
            emotion_int: integer number
            row_id: coordinate of the song for the small csv

        """
        if self.slice_mode:
            assert self.__len__() > n >= 0

            row_id = n // _N_SLICES_PER_SONG                # song_id_idx
            col_id = n - row_id * _N_SLICES_PER_SONG + 1       # slice_no

            # calculate start - end sample
            start_offset = (col_id if col_id == 0 else col_id - 1) * _SLICE_SIZE
            end_offset = start_offset + _SLICE_SIZE
            # read labels
            song_id = self.song_id_emotions_labels_frame.iloc[row_id, 0]
            emotion_label = self.song_id_emotions_labels_frame.iloc[row_id, col_id]
            if verbose:
                print(f'type song_id {type(song_id)} : {song_id} | type emotion_label : {type(emotion_label)} : {emotion_label}')

            # now read audio associated
            filename = str(song_id)+'.wav'
            if filename in self.wav_filenames:
                filename_path = os.path.join(self.audio_path, filename)
            else:
                print(f'{filename} does not exist in path')
                sys.exit(-1)
            _wav_slice = None
            if os.path.exists(filename_path):
                metadata = torchaudio.info(filename_path)
                if verbose:
                    self.print_metadata(_metadata=metadata, src=filename_path)

                waveform, sample_rate = torchaudio.load(filename_path)
                assert sample_rate == _SAMPLE_RATE
                if verbose:
                    print(f'BEFORE SAMPLE EXTRACTION: type of waveform: {type(waveform)} {waveform.shape}')
                # trim the tensor: from pyTorch to numpy
                waveform_array = waveform.detach().numpy()
                wave_trimmed = waveform_array[0][start_offset:end_offset]
                if verbose:
                    print(f'DURING SAMPLE EXTRACTION: type of waveform: {type(wave_trimmed)} {wave_trimmed.shape}')
                # restore the tensor: from numpy to pyTorch
                _waveform = torch.from_numpy(wave_trimmed)
                _wav_slice = torch.reshape(_waveform, shape=(1, _SLICE_SIZE))
                if verbose:
                    print(f'AFTER SAMPLE EXTRACTION: type of waveform: {type(_wav_slice)} {_wav_slice.shape}')
            else:
                print(f'The file at {filename_path} does not exist')

            del waveform, waveform_array, wave_trimmed, _waveform, filename_path, start_offset, end_offset
            # emotion_one_hot = int_to_one_hot(emotion_label)
            return _wav_slice, song_id, filename, emotion_label, {'row_id': row_id, 'col_id': col_id}

        else:
            assert self.__len__() > n >= 0
            row_id = n
            song_id = self.single_label_per_song_frame.iloc[row_id, 0]
            emotion_label = self.single_label_per_song_frame.iloc[row_id, 1]
            if verbose:
                print(f'type song_id {type(song_id)} : {song_id} | '
                      f'type emotion_label : {type(emotion_label)} : {emotion_label}')
            # now read audio associated
            filename = str(song_id) + '.wav'
            if filename in self.wav_filenames:
                filename_path = os.path.join(self.audio_path, filename)
            else:
                print(f'{filename} does not exist in path')
                sys.exit(-1)

            if os.path.exists(filename_path):
                metadata = torchaudio.info(filename_path)
                if verbose:
                    self.print_metadata(_metadata=metadata, src=filename_path)

                waveform, sample_rate = torchaudio.load(filename_path)
                assert sample_rate == _SAMPLE_RATE
                if verbose:
                    print(f'torchaudio.load SAMPLE EXTRACTION: type of waveform: {type(waveform)} {waveform.shape}')
            else:
                print(f'The file at {filename_path} does not exist')

            del filename_path

            # emotion_one_hot = int_to_one_hot(emotion_label)
            return waveform, song_id, filename, emotion_label, {'row_id': row_id}

    def stratified_song_level_split(self, test_fraction):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=0)
        splits = splitter.split(self.wav_filenames, self.labels_song_level)

        for train_index, test_index in splits:
            print(f'TRAIN INDEX: {train_index} type: {type(train_index)} shape: {train_index.shape}')
            print(f'TEST INDEX: {test_index} type: {type(test_index)} shape: {test_index.shape}')

        return train_index, test_index

    def print_info(self):
        print(f'Name: {self.name}\nData_root:\n'
              f'\tAudio: {self.audio_path}\n'
              f'\tLabels: {self.song_id_dominant_emotion_csv_path}\n\t{self.song_id_emotions_csv_path}\n'
              f'self.labels_song_level: \n{self.labels_song_level} {type(self.labels_song_level)} {len(self.labels_song_level)}\n\n '
              f'self.labels_slice_levels: \n{self.labels_slice_levels} {type(self.labels_slice_levels)} {len(self.labels_slice_levels)}\n\n'
              f'self.wav_filenames: \n{self.wav_filenames} {type(self.wav_filenames)}\nlen: {len(self.wav_filenames)}')

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

# %% emoMusicPTSubset


class emoMusicPTSubset(Subset):
    """
    extends torch.utils.data.Subset(dataset, indices)
    Subset of a dataset a t specified indices.
    :param dataset (Dataset) the whole dataset
    :indices(sequence) indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

# %%


# %% emoMusicPTDataloader class:


class emoMusicPTDataLoader(DataLoader):
    """
    Extends torch.utils.data.DataLoader. Combines a dataset and a sampler and provides an iterable over the given dataset
    """
    def __init__(self, dataset: emoMusicPTSubset, batch_size=1, shuffle=False, num_workers=1):
        super().__init__(dataset)
        '''   
        :param dataset:(torch.data.utils.Dataset) a map-style [implements __getitem__() and __len__()] or iterable-style
        
        :param batch_size: how many samples to load
        
        :param shuffle: if True data are reshuffled at every epoch
        
        :param sampler: strategy to draw samples from the dataset, can be any Iterable with __len__ implemented
        
        :param batch_sampler: like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size,
                                                                                        shuffle, sampler and drop_last
                                                                                        
        :param num_workers: how many subprocess to use for data loading
        
        :param collate_fn: merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading 
                            from a  map-style dataset
                            
        :param pin_memory: if True, data loader will copy Tensors into CUDA pinned memory before returning them.
        
        :param drop_last: if True drops the last incomplete batch
        
        :param timeout: 
        :param worker_init_fn: if not None, this will be called on each worker subprocess with the worker id
        
        :param prefetch_factor: number of samples loaded in advance by each worker: 
                                2 means there will be a total of 2*num_workers samples prefetched across all workers
                                
        :param persistent_workers: If True the data loader will not shutdown the worker processes after a dataset 
                                    has been consumed once. This allows to maintain the workers Dataset instances alive
        
        It represents a python iterable over a dataset, with support for:
            - map-style and iterable-style datasets, 
            - customizing data loading order,
            - automatic batching,
            - single and multi-process data loaing,
            - automatic memory pinning
        '''



# %%