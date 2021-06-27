import sys
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from ..DatasetMusic2emotion.tools import utils as u
from ..DatasetMusic2emotion.tools import va2emotion as va2emo
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

_MULTI_LABELS_PER_SONG_CSV = r'music_emotions_labels.csv'
_SINGLE_LABEL_PER_SONG_CSV = r'music_single_emotion_labels.csv'
_MUSIC_DATA_ROOT = r'musicSide_root_data'

verbose = False

_SAMPLE_RATE = 44100
_N_SLICES_PER_SONG = 61
_SLICE_SIZE = 22050
_SONG_SAMPLES_TOTAL_DURATION = _SLICE_SIZE * _N_SLICES_PER_SONG
_N_CLASSES = 8

SubsetsColors = {
    'train_set': '#30A666',  # verde scuro
    'test_set': '#A8251E',  # verde chiaro
    'val_set': '#B1E2F0',  # rosso scuro
}


class emoMusicPTDataset(Dataset):
    legion = False
    colab = False
    local = False

    def __init__(self, slice_mode=False, env=None):
        super().__init__()
        """
        This is a map-style dataset because it implements the __getitem__ and __len__ protocols
        it can be accessed as dataset[n] which read the idx-th sample and its corresponding label.

        slice_mode:
            if True the __getitem__(self, n)
                returns the slice of the song thinking n in (0, 744x61) with its label (the slice's one)
            else
                returns the whole song as a torch.Tensor([1, 1345050])

        """
        if env is not None:
            self._RUN_CONF = env.get('run_conf')
            self._REPO_ROOT = env.get('repo_root')
            self._CODE_ROOT = env.get('code_root')
            self._AUDIO_PATH = env.get('dataset_root')
            self._SONGID_DOMINANT_EMO_CSV = os.path.join(env.get('labels_root'), _SINGLE_LABEL_PER_SONG_CSV)
            self._SONGID_EMOTIONS_CSV = os.path.join(env.get('labels_root'), _MULTI_LABELS_PER_SONG_CSV)
            self._MUSIC_DATA_ROOT = env.get('music_data_root')
        else:
            print(f'Configuration Environment failed!')
            sys.exit(-1)

        if self._RUN_CONF == 'legion':
            legion = True
        elif self._RUN_CONF == 'colab':
            colab = True
        else:
            local = True

        self.name = 'emoMusicPT: pytorch Dataset wrapper'
        self.slice_mode = slice_mode
        self.num_classes = _N_CLASSES
        print(f'[emoMusicPTDataset.py]\t**** Creating {self.name} ****\n\trepo root: {self._REPO_ROOT}\n\tmusic data root: {self._MUSIC_DATA_ROOT}\n\tslice_mode: {self.slice_mode}\n')


        # CSV path to pd.DataFrame
        self.song_id_emotions_labels_frame = pd.read_csv(self._SONGID_EMOTIONS_CSV)
        self.single_label_per_song_frame = pd.read_csv(self._SONGID_DOMINANT_EMO_CSV)

        wav_filenames = [f for f in os.listdir(self._AUDIO_PATH) if not f.startswith('.')]
        wav_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # Actual data to refer
        self.wav_filenames = wav_filenames
        del wav_filenames
        #self.labels_song_level = self.single_label_per_song_frame.loc[:, 'emotion_label']
        #self.labels_slice_level = self.song_id_emotions_labels_frame.loc[1:, self.song_id_emotions_labels_frame.columns != 'song_id']
        self.labels_slice_level, self.song_ids, self.labels_song_level = u.extract_labels(self._SONGID_EMOTIONS_CSV)

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
                filename_path = os.path.join(self._AUDIO_PATH, filename)
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
            #song_id = self.single_label_per_song_frame.iloc[row_id, 0]
            #emotion_label = self.single_label_per_song_frame.iloc[row_id, 1]

            song_id = self.wav_filenames[n].split('.')[0]
            filename = self.wav_filenames[n]
            emotion_label = self.labels_song_level[n]

            if verbose:
                print(f'type song_id {type(song_id)} : {song_id} | '
                      f'type emotion_label : {type(emotion_label)} : {emotion_label}')
            # now read audio associated
            # filename = str(song_id) + '.wav'
            if len(filename) > 0:
                filename_path = os.path.join(self._AUDIO_PATH, filename)
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
        train_index = []
        test_index = []
        for train_index, test_index in splits:
            print(f'[emoMusicPTDataset.py] TRAIN INDEX: {train_index} type: {type(train_index)} shape: {train_index.shape}')
            print(f'[emoMusicPTDataset.py] TEST INDEX: {test_index} type: {type(test_index)} shape: {test_index.shape}')
        #train_index.sort()
        #test_index.sort()
        print(f'\n[emoMusicPTDataset.py] type train_indexes {type(train_index)}\ntype test_indexes {type(test_index)}')
        return train_index, test_index

    def stratifiedKFold_song_level(self, n_splits, shuffle, random_state=None):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)



    def print_info(self):
        print(f'Name: {self.name}\nData_root:\n'
              f'\tAudio: {self._AUDIO_PATH}\n'
              f'\tLabels: {self._SONGID_DOMINANT_EMO_CSV}\n\t{self._SONGID_EMOTIONS_CSV}\n'
              f'self.labels_song_level: \n{self.labels_song_level}\ntype: {type(self.labels_song_level)}\nlen: {len(self.labels_song_level)}\n\n '
              f'self.labels_slice_level: \n{self.labels_slice_level}\ntype: {type(self.labels_slice_level)}\nlen: {len(self.labels_slice_level)}\n\n'
              f'self.wav_filenames: \n{self.wav_filenames}\ntype: {type(self.wav_filenames)}\nlen: {len(self.wav_filenames)}\n\n')

    def plot_indices_distribution(self, labels, train_indexes, test_indexes, val_indexes=None):
        """
        Collect distrubution per subset
        """
        train_emotion_distribuion = np.zeros(self.num_classes)
        test_emotion_distribution = np.zeros(self.num_classes)


        for train_index in train_indexes:
            train_emotion_distribuion[labels[train_index]] += 1

        for test_index in test_indexes:
            test_emotion_distribution[labels[test_index]] += 1

        if val_indexes is not None:
            val_emotion_distribution = np.zeros(self.num_classes)
            for val_index in val_indexes:
                val_emotion_distribution[labels[test_index]] *= 1

            assert all(val_emotion_distribution) > 0

        assert all(train_emotion_distribuion) > 0 and all(test_emotion_distribution) > 0

        # calculate percentiles
        train_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)
        test_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)
        if val_indexes is not None:
            val_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)

        for i in range(0, self.num_classes):
            train_emotion_percentiles[i] = (train_emotion_distribuion[i] * 100) / sum(train_emotion_distribuion)
            test_emotion_percentiles[i] = (test_emotion_distribution[i] * 100) / sum(test_emotion_distribution)
            if val_indexes is not None:
                val_emotion_percentiles[i] = (val_emotion_distribution[i] * 100) / sum(val_emotion_distribution)
                assert all(sum(val_emotion_percentiles) == 100)

        assert np.round(sum(train_emotion_percentiles)) == 100 and np.round(sum(test_emotion_percentiles)) == 100

        n_rows = 1
        n_cols = 2
        if val_indexes is not None:
            n_cols = 3

        emotion_axes = np.arange(0, self.num_classes, 1)
        fig, axes = plt.subplots(n_rows, n_cols)  # with (1, 2) -> 1 row 2 columns -> 2 plots in a row
        fig.suptitle("Emotion distribution over splits")
        for col in range(n_cols):
            ax = axes[col]
            if col == 0:
                # ax.plot(emotion_axes, train_emotion_percentiles, SubsetsColors.get('train_set'), label="Emotion distribution over training set")
                ax.bar(emotion_axes, train_emotion_percentiles, width=0.8, bottom=0, align='center', color=SubsetsColors.get('train_set'), tick_label=emotion_axes)
                #for e in emotion_axes:
                ax.set_xlabel('emotions')
                ax.set_ylabel('Samples (%)')
                ax.set_title(f'Train')
            elif col == 1:
                # ax.plot(emotion_axes, test_emotion_percentiles, SubsetsColors.get('test_set'), label="Emotion distribution over test set")
                ax.bar(emotion_axes, test_emotion_percentiles, width=0.8, bottom=0, align='center', color=SubsetsColors.get('test_set'), tick_label=emotion_axes)
                ax.set_xlabel('emotions')
                ax.set_ylabel('Samples (%)')
                ax.set_title(f'Test')
            else:
                ax.plot(emotion_axes, test_emotion_percentiles, SubsetsColors.get('val_set'), label="Emotion distribution over validation set")
                ax.set_title(f'Test')
                ax.set_xlabel('epochs')
                ax.set_ylabel('Loss value')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)

                ax.set_xlabel('epochs')
                ax.set_ylabel('Accuracies (%)')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)

        plt.tight_layout()
        plt.show()


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
        self.len = len(indices)

    def __len__(self):
        return self.len

# %%


# %% emoMusicPTDataloader class:


class emoMusicPTDataLoader(DataLoader):
    """
    Extends torch.utils.data.DataLoader. Combines a dataset and a sampler and provides an iterable over the given dataset
    """
    def __init__(self, dataset: emoMusicPTSubset, batch_size=4, shuffle=False, collate_fn=None, num_workers=1, pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
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
        self.len = len(dataset)

    def __len__(self):
        return self.len

    def song_idx_to_slices_range(self, n):
        start = n * _N_SLICES_PER_SONG
        end = start + _N_SLICES_PER_SONG

        return start, end


# %%