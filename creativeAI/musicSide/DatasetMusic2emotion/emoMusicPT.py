import sys
import datetime
import torch
import torchaudio
import torchvision
from torchvision import transforms as T
from PIL import Image
import librosa
import librosa.display
import numpy as np
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from ..DatasetMusic2emotion.tools import utils as u
from ..DatasetMusic2emotion.tools import va2emotion as va2emo
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

_MULTI_LABELS_PER_SONG_CSV = r'music_emotions_labels.csv'
_SINGLE_LABEL_PER_SONG_CSV = r'music_single_emotion_labels.csv'

verbose = False
# dataset constant info
_SAMPLE_RATE = 44100
_N_SLICES_PER_SONG = 61
_SLICE_SIZE = 22050
_SAMPLE_TOTAL_DURATION = _SLICE_SIZE * _N_SLICES_PER_SONG
_N_CLASSES = len(va2emo.EMOTIONS_)

SubsetsColors = {
    'train_set': '#30A666',  # verde scuro
    'test_set': '#A8251E',  # verde chiaro
    'val_set': '#B1E2F0',  # rosso scuro
}

EmotionColors = va2emo.EMOTION_COLORS


class emoMusicPTDataset(Dataset):
    """
        This is a map-style dataset because it implements the __getitem__ and __len__ protocols
        it can be accessed as dataset[n] which read the idx-th sample and its corresponding label.

        slice_mode:
            if True the __getitem__(self, n)
                returns the slice of the song thinking n in (0, 744x61) with its label (the slice's one)
                as torch.Tensor([1, 22050])
            else
                returns the whole song as a torch.Tensor([1, 1345050])

    """

    def __init__(self, slice_mode=False, env=None, mfcc=False, melspec=False):
        super().__init__()

        if env is not None:
            self._RUN_CONF = env.get('run_config')
            self._REPO_ROOT = env.get('repo_root')
            self._CODE_ROOT = env.get('code_root')
            self._AUDIO_PATH = env.get('dataset_root')
            self._SONGID_DOMINANT_EMO_CSV = os.path.join(env.get('labels_root'), _SINGLE_LABEL_PER_SONG_CSV)
            self._SONGID_EMOTIONS_CSV = os.path.join(env.get('labels_root'), _MULTI_LABELS_PER_SONG_CSV)
            self._MUSIC_DATA_ROOT = env.get('music_data_root')
            self._SAVE_DIR_ROOT = env.get('save_dir_root')
            if melspec:
                self._n_fft = env.get("n_fft")
                self._hop_length = env.get("hop_length")
                self._n_mel = env.get("n_mel")
        else:
            print(f'Configuration Environment failed!')
            sys.exit(-1)

        self.is_mfcc_mode = False
        self.is_mel_spec_mode = False
        self.is_raw_audio_mode = False

        if melspec and not mfcc:
            self.name = 'emoMusicPTDataset - MelSpectrogram mode'
            self.is_mel_spec_mode = True
            self.mel_transformation = torchaudio.transforms.MelSpectrogram(
                sample_rate=_SAMPLE_RATE,
                n_fft=env.get('n_fft'),
                hop_length=env.get('hop_length'),
                n_mels=env.get('n_mel'))

        elif mfcc and not melspec:
            self.name = 'emoMusicPTDataset - MFCC mode'
            self.is_mfcc_mode = True
        elif not mfcc and not melspec:
            self.name = 'emoMusicPTDataset - raw audio'
            self.is_raw_audio_mode = True

        self.slice_mode = slice_mode
        self.num_classes = _N_CLASSES

        # CSV path to pd.DataFrame
        self.song_id_emotions_labels_frame = pd.read_csv(self._SONGID_EMOTIONS_CSV)
        self.single_label_per_song_frame = pd.read_csv(self._SONGID_DOMINANT_EMO_CSV)

        wav_filenames = [f for f in os.listdir(self._AUDIO_PATH) if not f.startswith('.')]
        wav_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.wav_filenames_sorted = wav_filenames
        # for each file expand 61 names
        if slice_mode:
            self.wav_filenames = self.expand_filenames(wav_filenames)
        else:
            self.wav_filenames = self.wav_filenames_sorted

        del wav_filenames
        # self.labels_song_level = self.single_label_per_song_frame.loc[:, 'emotion_label']
        # self.labels_slice_level = self.song_id_emotions_labels_frame.loc[1:, self.song_id_emotions_labels_frame.columns != 'song_id']
        self.labels_slice_level, self.song_ids, self.labels_song_level = u.extract_labels(self._SONGID_EMOTIONS_CSV)

        self.print_info()

    def get_save_dir(self):
        return self._SAVE_DIR_ROOT

    def __len__(self):
        return len(self.wav_filenames)  # if slice_mode: 744 x 61 else: 744

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

            # obtain song_id, slice_label and slice location
            song_slice = self.wav_filenames[n]
            slice_label = self.labels_slice_level[n]
            song_id = song_slice.split('_')[0]
            slice_no = int(song_slice.split('_')[1].split('.')[0])

            start_offset = slice_no * _SLICE_SIZE
            end_offset = start_offset + _SLICE_SIZE

            # now read audio associated
            filename = str(song_id) + '.wav'
            if filename in self.wav_filenames_sorted:
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
            if self.is_mfcc_mode:
                # TODO transform and
                return _wav_slice, song_id, filename, slice_label, slice_no
            elif self.is_mel_spec_mode:
                # TODO transform and
                return _wav_slice, song_id, filename, slice_label, slice_no
            else:
                return _wav_slice, song_id, filename, slice_label, slice_no

        else:
            assert self.__len__() > n >= 0
            row_id = n
            # song_id = self.single_label_per_song_frame.iloc[row_id, 0]
            # emotion_label = self.single_label_per_song_frame.iloc[row_id, 1]
            waveform = None
            song_id = self.wav_filenames[n].split('.')[0]
            filename = self.wav_filenames[n]
            emotion_label = self.labels_song_level[n]

            if verbose:
                print(f'\n\ttype song_id {type(song_id)} : {song_id}'
                      f'\n\ttype emotion_label : {type(emotion_label)} : {emotion_label}')

            # now read audio associated
            # filename = str(song_id) + '.wav'
            if len(filename) > 0:
                filename_path = os.path.join(self._AUDIO_PATH, filename)
            else:
                print(f'{filename} does not exist in path')
                sys.exit(-1)

            if os.path.exists(filename_path):

                # 1. raw audio mode
                if self.is_raw_audio_mode:
                    metadata = torchaudio.info(filename_path)
                    if verbose:
                        self.print_metadata(_metadata=metadata, src=filename_path)
                    waveform, sample_rate = torchaudio.load(filename_path)
                    assert sample_rate == _SAMPLE_RATE
                    if verbose: print(
                        f'torchaudio.load SAMPLE EXTRACTION: type of waveform: {type(waveform)} {waveform.shape}')
                    return waveform, song_id, filename, emotion_label, self.slice_mode

                # 2. mfcc_mode
                elif self.is_mfcc_mode:
                    waveform, sample_rate = u.librosa_load_wrap(filename_path, sr=_SAMPLE_RATE)
                    assert sample_rate == _SAMPLE_RATE
                    if verbose:
                        print(f'librosa.load SAMPLE EXTRACTION: type of waveform: {type(waveform)} {waveform.shape}')
                    mfcc_features_dict = u.extract_mfcc_features(waveform, n_fft=_SLICE_SIZE, hop_length=220)
                    #  and return features
                    return mfcc_features_dict, song_id, filename, emotion_label, self.slice_mode

                # 3. mel_spec_mode
                elif self.is_mel_spec_mode:
                    with_librosa = False
                    if not with_librosa:
                        waveform, sample_rate = torchaudio.load(filename_path)
                        assert sample_rate == _SAMPLE_RATE
                        # TODO convert to RGB
                        # mel_spec_norm = self.compose_trans(waveform)
                        # image = T.ToPILImage(mel_spec_norm)
                        mel_spec = self.mel_transformation(waveform)
                        # mel_spec_ = torchaudio.transforms.AmplitudeToDB()(mel_spec_)
                        # self.plot_spectrogram(mel_spec_)
                        # mel_spec_ = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(mel_spec_)
                        return mel_spec, song_id, filename, emotion_label, self.slice_mode

                    else:
                        waveform, sample_rate = u.librosa_load_wrap(filename_path, sr=_SAMPLE_RATE)

                        mel_spec = u.extract_mel_spectrogram_librosa(waveform, sr=sample_rate, n_mel=self._n_mel,
                                                                     n_fft=self._n_fft, hop_length=self._hop_length)
                        # TODO convert to RGB
                        #fig = plt.figure(figsize=(25, 10))
                        #librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time')
                        #plt.title('Mel Spectrogram')
                        #plt.colorbar(format='%+2.0f dB')
                        #plt.show()
                        return mel_spec, song_id, filename, emotion_label, self.slice_mode

                else:
                    print(f'Wrong Dataset Mode among raw_audio, mfcc, mel_spec')
                    sys.exit(-1)

            else:
                print(f'The file at {filename_path} does not exist')
                sys.exit(-1)

    def plot_spectrogram(self, mel_spec):
        fig = plt.figure(figsize=(25, 10))
        # plt.imshow(mel_spec.permute(1, 2, 0))
        # plt.savefig('/Users/head/Desktop/ciao.jpg')
        # plt.show()

    def stratified_song_level_split(self, test_fraction, criterion_mode):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=0)
        if not self.slice_mode:
            splits = splitter.split(self.wav_filenames, self.labels_song_level[:len(self.wav_filenames)])
            labels = self.labels_song_level[:len(self.wav_filenames)]
        else:
            splits = splitter.split(self.wav_filenames, self.labels_slice_level[:len(self.wav_filenames)])
            labels = self.labels_slice_level[:len(self.wav_filenames)]
        train_index = []
        test_index = []
        for train_index, test_index in splits:
            print(
                f'[emoMusicPTDataset.py] TRAIN INDEX: {train_index} type: {type(train_index)} shape: {train_index.shape}')
            print(f'[emoMusicPTDataset.py] TEST INDEX: {test_index} type: {type(test_index)} shape: {test_index.shape}')
        # train_index.sort()
        # test_index.sort()
        print(f'\n[emoMusicPTDataset.py] type train_indexes {type(train_index)}\ntype test_indexes {type(test_index)}')
        train_distribution, test_distribution, _ = self.calculate_classes_distribution(labels, train_index, test_index, None)
        class_distribution = np.zeros(self.num_classes)
        for i, v in enumerate(test_distribution):
            class_distribution[i] = train_distribution[i] + v
        assert sum(class_distribution) == len(self)

        if criterion_mode == 'freq':
            weights_for_criterion = np.ones(self.num_classes)
            weights_for_criterion /= class_distribution
        elif criterion_mode == 'max':
            weights_for_criterion = np.ones(self.num_classes)
            max = np.max(class_distribution)
            weights_for_criterion *= max
            for i, v in enumerate(weights_for_criterion):
                weights_for_criterion[i] = v / class_distribution[i]

        else:
            print("No criterion_weights_mode specified")
            return train_index, test_index, None

        return train_index, test_index, weights_for_criterion

    def expand_filenames(self, wav_filenames):
        abstract_filenames = []
        for f in wav_filenames:
            for i in range(0, _N_SLICES_PER_SONG):
                abstract_filenames.append(f'{f.split(".")[0]}_{i}.wav')
        return abstract_filenames

    def stratifiedKFold_song_level(self, n_splits, shuffle, random_state=None):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def calculate_classes_distribution(self, labels, train_indexes, test_indexes, val_indexes = None):
        train_emotion_distribuion = np.zeros(self.num_classes)
        test_emotion_distribution = np.zeros(self.num_classes)

        for train_index in train_indexes:
            train_emotion_distribuion[labels[train_index]] += 1

        for test_index in test_indexes:
            test_emotion_distribution[labels[test_index]] += 1

        if val_indexes is not None:
            val_emotion_distribution = np.zeros(self.num_classes)
            for val_index in val_indexes:
                val_emotion_distribution[labels[test_index]] += 1

            assert all(val_emotion_distribution) > 0
        assert all(train_emotion_distribuion) > 0 and all(test_emotion_distribution) > 0
        if val_indexes is not None:
            return train_emotion_distribuion, test_emotion_distribution, val_emotion_distribution
        else:
            return train_emotion_distribuion, test_emotion_distribution, None

    def calculate_classes_percentages(self, labels, train_indexes, test_indexes, val_indexes=None):
        train_emotion_distribution, test_emotion_distribution, val_emotion_distribution = \
            self.calculate_classes_distribution(labels, train_indexes, test_indexes, val_indexes)

        # calculate percentiles
        train_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)
        test_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)
        if val_indexes is not None:
            val_emotion_percentiles = np.zeros(self.num_classes, dtype=np.float32)

        for i in range(0, self.num_classes):
            train_emotion_percentiles[i] = (train_emotion_distribution[i] * 100) / sum(train_emotion_distribution)
            test_emotion_percentiles[i] = (test_emotion_distribution[i] * 100) / sum(test_emotion_distribution)
            if val_indexes is not None:
                val_emotion_percentiles[i] = (val_emotion_distribution[i] * 100) / sum(val_emotion_distribution)
                assert all(sum(val_emotion_percentiles) == 100)

        assert np.round(sum(train_emotion_percentiles)) == 100 and np.round(sum(test_emotion_percentiles)) == 100
        return train_emotion_percentiles, test_emotion_percentiles

    def plot_indices_distribution(self, labels, train_indexes, test_indexes, val_indexes=None):
        """
        Collect distrubution per subset
        """

        train_emotion_percentiles, test_emotion_percentiles = self.calculate_classes_percentages(labels, train_indexes, test_indexes, None)
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
                ax.bar(emotion_axes, train_emotion_percentiles,
                       width=0.8, bottom=0,
                       align='center',
                       color=[e for e in EmotionColors.values()], edgecolor=SubsetsColors.get('train_set'),
                       tick_label=emotion_axes)

                ax.set_xlabel('emotions')
                ax.set_ylabel('Samples (%)')
                ax.set_title(f'Train')
            elif col == 1:
                ax.bar(emotion_axes, test_emotion_percentiles,
                       width=0.8, bottom=0,
                       align='center',
                       color=[e for e in EmotionColors.values()], edgecolor=SubsetsColors.get('test_set'),
                       tick_label=emotion_axes)

                ax.set_xlabel('emotions')
                ax.set_ylabel('Samples (%)')
                ax.set_title(f'Test')
            else:
                ax.plot(emotion_axes, test_emotion_percentiles, SubsetsColors.get('val_set'),
                        label="Emotion distribution over validation set")
                ax.set_title(f'Test')
                ax.set_xlabel('epochs')
                ax.set_ylabel('Loss value')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)

                ax.set_xlabel('epochs')
                ax.set_ylabel('Accuracies (%)')
                ax.legend(loc='upper left', scatterpoints=1, frameon=True)

        plt.tight_layout()

        path_to_save = os.path.join(self._SAVE_DIR_ROOT, self.name)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        path_to_save = os.path.join(path_to_save, 'plots')
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        d = datetime.datetime.now()
        path_to_save = os.path.join(path_to_save, f'{u.format_timestamp(d)}_emotion_distribution_over_subsets.png')
        plt.savefig(path_to_save)

        # plt.show()

    def print_info(self):
        print(f'*** [emoMusicPTDataset.py] '
              f'creating {self.name} ****\n'
              f'\n\t--------------------'
              f'\n\trun_conf: {self._RUN_CONF}'
              f'\n\trepo root: {self._REPO_ROOT}'
              f'\n\tmusic data root: {self._MUSIC_DATA_ROOT}'
              f'\n\tslice_mode: {self.slice_mode}'
              f'\n\t--------------------'
              f'\n\tmfcc mode: {self.is_mfcc_mode}'
              f'\n\tmelspec mode: {self.is_mel_spec_mode}'
              f'\n\traw audio mode: {self.is_raw_audio_mode}'
              f'\n\t--------------------'
              f'\n\tAudio samples len: {self.__len__()}'
              f'\n\tLabels song level len: {len(self.labels_song_level)}'
              f'\n\tLabels slice level len: {len(self.labels_slice_level)}')
        if verbose:
            print(f'Data_root:\n'
                  f'\tAudio: {self._AUDIO_PATH}\n'
                  f'\tLabels: {self._SONGID_DOMINANT_EMO_CSV}\n\t{self._SONGID_EMOTIONS_CSV}\n'
                  f'\nself.labels_song_level:\n\t{self.labels_song_level}\n\ttype: {type(self.labels_song_level)}'
                  f'\n\tlen: {len(self.labels_song_level)}\n'
                  f'\nself.labels_slice_level:\n\t{self.labels_slice_level}\n\ttype: {type(self.labels_slice_level)}'
                  f'\n\tlen: {len(self.labels_slice_level)}\n'
                  f'\nself.wav_filenames:\n\t{self.wav_filenames}\n\ttype: {type(self.wav_filenames)}'
                  f'\n\tlen: {len(self.wav_filenames)}\n\n')

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

    def __init__(self, dataset: emoMusicPTSubset, batch_size=1, shuffle=False, collate_fn=None, num_workers=1,
                 pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)
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
# %%
