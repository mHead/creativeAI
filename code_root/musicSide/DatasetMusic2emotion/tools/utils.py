# TODO: need for safer code
import os
import numpy as np
import pandas as pd
import datetime


# %% 1. WAV Tools

def read_wavs(wav_dir, plot_wav=False, preprocess=False, verbose=True, verbose_deep=False):
    try:
        if not module_exists('scipy'):
            install_module('scipy')
            from scipy.io.wavfile import read
            from scipy import interpolate
        else:
            from scipy.io.wavfile import read
            from scipy import interpolate
        if plot_wav:
            import matplotlib.pyplot as plt

    except ImportError:
        print(f'{ImportError.__traceback__}')

    wav_filenames = []
    sample_rate = 0
    raw_audio_lengths = []
    raw_audio_vector = []

    for filename in os.listdir(wav_dir):
        sample_rate, raw_song = read(os.path.join(wav_dir, filename))
        if verbose_deep:
            print(f'samplerate: {sample_rate} Hz, len:{len(raw_song)}\n{raw_song}')

        if sample_rate != 44100:
            raw_song = convert_sample_rate(raw_song, old_sample_rate=sample_rate, new_sample_rate=44100)

        wav_filenames.append(filename)
        raw_audio_lengths.append(len(raw_song))
        raw_audio_vector.append(raw_song)
        if verbose_deep:
            print(f'samplerate: {sample_rate} Hz {raw_song}')

    if verbose:
        print(f'All {len(raw_audio_vector)} songs has been read, need preprocessing step.')

    del raw_song

    if preprocess:
        print(f'>>>Preprocessing Step:')
        del filename

        _min, _max = min_max(raw_audio_lengths)
        if verbose:
            print(f'min_max on  pure raw_audio_lengths returned {_min}, {_max}')

        _start = 15000 - 250
        _end = 45000 + 250

        assert sample_rate == 44100

        if verbose:
            print(f'Clip will be done starting from {_start} ms, to {_end} ms')
            print(f'from sample: {ms2samples(_start, sample_rate)}, to sample {ms2samples(_end, sample_rate)}')

        # we have to be sure that our clip could be performed (the clip is inside the music_length)
        if verbose:
            print(
                f'Padding needed due to minimum length: {_min} < last sample in clip {int(ms2samples(_end, sample_rate))}')

        padded_raw_audio_vector, padded_raw_audio_lengths = add_padding(audio_files=raw_audio_vector,
                                                                        padding_length=int(
                                                                            ms2samples(_end, sample_rate) - _min),
                                                                        boundary=int(ms2samples(_end, sample_rate)))
        del raw_audio_vector, raw_audio_lengths

        # passing _start, _end in ms
        clipped_raw_audio_files, clipped_length = clip_audio_files(padded_raw_audio_vector, start_ms=_start,
                                                                   end_ms=_end,
                                                                   sample_rate=sample_rate)
        del padded_raw_audio_vector, padded_raw_audio_lengths
        del _min, _max

        input500ms = 500
        if verbose:
            print(f'Defining an input value in ms {input500ms}')

        window_size, n_slices_per_song = calculate_window(input500ms, sample_rate, clipped_length)

        trimmed_raw_audio_files = trim_audio_files(clipped_raw_audio_files, window_size=window_size,
                                                   n_slices=n_slices_per_song)

        if verbose:
            print(
                f'The window_size is 22050:{window_size}, have been created n_slices per song 61:{n_slices_per_song}')
            print(f'trimmed_raw_audio_file.shape ({trimmed_raw_audio_files.shape})')

    if plot_wav:
        print(f'>>>Plot Step:\n')
        duration = len(raw_song) / sample_rate
        if verbose:
            print(f"Music duration: {duration}, samplerate: {sample_rate}, len(data): {len(raw_song)}")
            print(f"type duration: {type(duration)}, type samplerate: {type(sample_rate)}, type data: {type(raw_song)}")

        # time vector
        time = np.arange(0, duration, 1 / sample_rate)

        plt.plot(time, raw_song)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(filename)
        plt.show()

    return trimmed_raw_audio_files, clipped_length, sample_rate, n_slices_per_song, window_size


def clip_audio_files(padded_raw_audio_vector, start_ms, end_ms, sample_rate, verbose=True):
    clipped_raw_audio_files = []
    clipped_raw_audio_lengths = []
    start_sample = int(ms2samples(start_ms, sample_rate))
    end_sample = int(ms2samples(end_ms, sample_rate))

    if verbose:
        print(f'\t>> clip_audio_files will clip with these indices')
        print(f'\t start {start_sample} : end {end_sample}')
    assert (end_sample - start_sample) % 22050 == 0
    # for each song, clip and save
    for song in padded_raw_audio_vector:
        song = song[start_sample:end_sample]
        clipped_raw_audio_files.append(song)
        clipped_raw_audio_lengths.append(len(song))

    # print(f'{clipped_raw_audio_lengths}')

    # check lengths
    true = check_lengths(clipped_raw_audio_lengths)
    assert true

    return clipped_raw_audio_files, clipped_raw_audio_lengths[0]


'''
param raw_audio_files_vector contains #songs elements, each is the vector of samples values of the song
we want to trim each song independently (maintaining the relation between each song and the trimmed chunks)

return [
    [[first song slices], [22050] .., [#slices_per_song]], --> returns a np.array<Raw_Song>
    [[22050],..],
    ..
    [[last song slices]]
] 
'''


def trim_audio_files(clipped_raw_audio_files, window_size, n_slices):
    # get len of the first song (len is equal for all)
    audio_length = len(clipped_raw_audio_files[0])

    songs_trimmed = []  # dataset level
    for song in clipped_raw_audio_files:
        start_sample = 0
        end_sample = start_sample + window_size
        slices = []  # single song level

        for i in range(n_slices):
            assert (end_sample - start_sample) == window_size
            _slice = song[start_sample:end_sample]
            start_sample = end_sample
            end_sample = start_sample + window_size
            assert len(_slice) == window_size
            slices.append(_slice)

        slices = np.asarray(slices)
        songs_trimmed.append(slices)
    print(f'type slices (of one song) {type(slices)}, len {len(slices)}')
    songs_trimmed = np.asarray(songs_trimmed)
    print(
        f'All songs have been trimmed. They are contained inside trimmed_audio_files with shape: {songs_trimmed.shape}\ntype: {type(songs_trimmed)}\nlen:{type(songs_trimmed)}')

    return songs_trimmed


'''
calculates in terms of n_samples the window for the trim phase
a window of 22050 samples corresponds to a window of 500ms

returns padding_length, n_slices, input_in_ms2samples
'''


def calculate_window(input_in_ms, sample_rate, single_audio_length):
    input500ms_samples = ms2samples(input_in_ms, sample_rate)
    padding = False
    padding_length = 0

    # nSlices
    slices_single_song = single_audio_length // input500ms_samples
    print(f'{single_audio_length} // {input500ms_samples} = {slices_single_song}')
    # audio length in term  of samples
    song_samples_length = slices_single_song * input500ms_samples
    print(f'{song_samples_length} = {slices_single_song} * {input500ms_samples}')

    # checks
    if song_samples_length < single_audio_length:
        samples_last_slice = single_audio_length - song_samples_length
        assert samples_last_slice < input500ms_samples

        padding_length = input500ms_samples - samples_last_slice

        padding = True

    if padding:
        slices_single_song += 1

    return int(input500ms_samples), int(slices_single_song)


def add_padding(audio_files, padding_length, boundary):
    print(f'\t>>add_padding of length: {padding_length} samples')
    padded_songs = []
    padded_songs_lengths = []
    for song in audio_files:
        # if the song needs it, pad it
        if len(song) < boundary:
            song = np.pad(song, (0, boundary - len(song)), 'constant')
            assert len(song) == boundary
        padded_songs.append(song)
        padded_songs_lengths.append(len(song))

    return padded_songs, padded_songs_lengths


# this function reads the csv and prepare labels to be fed into the Network
def read_labels(labels_csv_path):
    dataframe = pd.read_csv(labels_csv_path)
    return dataframe


# this function reads the DataFrame and create the appropriate data structure for labels
def extract_labels(labels_df):
    labels = []
    song_ids = []
    for index, row in labels_df.iterrows():
        labels.append(row[1:])
        song_ids.append(row['song_id'])
    labels = np.asarray(labels)
    song_ids = np.asarray(song_ids)
    print(f'{labels.shape} : {labels}')
    return labels, song_ids


# %% 2. Utilities

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


def install_module(module):
    try:
        os.system(f'pip3 install {module}')
    except OSError:
        print(f'Ops: Errno {OSError.errno}\nTraceback: \n{OSError.__traceback__}')


def mp3_to_wav(source_mp3_dir, dst_wav_dir):
    if not os.path.exists(dst_wav_dir):
        os.mkdir(dst_wav_dir)

        try:
            if not module_exists('pydub'):
                install_module('pydub')

            from pydub import AudioSegment
        except ImportError:
            print(f'{ImportError.__traceback__}')

    for file in os.listdir(source_mp3_dir):
        sound = AudioSegment.from_mp3(os.path.join(source_mp3_dir, file))
        song_id = file.split('.')[0]
        song_id = song_id + '.wav'
        sound.export(os.path.join(dst_wav_dir, song_id), format="wav")

    assert len(source_mp3_dir) == len(dst_wav_dir)


def min_max(raw_audio_lengths) -> (int, int):
    _max = 0
    _min = np.Inf

    for i in range(0, len(raw_audio_lengths)):
        if raw_audio_lengths[i] < _min:
            _min = raw_audio_lengths[i]
        if raw_audio_lengths[i] > _max:
            _max = raw_audio_lengths[i]

    return _min, _max


def ms2samples(milliseconds, sample_rate):
    return (milliseconds / 1000) * sample_rate


def samples2ms(samples, sample_rate):
    return (samples / sample_rate) / 1000

def convert_sample_rate(raw_song, old_sample_rate, new_sample_rate):
    assert old_sample_rate != new_sample_rate

    try:
        if not module_exists('interpolate'):
            install_module('scipy')
            from scipy.io.wavfile import read
            from scipy import interpolate
        else:
            from scipy.io.wavfile import read
            from scipy import interpolate
    except ImportError:
        print(f'{ImportError.__traceback__}')

    duration = raw_song.shape[0] / old_sample_rate

    time_old = np.linspace(0, duration, raw_song.shape[0])
    time_new = np.linspace(0, duration, int(raw_song.shape[0] * new_sample_rate / old_sample_rate))

    interpolator = interpolate.interp1d(time_old, raw_song.T)
    new_audio = interpolator(time_new).T
    return new_audio

def remove_hidden_files(path_to_wav_files):
    filenames = []
    for file in os.listdir(path_to_wav_files):
        if file.startswith('.'):
            os.remove(os.path.join(path_to_wav_files, file))
        else:
            filename = file.split('.')[0]

            # if int(filename, base=10) in arousal_df['song_id'].values:
            #   filenames.append(file)

    print(f"clips_45 seconds len: {len(filenames)}\nContent:\n{filenames}")


def getCSV_info(path_to_csv):
    dataframe = pd.read_csv(path_to_csv)
    print(f'\t>>>> tools.utils module: getCSV_info({path_to_csv})\n')
    print(f'\t\tDataFrame.shape: {dataframe.shape}')
    print(f'\t\tDataFrame.head()\n{dataframe.head()}')


def check_lengths(array):
    first = array[0]
    for e in array:
        if e != first:
            return False
            break
    return True


def format_timestamp(current_timestamp: datetime):
    formatted = current_timestamp.date().__str__() + "_" + current_timestamp.time().__str__()
    formatted = formatted[:19]
    return formatted
