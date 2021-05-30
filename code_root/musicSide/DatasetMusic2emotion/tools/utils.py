# TODO: need for safer code
import os
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import gc

__SAMPLE_AT = 44100
__START_CLIP = 15000
__END_CLIP = 45000
__OFFSET = 250

__INPUT_500ms = 500

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

try:
    if not module_exists('pydub'):
        install_module('pydub')

    from pydub import AudioSegment
except ImportError:
    print(f'{ImportError.__traceback__}')

try:
    if not module_exists('gc'):
        install_module('gc')
        import gc

    if not module_exists('pprint'):
        install_module('pprint')
        import pprint
except ImportError:
    print(f'{ImportError.__traceback__}')


# %% 1. WAV Tools

def read_wavs(wav_dir, plot_wav=False, preprocess=False, verbose=True, verbose_deep=False):
    memory_management = True

    wav_filenames = []
    raw_audio_lengths = []
    raw_audio_vector = []

    n_resampled = 0
    wrong_frequencies = []
    for filename in os.listdir(wav_dir):
        sample_rate, raw_song = read(os.path.join(wav_dir, filename))
        if verbose_deep:
            print(f'samplerate: {sample_rate} Hz, len:{len(raw_song)}\n{raw_song}')

        if sample_rate != __SAMPLE_AT:
            if verbose_deep:
                print(f'\nConverting sample_rate...old:{sample_rate}, len raw_song: {len(raw_song)}')
            wrong_frequencies.append(sample_rate)
            raw_song, sample_rate = convert_sample_rate(raw_song, old_sample_rate=sample_rate, new_sample_rate=44100)
            n_resampled += 1
            if verbose_deep:
                print(f'\nConverting sample_rate...new:{sample_rate}, len raw_song: {len(raw_song)}')

        wav_filenames.append(filename)
        raw_audio_lengths.append(len(raw_song))
        raw_audio_vector.append(raw_song)
        if verbose_deep:
            print(f'samplerate: {sample_rate} Hz {raw_song}')

    if verbose:
        print(f'All {len(raw_audio_vector)} songs has been read, need preprocessing step.')
        print(f'Between the {len(raw_audio_vector)}, {n_resampled} had these frequencies:\n{wrong_frequencies}')

    del raw_song, sample_rate, wrong_frequencies, n_resampled

    if preprocess:
        print(f'>>>Preprocessing Step:')
        del filename

        _min, _max = min_max(raw_audio_lengths)
        if verbose:
            print(f'min_max on  pure raw_audio_lengths returned {_min}, {_max}')

        _start = __START_CLIP - __OFFSET
        _end = __END_CLIP + __OFFSET

        if verbose:
            print(f'Clip will be done starting from {_start} ms, to {_end} ms')
            print(f'from sample: {ms2samples(_start, __SAMPLE_AT)}, to sample {ms2samples(_end, __SAMPLE_AT)}')

        # we have to be sure that our clip could be performed (the clip is inside the music_length)
        if verbose:
            print(
                f'Padding needed due to minimum length: {_min} < last sample in clip {int(ms2samples(_end, __SAMPLE_AT))}')

        padded_raw_audio_vector, padded_raw_audio_lengths = add_padding(audio_files=raw_audio_vector,
                                                                        padding_length=int(
                                                                            ms2samples(_end, __SAMPLE_AT) - _min),
                                                                        boundary=int(ms2samples(_end, __SAMPLE_AT)))

        collected_garbage = gc.collect()
        print(f'unreachable objects: {collected_garbage}')
        print(f'remaining garbage: {gc.garbage}')
        del raw_audio_vector, raw_audio_lengths

        if verbose:
            print(f'Padding.. DONE\n\nCLIPPING...')
        # passing _start, _end in ms
        clipped_raw_audio_files, clipped_length = clip_audio_files(padded_raw_audio_vector, start_ms=_start,
                                                                   end_ms=_end,
                                                                   sample_rate=__SAMPLE_AT)
        del padded_raw_audio_vector, padded_raw_audio_lengths
        del _min, _max
        if verbose:
            print(f'Clipping.. DONE\n\nCalculating window size for the net input...')

        if verbose:
            print(f'Defining an input value in ms {__INPUT_500ms}')

        window_size, n_slices_per_song = calculate_window(__INPUT_500ms, __SAMPLE_AT, clipped_length)

        if verbose:
            print(f'DONE!\n\n window_size: {window_size}, n_slices_per_song: {n_slices_per_song}\texpected(22050, '
                  f'61)\n\nTRIMMING...')

        trimmed_raw_audio_files = trim_audio_files(clipped_raw_audio_files, window_size=window_size,
                                                   n_slices=n_slices_per_song)
        del clipped_raw_audio_files
        if verbose:
            print(
                f'The window_size is 22050:{window_size}, have been created n_slices per song 61:{n_slices_per_song}')
            print(f'trimmed_raw_audio_file.shape ({trimmed_raw_audio_files.shape})')

    if plot_wav:
        print(f'>>>Plot Step:\n')
        duration = len(raw_song) / __SAMPLE_AT
        if verbose:
            print(f"Music duration: {duration}, samplerate: {__SAMPLE_AT}, len(data): {len(raw_song)}")
            print(f"type duration: {type(duration)}, type samplerate: {type(__SAMPLE_AT)}, type data: {type(raw_song)}")

        # time vector
        time = np.arange(0, duration, 1 / __SAMPLE_AT)

        plt.plot(time, raw_song)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(filename)
        plt.show()

    return trimmed_raw_audio_files, clipped_length, __SAMPLE_AT, n_slices_per_song, window_size


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
        assert len(slices) == n_slices
        songs_trimmed.append(slices)

    print(f'type slices (of one song) {type(slices)}, len {len(slices)}')

    # print(f'{len(songs_trimmed) * len(songs_trimmed[0])*songs_trimmed[0].shape[1]}\n{type(songs_trimmed)}')
    songs_trimmed = np.asarray(songs_trimmed)
    print(
        f'All songs have been trimmed. They are contained inside trimmed_audio_files with shape: {songs_trimmed}\ntype: {type(songs_trimmed)}\nlen:{type(songs_trimmed)}')

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
    # print(f'{single_audio_length} // {input500ms_samples} = {slices_single_song}')
    # audio length in term  of samples
    song_samples_length = slices_single_song * input500ms_samples
    # print(f'{song_samples_length} = {slices_single_song} * {input500ms_samples}')

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


def mp3_to_wav(source_mp3_dir, dst_wav_dir):
    if not os.path.exists(dst_wav_dir):
        os.mkdir(dst_wav_dir)

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
    memory_management = True
    verbose = False
    duration = raw_song.shape[0] / old_sample_rate

    time_old = np.linspace(0, duration, raw_song.shape[0])
    time_new = np.linspace(0, duration, int(raw_song.shape[0] * new_sample_rate / old_sample_rate))

    interpolator = interpolate.interp1d(time_old, raw_song.T)
    new_audio = interpolator(time_new).T

    new_duration = new_audio.shape[0] / new_sample_rate

    if verbose:
        print(f'new_audio duration:{new_duration} new_audio.shape[0]{new_audio.shape[0]}\nold_audio duration:{duration} old_audio.shape[0]{raw_song.shape[0]}')
    assert int(duration) == int(new_duration)

    if memory_management:
        collected_garbage = gc.collect()
        if verbose:
            print(f'unreachable objects: {collected_garbage}')
            print(f'remaining garbage: {gc.garbage}')
        # break the cycle
        if collected_garbage > 0:
            gc.garbage[0].set_next(None)
            del gc.garbage[:]

    return new_audio, new_sample_rate


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
