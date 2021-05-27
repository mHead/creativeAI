import os

import numpy as np


# %% 1. WAV Tools

def read_wavs(wav_dir, verbose=False, plot_wav=False, preprocess=False):
    try:
        if not module_exists('scipy.io.wavfile'):
            install_module('scipy.io.wavfile')

        from scipy.io.wavfile import read
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
        if verbose:
            print(f'samplerate: {sample_rate} Hz, len:{len(raw_song)}\n{raw_song}')

        wav_filenames.append(filename)
        raw_audio_lengths.append(len(raw_song))
        raw_audio_vector.append(raw_song)
        if verbose:
            print(f'samplerate: {sample_rate} Hz {raw_song}')

    if verbose:
        print(f'All {len(raw_audio_vector)} songs has been read, need preprocessing step.')

    if preprocess:
        print(f'>>>Preprocessing Step:\n')
        _min, _max = min_max(raw_audio_lengths)
        _start = 14750
        _end = int(_min) + 250

        clipped_raw_audio_files, clipped_lengths = clip_audio_files(raw_audio_vector, start_ms=_start, end_ms=_min, sample_rate=sample_rate)

        _min, _max = min_max(clipped_lengths)
        assert _min == _max


        input500ms = 0.5
        window_size, n_slices_per_song, padding_length = calculate_window(input500ms, sample_rate, _min )
        trimmed_raw_audio_files = trim_audio_files(clipped_raw_audio_files, window_size=window_size, n_slices = n_slices_per_song, padding_length=padding_length)

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

    return raw_audio_lengths, raw_audio_vector, sample_rate


def clip_audio_files(raw_audio_vector, start_ms, end_ms, sample_rate) -> ():
    clipped_raw_audio_files = []
    clipped_raw_audio_lengths = []

    start_sample = ms2samples(start_ms, sample_rate=sample_rate)
    end_sample = ms2samples(end_ms, sample_rate=sample_rate)

    # for each song, clip and save
    for song in raw_audio_vector:
        song = song[start_sample:end_sample]
        clipped_raw_audio_files.append(song)
        clipped_raw_audio_lengths.append(len(song))

    # check lengths
    for i in range(0, len(clipped_raw_audio_files)):
        if i < len(clipped_raw_audio_files) - 1:
            assert (len(clipped_raw_audio_files[i]) == len(clipped_raw_audio_files[i + 1]))

    return np.asarray(clipped_raw_audio_files), np.asarray(clipped_raw_audio_lengths)


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
def trim_audio_files(clipped_raw_audio_files, window_size, n_slices, padding_length):

    # get len of the first song (len is equal for all)
    audio_length = len(clipped_raw_audio_files[0])

    if not padding_length is None:
        #TODO
        return None

    songs_trimmed = [] # dataset level
    for song in clipped_raw_audio_files:
        start_sample = 0
        end_sample = start_sample + window_size
        slices = [] # single song level
        for i in range(n_slices):
            slice = song[start_sample:end_sample]
            start_sample = end_sample + 1
            end_sample = start_sample + window_size
            slices.append(slice)
        slices = np.asarray(slices)
        songs_trimmed.append(slices)

    songs_trimmed = np.asarray(songs_trimmed)




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

    # audio length in term  of samples
    song_samples_length = slices_single_song * input500ms_samples

    # checks
    if song_samples_length < single_audio_length:
        samples_last_slice = single_audio_length - song_samples_length
        assert samples_last_slice < input500ms_samples

        padding_length = input500ms_samples - samples_last_slice

        padding = True

    if padding:
        slices_single_song += 1

    return input500ms_samples, slices_single_song, padding_length






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
    _min = np.max

    for i in range(0, len(raw_audio_lengths)):
        if raw_audio_lengths[i] < _min:
            _min = raw_audio_lengths[i]
        if raw_audio_lengths[i] > _max:
            _max = raw_audio_lengths[i]

    return min, max


def ms2samples(milliseconds, sample_rate):
    return (milliseconds / 1000) * sample_rate


def samples2ms(samples, sample_rate):
    return (samples / sample_rate) / 1000


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
