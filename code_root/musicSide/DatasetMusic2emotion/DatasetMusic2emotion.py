import os
import tools as u


class DatasetMusic2emotion:

    def __init__(self, dataset_root):
        self.emotions_csv_path = os.path.join(dataset_root, 'emotions_cont_average.csv')  # TODO
        self.emotions_labels = u.read_labels(self.emotions_csv_path)
        self.raw_audio_lengths, self.raw_audio, self.sample_rate = u.read_wavs(
            os.path.join(dataset_root, 'clips_45seconds_wav'))



