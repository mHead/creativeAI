import os
from ..musicSide.DatasetMusic2emotion.emoMusicPT import emoMusicPTDataset


def test_getitem(dataset: emoMusicPTDataset, sample_idx):
    if not dataset.slice_mode:
        assert dataset.__len__() == 744
        assert dataset.__len__() > sample_idx >= 0
    else:
        assert dataset.__len__() == 744 * 61
        assert dataset.__len__() > sample_idx >= 0

    sample, song_id, filename, label, label_coord = dataset.__getitem__(sample_idx)

    print(
        f'sample : {sample_idx} in [0-{dataset.__len__() - 1}] is in filename: {filename} song_id: {song_id} with label: {label} and label_coordinates in DataFrame: {label_coord}\n')
    print(f'data:\n\t{sample}\n'
          f'type sample: {type(sample)}\t'
          f'type label: {type(label)}')

    return True

