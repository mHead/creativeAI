import os
import sys
# utils
from musicSide.DatasetMusic2emotion.tools import va2emotion as va2emo
from musicSide.DatasetMusic2emotion.tools import utils as u
# Datasets
from musicSide.DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from musicSide.DatasetMusic2emotion.emoMusicPT import emoMusicPT
# Models
from musicSide.Model.CNN_biGRU import CNN_BiGRU
from musicSide.Model.TorchModel import TorchModel
from musicSide.Model.Benchmark import Benchmark
from sklearn.model_selection import StratifiedShuffleSplit
# %% Using ArgumentParser
#import argparse

#parser = argparse.ArgumentParser(description='main arguments')
#parser.add_argument('-v', '--verbose', nargs='?', default=False, help='Run the program in verbose mode')

#parser.add_argument('-tf', '--tensorflow', dest='Tensorflow', type=str, help='Run the program with Tensorflow backend and Keras frontend')
#parser.add_argument('-pt', '--pytorch', dest='PyTorch', type=str, help='Run the program with PyTorch')

#parser.add_argument('-r', '--repo_root', dest='repo_root',  type=str, help='It must be followed by the repo root path')

#args = parser.parse_args()
#print(args)
# %%
print(sys.argv)
argv = sys.argv[1:]
print(f'[main.py] argv: {argv} type: {type(argv)}')

repo_root = r''
verbose = False
generate_csv = False
pick_repo = False
run_config = ''
save_csv_path = r''
#main config

keras_ = False
pytorch_ = True

for arg in argv:
    if pick_repo:
        repo_root = arg
    if generate_csv:
        save_csv_path = arg

    if arg.__eq__("--verbose"):
        verbose = True
    if arg.__eq__("--repo_root"):
        pick_repo = True
    if arg.__eq__("--generate_csv"):
        generate_csv = True
    if arg.__eq__("--legion"):
        run_config = 'legion'

print(repo_root)
assert len(repo_root) != 0
music_data_root = os.path.join(repo_root, r'musicSide_root_data')
image_data_root = os.path.join(repo_root, r'imageSide_root_data')
code_root = os.path.join(repo_root, r'code_root')
save_dir_root = os.path.join(repo_root, r'saves_dir')

music_dataset_path = os.path.join(music_data_root, 'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG')

if not os.path.exists(save_dir_root):
    os.mkdir(save_dir_root)

music_labels_csv_root = os.path.join(music_data_root, '[labels]emotion_average_dataset_csv')
save_music_emo_csv_path = os.path.join(music_labels_csv_root, 'music_emotions_labels.csv')

if __name__ == '__main__':

    if verbose:
        print(f'\n\n\nStarting main with: {music_data_root} as the root for music data\n'
              f'{image_data_root} as the root for image data\n'
              f'{code_root} as the root for code\n\n\n')

    if verbose:
        print(f'Checking existence of {music_labels_csv_root}'
              f'If exists check if empty, otherwise create the folder and generate the csv')

    # %% Create music-emotions labels csv from arousal and valence csv
    if generate_csv:
        if not os.path.exists(music_labels_csv_root):
            os.mkdir(music_labels_csv_root)
            if verbose:
                print(f'The path did not existed, starting csv generation and save at {save_music_emo_csv_path}')

            va2emo.generateMusicEmo_csv(save_music_emo_csv_path, music_data_root)
        else:
            if not os.listdir(music_labels_csv_root):  # directory is empty
                if verbose:
                    print(
                        f'The path exists but is empty, starting csv generation and save at {save_music_emo_csv_path}')

                va2emo.generateMusicEmo_csv(save_music_emo_csv_path, music_data_root)
            else:
                # directory is not empty, file exist
                print(f'\n\n>>The file {save_music_emo_csv_path} already exists\n\n')
                u.getCSV_info(save_music_emo_csv_path)
    # %%

    # %% Keras Main
    if keras_:
        b = Benchmark("keras_dataset_timer")
        b.start_timer()
        music2emotion_Dataset = DatasetMusic2emotion(data_root=music_data_root, train_frac=0.9, run_config=run_config, preprocess=False)
        print(f'Hey I am: {music2emotion_Dataset}')
        b.end_timer()

        b = Benchmark("keras_model_timer")
        b.start_timer()
        music2emotion_Model = CNN_BiGRU(music2emotion_Dataset, save_dir=save_dir_root, do_train=True, do_test=False,
                                        load_model=False, load_model_path=(None, None))

        b.end_timer()
    # %%

    # %% PyTorch Main
    if pytorch_:
        b = Benchmark("pytorch_dataset_timer")
        b.start_timer()
        pytorch_dataset = emoMusicPT(dataset_root=music_dataset_path, slice_mode=False)
        print(f'\n***** main: emoMusicPT created *****\n\n')

        sample_idx = 61

        if not pytorch_dataset.slice_mode:
            assert pytorch_dataset.__len__() == 744
            assert pytorch_dataset.__len__() > sample_idx >= 0
        else:
            assert pytorch_dataset.__len__() == 744 * 61
            assert pytorch_dataset.__len__() > sample_idx >= 0

        sample, song_id, filename, label, label_coord = pytorch_dataset.__getitem__(sample_idx)

        print(f'sample : {sample_idx} in [0-{pytorch_dataset.__len__() - 1}] is in filename: {filename} song_id: {song_id} with label: {label} and label_coordinates in DataFrame: {label_coord}\n')
        print(f'data:\n\t{sample}\n'
              f'type sample: {type(sample)}\t'
              f'type label: {type(label)}')

        test_frac = 0.1
        train_indexes, test_indexes = pytorch_dataset.stratified_song_level_split(test_fraction=test_frac)

        #train_dl = DataLoader(train_indexes, batch_size=32, shuffle=False)
        #test_dl = DataLoader(test_indexes, batch_size=32, shuffle=False)



        b.end_timer()

        #b = Benchmark("pytorch_model_timer")
        #first_model = TorchModel(pytorch_dataset, save_dir_root=save_dir_root)
        #first_model.train()
        #b.end_timer()
        #first_model.print_statistics()
        #first_model.test()
    # %%
    sys.exit(0)
