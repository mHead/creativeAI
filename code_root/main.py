import os
import sys
# utils
from musicSide.DatasetMusic2emotion.tools import va2emotion as va2emo
from musicSide.DatasetMusic2emotion.tools import utils as u
# Datasets
from musicSide.DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from musicSide.DatasetMusic2emotion.emoMusicPT import emoMusicPTDataset, emoMusicPTDataLoader, emoMusicPTSubset
# Models
from musicSide.Model.CNN_biGRU import CNN_BiGRU
from musicSide.Model.TorchModel import TorchModel
from musicSide.Model.Runner import Runner
from musicSide.Model.Benchmark import Benchmark
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.cuda as cuda
from torch.utils.data import Subset
verbose = False
if verbose:
    print(f'Using torch version: {torch.__version__}')

    if cuda.is_available():
        print(f'\t- GPUs available: {cuda.device_count()}')
        print(f'\t- Current device index: {cuda.current_device()}')
    else:
        print(f'\t- GPUs available: {cuda.device_count()}')
        print(f'\t- Cuda is NOT available\n')
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
# print(sys.argv)
argv = sys.argv[1:]
# print(f'[main.py] argv: {argv} type: {type(argv)}')

repo_root = r''
verbose = True
generate_csv = False
pick_repo = False
run_config = ''
save_csv_path = r''
# main framework switch config
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
    if arg.__eq__("--colab"):
        run_config = 'colab'

# print(repo_root)
assert len(repo_root) != 0

music_data_root = os.path.join(repo_root, r'musicSide_root_data')
image_data_root = os.path.join(repo_root, r'imageSide_root_data')
code_root = os.path.join(repo_root, r'code_root')
save_dir_root = os.path.join(repo_root, r'saves_dir')

modelVersions = {
    0: 'Baseline',
    1: 'v1',
    2: 'v2',
}

versionsConfig = {
    'Baseline': '',
    'v1': {'batch_size': 4, 'n_workers': 2}
}

ConfigurationDict = {
    'run_config': '',
    'repo_root': '',
    'code_root': '',
    'music_data_root': '',
    'dataset_root': '',
    'labels_root': '',
    'save_dir_root': '',
    'model_version': modelVersions.get(1),
    'batch_size': '',
    'n_workers': ''
}
train_model_conf = versionsConfig.get(ConfigurationDict.get('model_version'))

ConfigurationDict.__setitem__('batch_size', train_model_conf['batch_size'])
ConfigurationDict.__setitem__('n_workers', train_model_conf['n_workers'])
if run_config == 'legion':
    music_dataset_path = os.path.join(music_data_root, 'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed')

elif run_config == 'colab':
    music_dataset_path = os.path.join(music_data_root, 'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG')
else:
    music_dataset_path = os.path.join(music_data_root, 'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG')

if not os.path.exists(music_dataset_path):
    print(f'Path: {music_dataset_path} does not exists. Exiting')
    sys.exit(-1)

if not os.path.exists(save_dir_root):
    os.mkdir(save_dir_root)

music_labels_csv_root = os.path.join(music_data_root, '[labels]emotion_average_dataset_csv')
save_music_emo_csv_path = os.path.join(music_labels_csv_root, 'music_emotions_labels.csv')

ConfigurationDict.__setitem__('run_config', run_config)
ConfigurationDict.__setitem__('repo_root', repo_root)
ConfigurationDict.__setitem__('code_root', code_root)
ConfigurationDict.__setitem__('music_data_root', music_data_root)
ConfigurationDict.__setitem__('dataset_root', music_dataset_path)
ConfigurationDict.__setitem__('labels_root', music_labels_csv_root)
ConfigurationDict.__setitem__('save_dir_root', save_dir_root)


if __name__ == '__main__':

    if verbose:
        print(f'\n\n\nStarting main with: {music_data_root} as the root for music data\n'
              f'{music_dataset_path} as the path of the raw audio data'
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

        b = Benchmark("[main.py] pytorch_dataset_timer")
        b.start_timer()
        # Create the Dataset Object
        pytorch_dataset = emoMusicPTDataset(slice_mode=False, env=ConfigurationDict)
        print(f'\n***** [main.py]: emoMusicPT created*****\n\temoMusic slice_mode: {pytorch_dataset.slice_mode}\n\n')

        # Make Train/Test splits indexes (at song level) -> maintain the order inside the song
        test_frac = 0.1
        train_indexes, test_indexes = pytorch_dataset.stratified_song_level_split(test_fraction=test_frac)
        # Plot splits
        pytorch_dataset.plot_indices_distribution(pytorch_dataset.labels_song_level, train_indexes, test_indexes, val_indexes=None)
        # train_indexes, val_indexes = pytorch_dataset.stratified_song_level_split(test_fraction=test_frac)

        # Defines Dataloaders
        train_set = emoMusicPTSubset(pytorch_dataset, train_indexes)
        test_set = emoMusicPTSubset(pytorch_dataset, test_indexes)
        # val_set = emoMusicPTSubset(pytorch_dataset, val_indexes)
        print(f'\n***** [main.py]: emoMusicPTSubset for train/val/test created *****\n\n')

        train_DataLoader = emoMusicPTDataLoader(train_set, batch_size=ConfigurationDict.get('batch_size'), shuffle=False, num_workers=int(ConfigurationDict.get('n_workers')))
        test_DataLoader = emoMusicPTDataLoader(test_set, batch_size=ConfigurationDict.get('batch_size'), shuffle=False, num_workers=int(ConfigurationDict.get('n_workers')))
        # val_DataLoader = emoMusicPTDataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
        val_DataLoader = None
        print(f'\n***** [main.py]: emoMusicPTDataLoader for train/test created *****\n\n')
        b.end_timer()
        del b

        b = Benchmark("[main.py] pytorch_model_timer")
        b.start_timer()
        first_model = TorchModel(pytorch_dataset, train_DataLoader, test_DataLoader, val_DataLoader, save_dir_root=save_dir_root, version=ConfigurationDict.get("model_version"))
        b.end_timer()
        del b

        b = Benchmark("[main.py] runner_timer")
        b.start_timer()
        runner = Runner(first_model)
        exit_code = runner.train()
        b.end_timer()
        del b

        if exit_code == runner.SUCCESS:
            print(f'[main.py] Training Done! exit_code: {exit_code} -> SUCCESS')

            b = Benchmark("[main.py] runner.eval() timer")
            b.start_timer()
            exit_code = runner.eval()
            b.end_timer()
            del b

            if exit_code == runner.SUCCESS:
                print(f'[main.py] Evaluation of Test set Done! exit_code: {exit_code} -> SUCCESS')
            elif exit_code == runner.FAILURE:
                print(f'[main.py] Evaluation Failed! exit_code: {exit_code} -> FAILURE')
            else:
                print(f'[main.py] Evaluation returned with a not expected exit code {exit_code} -> unknown')
        elif exit_code == runner.FAILURE:
            print(f'[main.py] Training Failed! exit_code: {exit_code} -> FAILURE')
        else:
            print(f'[main.py] Training returned with a not expected exit code {exit_code} -> unknown')

    # %%
    sys.exit(0)
