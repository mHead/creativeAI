import os
import sys
import argparse
import torch
import torchaudio
import torch.cuda as cuda
# utils
from musicSide.DatasetMusic2emotion.tools import va2emotion as va2emo
from musicSide.DatasetMusic2emotion.tools import utils as u
# Datasets
from musicSide.DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
from musicSide.DatasetMusic2emotion.emoMusicPT import emoMusicPTDataset, emoMusicPTDataLoader, emoMusicPTSubset
# Models
from musicSide.Model.CNN_biGRU import CNN_BiGRU
from musicSide.Model.TorchModel import TorchModel
from musicSide.Model.TorchM5 import TorchM5
from musicSide.Model.MFCC_baseline import MFCC_baseline
from musicSide.Model.MEL_baseline import MEL_baseline
from musicSide.Model.Runner import Runner
from musicSide.Model.MEL_Runner import MEL_Runner
from musicSide.Model.Benchmark import Benchmark


which_torch = False
if which_torch:
    print(f'Using torch version: {torch.__version__}')

    if cuda.is_available():
        print(f'\t- GPUs available: {cuda.device_count()}')
        print(f'\t- Current device index: {cuda.current_device()}')
    else:
        print(f'\t- GPUs available: {cuda.device_count()}')
        print(f'\t- Cuda is NOT available\n')


def get_conf(_repo_root):
    _run_config = None
    if repo_root.startswith('/Users/head'):
        _run_config = 'local'
    elif repo_root.startswith('/home/mtesta'):
        _run_config = 'legion'
    elif repo_root.startswith('/content'):
        _run_config = 'colab'
    return _run_config


# main framework switch config
keras_ = False
pytorch_ = True

parser = argparse.ArgumentParser(description='creativeAI/main arguments Parser')

group0 = parser.add_mutually_exclusive_group()
group0.add_argument('-v', '--verbose', action='store_true', help='Run the program in verbose mode')
group0.add_argument('-q', '--quiet', action='store_true', help='Run the program in quiet mode')
parser.add_argument('-r', '--repo_root', dest='repo_root', type=str, help='It must be followed by the repo root path')
parser.add_argument('-csv', '--generatecsv', dest='save_csv_to_path', type=str, required=False,
                    help='If you want to apply the va2emo map')

group1 = parser.add_mutually_exclusive_group()
group1.add_argument('-raw', '--raw_audio', action='store_true', help='Raw-audio task')
group1.add_argument('-mfcc', '--mfcc_coeff', action='store_true', help='MFCC task')
group1.add_argument('-mel', '--mel_spec', action='store_true', help='Mel_spec task')

args = parser.parse_args()

# print('args:', args)
repo_root = args.repo_root
verbose = args.verbose
run_config = get_conf(repo_root)
save_csv_path = args.save_csv_to_path

TASK = None
if args.raw_audio:
    TASK = 'raw'
elif args.mfcc_coeff:
    TASK = 'mfcc'
elif args.mel_spec:
    TASK = 'mel'
else:
    print('No Task selected. Use -raw || -mfcc || -mel')
    sys.exit(-1)

assert run_config is not None
# print(f'\nThe package will be executed on ** {run_config} ** environment configuration\n')

# Configure paths
music_data_root = os.path.join(repo_root, r'musicSide_root_data')
image_data_root = os.path.join(repo_root, r'imageSide_root_data')
code_root = os.path.join(repo_root, r'creativeAI')
save_dir_root = os.path.join(repo_root, r'saves_dir')
music_dataset_path = os.path.join(music_data_root, 'MusicEmo_dataset_raw_wav/clips_30seconds_preprocessed_BIG')

if not os.path.exists(music_dataset_path):
    print(f'Path: {music_dataset_path} does not exists. Exiting')
    sys.exit(-1)

if not os.path.exists(save_dir_root):
    os.mkdir(save_dir_root)

music_labels_csv_root = os.path.join(music_data_root, '[labels]emotion_average_dataset_csv')
save_music_emo_csv_path = os.path.join(music_labels_csv_root, 'music_emotions_labels.csv')

__MODEL__VERSION__ = 5

modelVersions = {
    0: 'Baseline',
    1: 'v1',
    2: 'v2',
    3: 'M5',
    4: 'MEL_resnet_baseline_v1',
    5: 'MEL_resnet_baseline_v2'
}

versionsConfig = {
    'Baseline': '',
    'v1': {'batch_size': 4, 'n_workers': 2},
    'v2': {'batch_size': 4, 'n_workers': 2},
    'M5': {'batch_size': 2, 'n_workers': 2},
    'MEL_resnet_baseline_v1': {'batch_size': 32, 'n_workers': 2},
    'MEL_resnet_baseline_v2': {'batch_size': 16, 'n_workers': 2}
}

ConfigurationDict = {
    'run_config': '',
    'repo_root': '',
    'code_root': '',
    'music_data_root': '',
    'dataset_root': '',
    'labels_root': '',
    'save_dir_root': '',
    'model_version': modelVersions.get(__MODEL__VERSION__),
    'batch_size': '',
    'n_workers': ''
}
train_model_conf = versionsConfig.get(ConfigurationDict.get('model_version'))

ConfigurationDict.__setitem__('batch_size', train_model_conf['batch_size'])
ConfigurationDict.__setitem__('n_workers', train_model_conf['n_workers'])
ConfigurationDict.__setitem__('run_config', run_config)
ConfigurationDict.__setitem__('repo_root', repo_root)
ConfigurationDict.__setitem__('code_root', code_root)
ConfigurationDict.__setitem__('music_data_root', music_data_root)
ConfigurationDict.__setitem__('dataset_root', music_dataset_path)
ConfigurationDict.__setitem__('labels_root', music_labels_csv_root)
ConfigurationDict.__setitem__('save_dir_root', save_dir_root)


# %% PyTorch Main
def evaluate_model(exit_code, runner):
    if exit_code == runner.SUCCESS:
        print(f'[main.py] Training Done! exit_code: {exit_code} -> SUCCESS')

        exit_code = runner.eval()

        if exit_code == runner.SUCCESS:
            print(f'[main.py] Evaluation of Test set Done! exit_code: {exit_code} -> SUCCESS')
            del runner
        elif exit_code == runner.FAILURE:
            print(f'[main.py] Evaluation Failed! exit_code: {exit_code} -> FAILURE')
            del runner
        else:
            del runner
            print(f'[main.py] Evaluation returned with a not expected exit code {exit_code} -> unknown')
    elif exit_code == runner.FAILURE:
        del runner
        print(f'[main.py] Training Failed! exit_code: {exit_code} -> FAILURE')
    else:
        del runner
        print(f'[main.py] Training returned with a not expected exit code {exit_code} -> unknown')

def pytorch_main():

    if pytorch_:
        # Create the Dataset Object, TASK variable will decide how __getitem__ works
        pytorch_dataset = None
        if TASK == 'raw':
            pytorch_dataset = emoMusicPTDataset(slice_mode=False, env=ConfigurationDict)
        elif TASK == 'mfcc':
            pytorch_dataset = emoMusicPTDataset(slice_mode=False, env=ConfigurationDict, mfcc=True)
        elif TASK == 'mel':
            ConfigurationDict.__setitem__('n_fft', 2048)
            ConfigurationDict.__setitem__('hop_length', 1024)
            ConfigurationDict.__setitem__('n_mel', 224)
            ConfigurationDict.__setitem__('sample_rate', 44100)
            pytorch_dataset = emoMusicPTDataset(slice_mode=False, env=ConfigurationDict, melspec=True)

        print(f'\n***** [main.py]: emoMusicPT created for the task: {TASK} *****\n')

        test_frac = 0.1
        train_indexes, test_indexes = pytorch_dataset.stratified_song_level_split(test_fraction=test_frac)

        # Plot splits
        if not pytorch_dataset.slice_mode:
            pytorch_dataset.plot_indices_distribution(pytorch_dataset.labels_song_level, train_indexes, test_indexes,
                                                      val_indexes=None)
        else:
            pytorch_dataset.plot_indices_distribution(pytorch_dataset.labels_slice_level, train_indexes, test_indexes,
                                                      val_indexes=None)
        # train_indexes, val_indexes = pytorch_dataset.stratified_song_level_split(test_fraction=test_frac)

        # Defines Dataloaders
        train_set = emoMusicPTSubset(pytorch_dataset, train_indexes)
        test_set = emoMusicPTSubset(pytorch_dataset, test_indexes)
        # val_set = emoMusicPTSubset(pytorch_dataset, val_indexes)
        print(f'\n***** [main.py]: emoMusicPTSubset for train/test created *****\n\n')

        train_DataLoader = emoMusicPTDataLoader(train_set, batch_size=ConfigurationDict.get('batch_size'),
                                                shuffle=False, num_workers=int(ConfigurationDict.get('n_workers')))
        test_DataLoader = emoMusicPTDataLoader(test_set, batch_size=ConfigurationDict.get('batch_size'), shuffle=False,
                                               num_workers=int(ConfigurationDict.get('n_workers')))
        # val_DataLoader = emoMusicPTDataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
        val_DataLoader = None
        print(f'\n***** [main.py]: emoMusicPTDataLoader for train/test created *****\n\n'
              f'\ttrain_dl len: {len(train_DataLoader)}'
              f'\ttest_dl len:{len(test_DataLoader)}'
              f'\n-------------\n'
              f'\tbatch_size: {ConfigurationDict.get("batch_size")}'
              f'\tn_workers: {ConfigurationDict.get("n_workers")}\n')

        '''
        b = Benchmark("[main.py] pytorch_model_timer")
        b.start_timer()
        baseline_model = TorchModel(pytorch_dataset, train_DataLoader, test_DataLoader, val_DataLoader, save_dir_root=save_dir_root, version=ConfigurationDict.get("model_version"), n_gru=1)

        b.end_timer()
        del b
        '''

        # Defining training Policies
        TrainingSettings = {
            "batch_size": ConfigurationDict.get('batch_size'),
            "epochs": 40,
            "print_preds_every": 250,
            "learning_rate": 0.001,
            "stopping_rate": 1e-7,
            "weight_decay": 0.0001,
            "momentum": 0.8,
            "model_versions": modelVersions,
            "actual_version": __MODEL__VERSION__
        }

        TrainingPolicies = {
            "monitor": 'val_loss',
            "mode": 'min',
            "factor": 0.9,
            "patience": 20,
            "min_lr": 0.000001,
            "verbose": 1
        }

        TrainSavingsPolicies = {
            "plots_save_dir": 'plots',
            "best_models_save_dir": 'best_models',
            "tensorboard_outs": 'tb_outputs',
            "monitor": 'val_categorical_accuracy',
            "quiet": 0,
            "verbose": 1,
            "run_config": ConfigurationDict.get('run_config')
        }

        # collect
        bundle = {**TrainingSettings, **TrainingPolicies, **TrainSavingsPolicies}
        print(TrainingSettings)
        '''
        model, optim, runner_settings, loaded_checkpoint['metadata'] = runner.load_model(path)
        '''
        exit_code = None
        if TASK == 'raw':
            ks_list = [110, 220, 440, 880]
            kfm_list = [8, 8*8, 8*16, 8*32]
            for ks in ks_list:
                for kfm in kfm_list:
                    hyperparams = {
                        "__CONFIG__": __MODEL__VERSION__,
                        "n_input": 1,  # the real audio channel is called n_input
                        "n_output": pytorch_dataset.num_classes,
                        "kernel_size": ks,
                        "kernel_shift": ks // 2,
                        "kernel_features_maps": kfm,
                        # n_channel in the constructor of conv1D (not the channel of audio, here is a misleading nomenclature from documentation)
                        "groups": 1,
                        "dropout": True,
                        "dropout_p": 0.25,
                        "slice_mode": pytorch_dataset.slice_mode
                    }

                    model = TorchM5(hyperparams=hyperparams)

                    model.save_dir = pytorch_dataset.get_save_dir()
                    model.example_0, model.ex0_songid, model.ex0_filename, model.ex0_label, model.slice_no = \
                        train_DataLoader.dataset[0]
                    model.input_shape = model.example_0.shape

                    runner = Runner(_model=model, _train_dl=train_DataLoader, _test_dl=test_DataLoader, _bundle=bundle,
                                    task=TASK)
                    exit_code = runner.train()
                    evaluate_model(exit_code, runner)

        elif TASK == 'mfcc':
            '''
            model = MFCC_baseline(False, hyperparams)
            model.save_dir = pytorch_dataset.get_save_dir()
            model.mfcc_features_dict_ex0, model.ex0_songid, model.ex0_filename, model.ex0_label, model.slice_no = \
                train_DataLoader.dataset[0]
            model.input_shape = model.example_0.shape
            '''
        elif TASK == 'mel':
            dropout_list = [0.1, 0.15, 0.2, 0.25]
            for d in dropout_list:
                hyperparams = {
                    "__CONFIG__": __MODEL__VERSION__,
                    "n_output": pytorch_dataset.num_classes,
                    "dropout": True,
                    "dropout_p": d,
                }
                model = MEL_baseline(verbose=False, hyperparams=hyperparams)
                model.save_dir = pytorch_dataset.get_save_dir()
                model.ex0_mel, model.ex0_songid, model.ex0_filename, model.ex0_label, model.slice_no = train_DataLoader.dataset[0]
                model.input_shape = model.ex0_mel.shape
                runner = MEL_Runner(_model=model, _train_dl=train_DataLoader, _test_dl=test_DataLoader, _bundle=bundle,
                                task=TASK)
                exit_code = runner.train()
                evaluate_model(exit_code, runner)
        else:
            print('Task not defined')
            sys.exit(-1)


def generate_csv():
    # %% Create music-emotions labels csv from arousal and valence csv
    if verbose:
        print(f'Checking existence of {music_labels_csv_root}\n'
              f'If exists: check if empty else: create the folder and generate the csv')
    if save_csv_path is not None:
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
                u.displayCSV_info(save_music_emo_csv_path)


def keras_main():
    if keras_:
        b = Benchmark("keras_dataset_timer")
        b.start_timer()
        music2emotion_Dataset = DatasetMusic2emotion(data_root=music_data_root, train_frac=0.9, run_config=run_config,
                                                     preprocess=False)
        print(f'Hey I am: {music2emotion_Dataset}')
        b.end_timer()

        b = Benchmark("keras_model_timer")
        b.start_timer()
        music2emotion_Model = CNN_BiGRU(music2emotion_Dataset, save_dir=save_dir_root, do_train=True, do_test=False,
                                        load_model=False, load_model_path=(None, None))

        b.end_timer()


if __name__ == '__main__':

    if verbose:
        print(f'\n\nStarting main with env={run_config} with: '
              f'\n\t- {music_data_root} as the root for music data'
              f'\n\t- {music_dataset_path} as the path of the raw audio data'
              f'\n\t- {image_data_root} as the root for image data'
              f'\n\t- {code_root} as the root for code'
              f'\n\n')

    generate_csv()

    if keras_:
        keras_main()
        sys.exit(0)
    elif pytorch_:
        pytorch_main()
        sys.exit(0)
    else:
        print('No main to start')
        sys.exit(0)
