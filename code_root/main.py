from musicSide.DatasetMusic2emotion.tools import va2emotion as va2emo
from musicSide.DatasetMusic2emotion.tools import utils as u
from musicSide.DatasetMusic2emotion.DatasetMusic2emotion import DatasetMusic2emotion
import os
import sys

verbose = True

music_data_root = r'/Users/head/Documents/GitHub/creativeAI/musicSide_root_data'
image_data_root = r'/Users/head/Documents/GitHub/creativeAI/imageSide_root_data'
code_root = r'/Users/head/Documents/GitHub/creativeAI/code_root'

music_labels_csv_root = os.path.join(music_data_root, '[labels]emotion_average_dataset_csv')
save_music_emo_csv_path = os.path.join(music_labels_csv_root, 'music_emotions_labels.csv')

if __name__ == '__main__':

    if verbose:
        print(f'Starting main with: {music_data_root} as the root for music data\n'
              f'{image_data_root} as the root for image data\n'
              f'{code_root} as the root for code')

    if verbose:
        print(f'Checking existence of {music_labels_csv_root}'
              f'If exists check if empty, otherwise create the folder and generate the csv')
    # %% Create music-emotions labels csv from arousal and valence csv
    if not os.path.exists(music_labels_csv_root):
        os.mkdir(music_labels_csv_root)
        if verbose:
            print(f'The path did not existed, starting csv generation and save at {save_music_emo_csv_path}')

        va2emo.generateMusicEmo_csv(save_music_emo_csv_path, music_data_root)
    else:
        if not os.listdir(music_labels_csv_root):  # directory is empty
            if verbose:
                print(f'The path exists but is empty, starting csv generation and save at {save_music_emo_csv_path}')

            va2emo.generateMusicEmo_csv(save_music_emo_csv_path, music_data_root)
        else:
            # directory is not empty, file exist
            print(f'\n\n>>The file {save_music_emo_csv_path} already exists\n\n')
            u.getCSV_info(save_music_emo_csv_path)
    # %%

    music2emotion_Dataset = DatasetMusic2emotion(music_data_root)

    print(f'Hey {music2emotion_Dataset}')
    #music2emotion_Dataset.print_shapes()
    sys.exit(0)
    # music2emotion_Dataset.print_shapes()
