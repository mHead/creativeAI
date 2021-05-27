# This script contains map functions from point of coordinates (V, A) to emotion.
# 21 may '21 : va2emotion_atan2 function + get_nearest
# 25 may '21 : generate csv containing music and emotion label every 500ms
import os
import math as m
from typing import List, Any, Union

import numpy as np
import pandas as pd

EMOTIONS = {
    0: 'amusement',
    1: 'contentment',
    2: 'awe',
    3: 'excitement',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'sadness',
    8: 'something else'
}

QUADRANTS_EMOTIONS = {
    'Q1.1': 0,
    'Q1.2': [2, 3],
    'Q2.1': [4, 6],
    'Q2.2': 5,
    'Q3.1': 7,
    'Q3.2': 7,
    'Q4.1': 1,
    'Q4.2': 1
}
EMOTION_COORDINATES = {  # Q4.1 and Q3.2 are not used
    0: [0.55, 0.2],  # amusement     Q1.1
    1: [0.82, -0.56],  # contentment   Q4.2
    2: [0.42, 0.89],  # awe           Q1.2 <<
    3: [0.7, 0.72],  # excitement    Q1.2 <<
    4: [-0.42, 0.78],  # anger         Q2.1 --
    5: [-0.67, 0.49],  # disgust       Q2.2
    6: [-0.11, 0.78],  # fear          Q2.1 --
    7: [-0.82, -0.4],  # sadness       Q3.1
    8: [0, 0],  # something else
}
musicSide_root = r'/Users/head/Documents/GitHub/creativeAI/saves_dir/musicSide/annotations_average_csv'

arousal_mean_csv_ = r'arousal_cont_average.csv'
valence_mean_csv_ = r'valence_cont_average.csv'

arousal_std_csv_ = r'arousal_cont_std.csv'
valence_std_csv_ = r'valence_cont_std.csv'

arousal_mean_csv_absolute = os.path.join(musicSide_root, arousal_mean_csv_)
valence_mean_csv_absolute = os.path.join(musicSide_root, valence_mean_csv_)

save_path_relative = r'/Users/head/Documents/GitHub/creativeAI/saves_dir/musicSide/annotations_music-emotion_cont'
if not os.path.exists(save_path_relative):
    os.mkdir(save_path_relative)

music_emotion_csv_ = r'music_emotion_values.csv'
save_path_absolute_csv = os.path.join(save_path_relative, music_emotion_csv_ )

'''
get_nearest(valence, arousal, string, verbose) function

return nearest emotion
TODO: handle the case in which the distances are equals
'''


def get_nearest(x, y, quadrant=None, verbose=False):
    emotions = QUADRANTS_EMOTIONS.get(quadrant)
    min_distance = np.Inf
    max_distance = 0

    for e in emotions:

        xe = EMOTION_COORDINATES.get(e)[0]
        ye = EMOTION_COORDINATES.get(e)[1]
        if verbose:
            print(f'Euclidean distance between: {x, y} and {xe, ye} : {EMOTIONS.get(e)}')
        euclidean_distance = m.sqrt((xe - x) ** 2 + (ye - y) ** 2)
        if euclidean_distance < min_distance:
            max_distance = min_distance
            min_distance = euclidean_distance
            target_emotion_label = e
        else:
            max_distance = euclidean_distance
    if verbose:
        print(
            f"The point (V:{x}, A{y}) is classified as {target_emotion_label}:{EMOTIONS.get(target_emotion_label)}, "
            f"since {min_distance} < {max_distance}")
    return target_emotion_label


'''
va2emotion_atan2(valence, arousal) function

return emo_idx, emotion
TODO: handle (0,0), axis and bisectors
'''


def va2emotion_atan2(x, y, verbose=False):
    sign = x * y

    if sign > 0:  # Q1 || Q3 atan2(y, x) will give me the same angle
        angle = m.degrees(m.atan2(y, x))
        if verbose:
            print(f'{angle}')

        if y > 0 and x > 0:  # Q1
            if angle < 45.0:
                if verbose:
                    print(f'We are in Q1.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q1.1')
                emotion = EMOTIONS.get(emo_idx)

            else:
                if verbose:
                    print(f'We are in Q1.2')
                # emo_idx = QUADRANTS_EMOTIONS.get('Q1.2')
                emo_idx = get_nearest(x, y, 'Q1.2')
                emotion = EMOTIONS.get(emo_idx)

        elif y < 0 and x < 0:
            if angle < -135.0:
                if verbose:
                    print(f'We are in Q3.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q3.1')
                emotion = EMOTIONS.get(emo_idx)

            else:
                if verbose:
                    print(f'We are in Q3.2, classify as Q3.1 because none is here')
                emo_idx = QUADRANTS_EMOTIONS.get('Q3.2')
                emotion = EMOTIONS.get(emo_idx)

    elif sign < 0:  # Q2 || Q4
        angle = m.degrees(m.atan2(y, x))
        if verbose:
            print(f'{angle}')
        if x < 0:  # Q2
            if angle < 135.0:
                if verbose:
                    print(f'We are in Q2.1')
                # emo_idx = QUADRANTS_EMOTIONS.get('Q2.1')
                emo_idx = get_nearest(x, y, 'Q2.1')
                emotion = EMOTIONS.get(emo_idx)
            else:
                if verbose:
                    print(f'We are in Q2.2')
                emo_idx = QUADRANTS_EMOTIONS.get('Q2.2')
                emotion = EMOTIONS.get(emo_idx)
        elif y < 0:  # Q4
            if angle < -45.0:
                if verbose:
                    print(f'We are in Q4.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q4.1')
                emotion = EMOTIONS.get(emo_idx)
            else:
                if verbose:
                    print(f'We are in Q4.2')
                emo_idx = QUADRANTS_EMOTIONS.get('Q4.2')
                emotion = EMOTIONS.get(emo_idx)

    else:  # on axis, sign == 0
        if verbose:
            print(f'y: {y}, x: {x}')
        # emo_idx =
    return emo_idx


#emo_label = va2emotion_atan2(-0.2, 1.0)
#print(f'label: {emo_label}')


def va2emotions(valences, arousals):
    emotions = []
    for i in range(len(valences)):
        emotions.append(va2emotion_atan2(valences[i], arousals[i]))

    return emotions

'''
generate music-continuous-emotion-csv:
generateMusicEmo_csv

'''


def generateMusicEmo_csv(save_path_absolute, valence_path, arousal_path):
    def update_dict(dictionary, _id, array):
        if _id not in dictionary.keys():
            #print(f'adding {_id} : {array}')
            dictionary.update({_id: array})

    valence_mean_x = {}
    arousal_mean_y = {}
    songs_id = []

    valence_df = pd.read_csv(valence_path)
    arousal_df = pd.read_csv(arousal_path)

    # valence_mean_dict_update
    for index, row in valence_df.iterrows():
        song_id = row['song_id'].astype('int')
        songs_id.append(song_id)
        song_values = row[1:]
        song_values = song_values.to_numpy()
        update_dict(valence_mean_x, song_id, song_values)

    # arousal_mean_dict_update
    for index, row in arousal_df.iterrows():
        song_id = row['song_id'].astype('int')
        song_values = row[1:]
        song_values = song_values.to_numpy()
        update_dict(arousal_mean_y, song_id, song_values)


    print(f'{len(songs_id)}, {len(arousal_mean_y)}, {len(valence_mean_x)}')
    assert len(songs_id) == len(arousal_mean_y) == len(valence_mean_x)

    # select the first row of the dataframecre
    emotions_per_song = pd.DataFrame(None, index=['song_id'], columns=list(arousal_df))
    #print(f'{arousal_mean_y}')

    counter = 0
    for sid in valence_mean_x.keys():
        counter += 1
        valences = valence_mean_x.get(sid)
        arousals = arousal_mean_y.get(sid)
        emotions = va2emotions(valences, arousals)
        print(f'sid: {sid} len: {len(emotions)}\n{emotions}')
        if counter == 20:
            break







generateMusicEmo_csv(save_path_absolute_csv, valence_mean_csv_absolute, arousal_mean_csv_absolute)
