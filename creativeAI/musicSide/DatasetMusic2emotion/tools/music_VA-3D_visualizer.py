import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

musicSide_root = r'/Users/head/Desktop/annotations_Average_csv'

arousal_mean_csv_ = r'arousal_cont_average.csv'
valence_mean_csv_ = r'valence_cont_average.csv'

arousal_std_csv_ = r'arousal_cont_std.csv'
valence_std_csv_ = r'valence_cont_std.csv'

# VA mean
arousal_mean_csv_absolute = os.path.join(musicSide_root, arousal_mean_csv_)
valence_mean_csv_absolute = os.path.join(musicSide_root, valence_mean_csv_)

arousal_mean_df = pd.read_csv(arousal_mean_csv_absolute)
valence_mean_df = pd.read_csv(valence_mean_csv_absolute)

# VA std
arousal_std_csv_absolute = os.path.join(musicSide_root, arousal_std_csv_)
valence_std_csv_absolute = os.path.join(musicSide_root, valence_std_csv_)

arousal_std_df = pd.read_csv(arousal_std_csv_absolute)
valence_std_df = pd.read_csv(valence_std_csv_absolute)


def update_dict(dictionary, _id, array):
    if _id not in dictionary.keys():
        dictionary.update({_id: array})


valence_mean_x = {}
arousal_mean_y = {}

songs_id = []

# valence_mean_dict_update
for index, row in valence_mean_df.iterrows():
    song_id = row['song_id'].astype('int')
    songs_id.append(song_id)
    song_values = row[1:]
    song_values = song_values.to_numpy()
    update_dict(valence_mean_x, song_id, song_values)

# arousal_mean_dict_update
for index, row in arousal_mean_df.iterrows():
    song_id = row['song_id'].astype('int')
    song_values = row[1:]
    song_values = song_values.to_numpy()
    update_dict(arousal_mean_y, song_id, song_values)

assert (len(songs_id) == len(arousal_mean_y) == len(valence_mean_x))

sid = 24
time = np.arange(0, 61, 1)
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(time, valence_mean_x[sid], arousal_mean_y[sid])
ax.set_title(f'V-A mean values for emoMusic song_id: {sid}')
ax.set_xlabel('time-steps')
ax.set_ylabel('Valence mean (V)')
ax.set_zlabel('Arousal mean (A)')
ax.margins(tight=True)
plt.show()
