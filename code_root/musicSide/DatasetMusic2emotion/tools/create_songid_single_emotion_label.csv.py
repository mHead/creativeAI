import math

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
source_csv_path = r'/Users/head/Documents/GitHub/creativeAI/musicSide_root_data/[labels]emotion_average_dataset_csv/music_emotions_labels.csv'
dst_csv_path = r'/Users/head/Documents/GitHub/creativeAI/musicSide_root_data/[labels]emotion_average_dataset_csv/music_single_emotion_labels.csv'

labels = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'] # 8: there is no something-else
colors = ['#\t', '#FFFF00', '#87CEEB', '#DC143C', '#000080', '#F0E68C', '#C0C0C0', '#696969', '#228B22']


def _max(occ):
    __max = 0
    __index = 0
    for k in range(len(occ)):
        if occ[k] > __max:
            __max = occ[k]
            __index = k

    return __index


def update_dict(dictionary, _id, emotion_label):
    if _id not in dictionary.keys():
        # print(f'adding {_id} : {array}')
        dictionary.update({_id: emotion_label})


dataframe = pd.read_csv(source_csv_path)
print(dataframe.head())

global_emotions_occurrences = np.zeros(8)
percentages = np.zeros(8)

songid_emotion = {}
songid_emotion_distribution = {}

data = []
dataframe_to_save = pd.DataFrame(index=['song_id'], columns=['emotion'])

for index, row in dataframe.iterrows():
    occurrences = np.zeros(8)
    song_id = row['song_id']
    emotions = np.asarray(row[1:])

    for emo in emotions:
        occurrences[emo] += 1
    print(f'one song occurrences {occurrences}\tglobal occurrences:{global_emotions_occurrences}')
    index = _max(occurrences) #index is the emotion i have to write on the dataframe to save

    update_dict(songid_emotion, song_id, index)
    update_dict(songid_emotion_distribution, song_id, emotions)

    # update global occurrences [123, 431..] for all dataset -> then calculate percentages and plot
    for i in range(len(occurrences)):
        global_emotions_occurrences[i] += occurrences[i]





def write_dict_to_csv():
    columns = ['song_id', 'emotion']
    try:
        with open(dst_csv_path, 'w') as csv_to_save:
            writer = csv.writer(csv_to_save)
            for key, value in songid_emotion.items():
                writer.writerow([key, value])
    except IOError:
        print('I/O Error')


def percentage_array(array):
    sum = array.sum()
    print(f'sum of global occurences is: {sum}')
    assert sum == 45384 #61 slices * 744 songs
    for i in range(len(array)):
        percentages[i] = (100 * array[i]) / sum
    print(percentages)
    assert percentages.sum() == 100



assert len(songid_emotion) == 744
print(f'tranforming {global_emotions_occurrences} in percentiles, to explain the dataset from an emotion-centric point of view')


# following artemis colors
def plot_emotion_distribution_pie_chart(labels, sizes):
    explode = (0.2, 0, 0, 0, 0, 0, 0, 0)
    assert len(sizes) == len(labels)
    #plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
    pie_rounded = np.round(sizes, 2)
    pie_labels = []
    for pe in pie_rounded:
        pe = pe.astype('str')
        pie_labels.append(pe+'%')

    patches, texts = plt.pie(sizes, colors=colors, shadow=False, startangle=45, labels=pie_labels)
    plt.legend(patches, labels, loc='best')
    plt.axis('equal')
    plt.title('Emotion Distribution over emoMusic Dataset')
    #plt.tight_layout()
    plt.show()


def plot_emotion_distribution_bar_chart(labels, sizes):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, sizes, align='center', color=colors)
    plt.xticks(y_pos, labels, rotation=15)
    plt.ylabel('emotion percentage value')
    plt.title('Emotion Distribution over emoMusic Dataset')
    plt.tight_layout()
    plt.show()

# histogram is a mapping from bins (interval) to frequencies thus the raw sums
# here we are mapping from one single emotion to its occurrence in the dataset(thus we can have a really small (2-3 emotions as bins) frequency distribution of emotion on the song
# instead we can make bins: all emotions, and plot just the raw values for each row of the dataset
# TODO: the loop is done in
def plot_song_emotion_distribution_frequencies_bar_chart(labels, song_id):
    # use songid_emotion_distribution dictionary to print the emotion_distribution, given a song

    return



def plot_dataset_emotion_distribution_frequencies_bar_chart(labels, dataset_emotions_occurrences):
    freq_series = pd.Series(dataset_emotions_occurrences)
    ax = freq_series.plot(kind='bar', align='center', color=colors, edgecolor='white')
    ax.set_title(f'Dataset emotion distribution on {int(dataset_emotions_occurrences.sum())} bins')
    ax.set_xlabel('emotions')
    ax.set_ylabel('nbins')
    ax.set_xticklabels(labels, rotation=15)
    # ax.set_yticklabels(dataset_emotions_occurrences)
    ax.set_ylim(0, 16000)
    add_labels_to_bars(ax)
    plt.tight_layout()
    plt.show()


def add_labels_to_bars(ax, spacing=5):
    for rect in ax.patches:
        _y = rect.get_height()
        _x = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = 'bottom'
        if _y < 0:
            space *= -1
            va = 'top'

        label = "{:.0f}".format(_y)+' nb'
        ax.annotate(label, (_x, _y), xytext=(0, space), textcoords='offset points', ha='center', va=va)







# TODO: def plot_dataset_color_spectrum(dataset_emotions_occurrences):

percentage_array(global_emotions_occurrences)

plot_emotion_distribution_pie_chart(labels, percentages)
# plot_emotion_distribution_bar_chart(labels, percentages)

#plot_dataset_emotion_distribution_frequencies_bar_chart(labels, global_emotions_occurrences)

#write_dict_to_csv()
