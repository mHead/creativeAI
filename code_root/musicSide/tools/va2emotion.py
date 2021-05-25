# This script contains map functions from point of coordinates (V, A) to emotion.
# 21 may '21 : va2emotion_atan2 function + get_nearest

import math as m
import numpy as np

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
    'Q3.2': 'what emotion could lay in this quadrant?',
    'Q4.1': 'what emotion could lay in this quadrant?',
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

'''
get_nearest(valence, arousal, string, verbose) function

return nearest emotion
TODO: handle the case in which the distances are equals
'''


def get_nearest(x, y, quadrant=None, verbose=True):
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


def va2emotion_atan2(x, y):
    sign = x * y

    if sign > 0:  # Q1 || Q3 atan2(y, x) will give me the same angle
        angle = m.degrees(m.atan2(y, x))
        print(f'{angle}')

        if y > 0 and x > 0:  # Q1
            if angle < 45.0:
                print(f'We are in Q1.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q1.1')
                emotion = EMOTIONS.get(emo_idx)

            else:
                print(f'We are in Q1.2')
                # emo_idx = QUADRANTS_EMOTIONS.get('Q1.2')
                emo_idx = get_nearest(x, y, 'Q1.2')
                emotion = EMOTIONS.get(emo_idx)

        elif y < 0 and x < 0:
            if angle < -135.0:
                print(f'We are in Q3.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q3.1')
                emotion = EMOTIONS.get(emo_idx)

            else:
                print(f'We are in Q3.2')
                emo_idx = QUADRANTS_EMOTIONS.get('Q3.2')
                emotion = EMOTIONS.get(emo_idx)

    elif sign < 0:  # Q2 || Q4
        angle = m.degrees(m.atan2(y, x))
        print(f'{angle}')
        if x < 0:  # Q2
            if angle < 135.0:
                print(f'We are in Q2.1')
                # emo_idx = QUADRANTS_EMOTIONS.get('Q2.1')
                emo_idx = get_nearest(x, y, 'Q2.1')
                emotion = EMOTIONS.get(emo_idx)
            else:
                print(f'We are in Q2.2')
                emo_idx = QUADRANTS_EMOTIONS.get('Q2.2')
                emotion = EMOTIONS.get(emo_idx)
        elif y < 0:  # Q4
            if angle < -45.0:
                print(f'We are in Q4.1')
                emo_idx = QUADRANTS_EMOTIONS.get('Q4.1')
                emotion = EMOTIONS.get(emo_idx)
            else:
                print(f'We are in Q4.2')
                emo_idx = QUADRANTS_EMOTIONS.get('Q4.2')
                emotion = EMOTIONS.get(emo_idx)

    else:  # on axis, sign == 0
        print(f'y: {y}, x: {x}')
        # emo_idx =
    return emo_idx, emotion


emo_label, emotion = va2emotion_atan2(-0.2, 1.0)
print(f'emotion: {emotion}, label: {emo_label}')
