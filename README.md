# creativeAI
**Code is under development** <br>
Master's Degree Thesis: translate music language in the visive art language. <br>
Codebase created and maintained by Marco Testa. <br>

## Introduction
Given:
```
- Music domain : D(music) = M
- Image domain : D(image) = I
```
The aim of the work is to build up a Deep Neural Network capable of translating from M to I.

## Datasets
https://cvml.unige.ch/databases/emoMusic/ but if you want the preprocessed version used in this work, contact me at marco_testa@icloud.com

## High-level pipeline
```
*.wav -> EmotionExtractor -> Generator -> painting
```
### Installation
This code has been tested with Python 3.6.9, Pytorch 1.3.1, CUDA 10.0 on Ubuntu 16.04.

Assuming some (potentially) virtual environment and __python 3x__ 
```Console
git clone https://github.com/mHead/creativeAI.git
cd creativeAI
pip install -e .
```
This will install the repo with all its dependencies (listed in setup.py) and will enable you to do things like:
``` 
from creativeAI.musicSide.Model import xx
```
(provided you add this creativeAI repo in your PYTHON-PATH. (Following example is for Colab) 
```
!echo $PYTHONPATH
import os
os.environ['PYTHONPATH'] += ":/content/creativeAI"
!echo $PYTHONPATH
```

### Pretrained Models
   * [Music-to-Emotion](https://www.dropbox.com/)
