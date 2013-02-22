from scikits.audiolab import Sndfile
import numpy as np
import scipy as sp
import pandas as pd
import pywt as wt

train_data = pd.read_csv('data/train.csv', nrows=1000) # 30k total

train_clips = []
for _, data in train_data.iterrows():
    train_clips.append(data['clip_name'])

