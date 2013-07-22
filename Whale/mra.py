from scikits.audiolab import Sndfile
import numpy as np
import scipy as sp
import pandas as pd
import pywt


if __name__ == '__main__':
    ntrain = 20000
    nvalid = 10000
    ntest = 54503
    
    data_train = pd.read_csv('data/train.csv', nrows=30000)

    for i, fn, is_whale in data_train.itertuples():
        s = Sndfile('data/train/' + fn, 'r')
        ex = s.read_frames(s.nframes)
        
        s.close()
