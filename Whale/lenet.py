# import pywt as wt TODO: wavedec / swt
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
import scipy as sp
from scikits.audiolab import Sndfile
from numpy.fft import fft
import pylab
from ann import *

def stft(x, w=256, h=1):
    W = sp.hamming(w)
    return np.array([fft(W * x[b:b+w]) for b in range(0, len(x)-w, h)])
            
if __name__ == '__main__':
    # ntrain = 20000
    # nvalid = 10000
    ntrain = 800
    nvalid = 200
    ntest = 54503

    train_data = pd.read_csv('data/train.csv')

    f = 2000
    w = 256
    h = 32

    image_shape = (w / 2, (2 * f - w) / h)
    
    data_x = np.zeros((ntrain,) + image_shape, dtype=theano.config.floatX)
    data_y = np.zeros(ntrain, dtype=theano.config.floatX)

    valid_x = np.zeros((nvalid,) + image_shape, dtype=theano.config.floatX)
    valid_y = np.zeros(nvalid, dtype=theano.config.floatX)
    
    for i, fn, is_whale in train_data.itertuples():
        s = Sndfile('data/train/' + fn, 'r')
        ex = s.read_frames(s.nframes)
        tr = np.abs(stft(ex, w, h)).transpose()
        tr = tr[:image_shape[0]]
        s.close()
        if i < ntrain:
            data_x[i,:,:] = tr
            data_y[i] = int(is_whale)
        elif i < ntrain+nvalid:
            valid_x[i-ntrain,:,:] = tr
            valid_y[i-ntrain] = int(is_whale)
        else: break

    print 'finished loading sound data'

    data_x /= data_x.max()

    train_set_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = T.cast(theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True), 'int32')

    valid_set_x = theano.shared(np.asarray(valid_x, dtype=theano.config.floatX), borrow=True)
    valid_set_y = T.cast(theano.shared(np.asarray(valid_y, dtype=theano.config.floatX), borrow=True), 'int32')

    data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

    print 'finished creating as shared'
        
    x = T.tensor3('x')
    y = T.ivector('y')
    
    rng = np.random.RandomState(2)
    
    nkerns = [20, 50]
    filter_size = [(13, 13), (13, 13)]
    batch_size = 20
        
    layer0_input = x.reshape((batch_size, 1) + image_shape)
    
    layer0 = ConvPoolLayer(rng, input=layer0_input,
                           image_shape=(batch_size, 1) + image_shape,
                           filter_shape=(nkerns[0], 1) + filter_size[0], poolsize=(2, 2))
    layer1 = ConvPoolLayer(rng, input=layer0.output,
                           image_shape=(batch_size, nkerns[0],
                                        (image_shape[0]-filter_size[0][0]+1)/2,
                                        (image_shape[1]-filter_size[0][1]+1)/2),
                           filter_shape=(nkerns[1], nkerns[0]) + filter_size[1], poolsize=(2, 2))
    layer2 = HiddenLayer(rng, input=layer1.output.flatten(2),
                         n_in=nkerns[1] *
                           ((image_shape[0]-filter_size[0][0]+1)/2-filter_size[1][0]+1)/2 *
                           ((image_shape[1]-filter_size[0][1]+1)/2-filter_size[1][1]+1)/2,
                         n_out = 500,
                         activation=T.tanh)
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    cost = layer3.neg_log_likelihood(y)
    errors = layer3.errors(y)
    
    model = [layer0, layer1, layer2, layer3]
    trainer = SGDTrain(x, y, data, model, cost, errors, learning_rate=.1, batch_size=batch_size)
    layer3.y_pred(test)
