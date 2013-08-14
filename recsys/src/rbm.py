import numpy as np
import pandas as pd

import logging

from sklearn.base import BaseEstimator, ClassifierMixin

class GaussianRBM(BaseEstimator, ClassifierMixin):
    pass

class ConditionalRBM(BaseEstimator, ClassifierMixin):
    pass

class ConditionalFactoredRBM(BaseEstimator, ClassifierMixin):
    pass

def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(x):
    return x

def bernoulli(p):
    return np.random.rand(*p.shape) < p

def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    return e / numpy.sum(e, axis=1)

class RBM(BaseEstimator, ClassifierMixin):
    def __init__(self, T=1, n_hidden=100, hidden_type='binary', scale = 0.001, rating_levels=[1,2,3,4,5],
                 momentum=0., lam=0.01, target_sparsity=None, batch_size=200, learning_rate=0.2):
        self.T = T
        self.n_hidden = n_hidden
        self.hidden_type = hidden_type
        self.scale = scale
        self.rating_levels = pd.Index(rating_levels)
        self.momentum = momentum
        self.lam = lam
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.h_bias_ = np.array(None)
        self.v_bias_ = np.array(None)
        self.weights_ = np.array(None)
        self._hidden = None

    def fit(items, ratings):
        self._item_index = items.index
        self.h_bias_ = 2 * self.scale * np.random.randn(self.n_hidden)
        self.v_bias_ = self.scale * np.random.randn(self._item_index.shape[0], self.n_rating_levels)
        self.weights_ = self.scale * np.random.randn(self.n_hidden, self.n_visible)
        ## self._hidden = self.hidden_type=='binary' and sigmoid or identity

        for uid in ratings.groupby('user_id').groups:
            gradients = self.calculate_gradients(visible)
            self.apply_gradients(*gradients, learning_rate=learning_rate)
        
    @property
    def n_visible(self): return self.v_bias_.shape[0]

    @property
    def n_rating_levels(self): return len(self.rating_levels)

    def iter_passes(self, visible, item_indices=None):
        if item_indices is None:
            item_indices = slice(None, None)
        while True:
            hidden = sigmoid(np.dot(self.weights_[:,item_indices], visible.T).T + self.h_bias)
            yield visible, hidden
            visible = softmax(np.dot(hidden, self.weights[:,item_indices]) + self.v_bias[item_indices])

    def reconstruct(self, visible, passes=1):
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i+1 == passes:
                return visible
            
    def calculate_gradients(self, visible_batch):
        passes = self.rbm.iter_passes(visible_batch)
        v0, h0 = passes.next()
        v1, h1 = passes.next()

        ## TODO: use analytic gradients, cd-T
        gw = (np.dot(h0.T, v0) - np.dot(h1.T, v1)) / len(visible_batch)
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)
        if self.target_sparsity is not None:
            gh = self.target_sparsity - h0.mean(axis=0)

        logging.debug('dispacement: %.3g, hidden std: %.3g', np.linalg.norm(gv), h0.std(axis=1).mean())

        return gw, gv, gh

    def apply_gradients(self, weights, visible, hidden, learning_rate=0.2):
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (g - l2 * target)
            target += learning_rate * g
            _g[:] = g

        update('v_bias', visible, self.grad_vis)
        update('h_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)
