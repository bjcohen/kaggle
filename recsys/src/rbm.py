import numpy as np
import pandas as pd

import logging
import itertools
import time

import eda

from sklearn.base import BaseEstimator, ClassifierMixin

## TODO
class GaussianRBM(BaseEstimator, ClassifierMixin):
    pass

## TODO
class ConditionalFactoredRBM(BaseEstimator, ClassifierMixin):
    pass

def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(x):
    return x

def bernoulli(p):
    return np.random.rand(*p.shape) < p

def softmax(w):
    n = w.shape[0]
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(n, 1)
    e = np.exp(w - maxes)
    return e / np.sum(e, axis=1).reshape(n, 1)

class RBM(BaseEstimator, ClassifierMixin):
    def __init__(self, T=1, n_hidden=100, hidden_type='binary', scale = 0.01, rating_levels=pd.Index([1,2,3,4,5]),
                 momentum=0., lam=0.01, batch_size=200, learning_rate=0.2, epochs=1, conditional=True):
        self.T = T                           # TODO: use adaptive CD-T (increasing T in steps)
        self.n_hidden = n_hidden
        self.hidden_type = hidden_type
        self.scale = scale
        self.rating_levels = rating_levels
        self.momentum = momentum
        self.lam = lam
        self.batch_size = batch_size # TODO: possibly use this - SMH used n=1000 minibatchs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.conditional = conditional

        self.h_bias_ = None
        self.v_bias_ = None
        self.weights_ = None
        self._hidden = None
        self.implicit_weights_ = None

    def fit(self, items, ratings, implicit=None):
        if implicit is None and self.conditional:
            raise RuntimeError('Need implicit ratings for conditional rbm')
        
        self._item_index = pd.Index(items.index.unique())
        self._implicit_map = implicit.groupby('user_id').groups
        self._implicit_map = {k : self._item_index.reindex(implicit.loc[self._implicit_map[k], 'business_id'])[1] for k in self._implicit_map}
        self._ratings = ratings.copy()

        self.h_bias_ = self.h_bias_ if self.h_bias_ is not None else self.scale * np.random.randn(self.n_hidden)
        self.v_bias_ = self.v_bias_ if self.v_bias_ is not None else self.scale *  np.random.randn(items.index.shape[0], self.n_rating_levels)
        self.weights_ = self.weights_ if self.weights_ is not None else  self.scale * np.random.randn(self.n_rating_levels, self.n_hidden, self.n_visible)
        self.implicit_weights_ = self.implicit_weights_ if self.implicit_weights_ is not None else self.scale * np.random.randn(self.n_hidden, self.n_visible)

        self._grad_hid = np.zeros_like(self.h_bias_)
        self._grad_vis = np.zeros_like(self.v_bias_)
        self._grad_weights = np.zeros_like(self.weights_)
        self._grad_implicit_weights = np.zeros_like(self.implicit_weights_)
        if self.conditional:
            self._hidden = self._conditional
        else:
            self._hidden = self._binary
        
        ratings_map = ratings.groupby('user_id').groups
        for ei in range(self.epochs):
            print 'training epoch %d' % ei
            t1 = time.clock()
            for uid in ratings_map:
                ratings_uid = ratings.loc[ratings_map[uid],'stars']
                bids = ratings.loc[ratings_map[uid],'business_id']
                
                n_ratings = ratings_uid.shape[0]
                _, bid_indices = self._item_index.reindex(bids)
                ratings_ = np.zeros((n_ratings, self.n_rating_levels))
                ratings_[np.arange(n_ratings), self.rating_levels.reindex(ratings_uid)[1]] = 1.

                if self.conditional:
                    r = self._implicit_map[uid]
                else: r = None
                gradients = self.calculate_gradients(ratings_, bid_indices, r)
                self.apply_gradients(*gradients, learning_rate=self.learning_rate, item_indices=bid_indices, r=r)
            print 'took %d seconds' % (time.clock() - t1)

    def _binary(self, item_indices, visible, r=None):
        return sigmoid(np.tensordot(self.weights_[:,:,item_indices], visible, axes=([0,2],[1,0])).T + self.h_bias_)

    def _conditional(self, item_indices, visible, r):
        svw = np.tensordot(self.weights_[:,:,item_indices], visible, axes=([0,2],[1,0])).T
        srd = self.implicit_weights_[:,r].sum(axis=1)
        return sigmoid(svw + srd + self.h_bias_)
    
    @property
    def n_visible(self): return self.v_bias_.shape[0]

    @property
    def n_rating_levels(self): return self.rating_levels.shape[0]

    def iter_passes(self, visible, item_indices=None, r=None):
        if item_indices is None:
            item_indices = slice(None, None)
        while True:
            ## weights: ratings x nhidden x nvisible
            ## visible: nvisible x ratings
            ## h_bias : nhidden
            hidden = bernoulli(self._hidden(item_indices, visible, r))
            yield visible, hidden
            ## hidden: nhidden
            ## weights: ratings x nhidden x nvisible
            ## v_bias: nvisible x ratings
            visible = softmax(np.tensordot(hidden, self.weights_[:,:,item_indices], axes=(0, 1)).T + self.v_bias_[item_indices,:])

    def calculate_gradients(self, visible, item_indices=None, r=None):
        passes = self.iter_passes(visible, item_indices, r)
        
        v0, h0 = passes.next()
        vs, hs = zip(*itertools.islice(passes, 0, self.T))
        v1 = np.mean(np.vstack(vs), axis=0)
        h1 = np.mean(np.vstack(hs), axis=0)

        ## h: nhidden
        ## v: nvisible * nratings
        ## w: ratings x nhidden x nvisible
        nhidden = h0.shape[0]
        nvisible, nratings = v0.shape
        gw = np.rollaxis(np.tensordot(v0.T, h0, axes=0), 2, 1) - np.rollaxis(np.tensordot(v1.reshape(self.n_rating_levels, 1), h1, axes=0), 2, 1)
        gv = (v0 - v1)
        gh = (h0 - h1)

        return gw, gv, gh

    def apply_gradients(self, weights, visible, hidden, learning_rate, item_indices=None, r=None):
        if item_indices is None:
            item_indices = slice(None, None)

        self._grad_vis *= self.momentum
        self._grad_vis[item_indices,:] +=  learning_rate * (visible - self.lam * self.v_bias_[item_indices,:])
        self.v_bias_ += self._grad_vis

        self._grad_hid = self.momentum * self._grad_hid + learning_rate * (hidden - self.lam * self.h_bias_)
        self.h_bias_ += self._grad_hid

        self._grad_weights *= self.momentum
        self._grad_weights[:,:,item_indices] += learning_rate * (weights - self.lam * self.weights_[:,:,item_indices])
        self.weights_ += self._grad_weights

        if self.conditional:
            self._grad_implicit_weights *= self.momentum
            self._grad_implicit_weights[:,r] += learning_rate * (hidden - self.lam * self.h_bias_).reshape(self.n_hidden, 1)
            self.implicit_weights_ += self._grad_implicit_weights
        

    def predict(self, to_predict, method='exp'):
        '''
        Make predictions from fitted model

        Parameters
        ----------
        connections : vector of user-item pairs of connections

        to_predict : vector of user-item pairs to predict

        method : method, map = argmax unnormalized score
                         exp = expectation over normalized scores
                         prob = normalized probabilities

        Returns
        -------
        y : predicted ratings
        '''
        if method not in ['exp', 'map', 'prob']:
            raise NotImplementedError

        n_pred = to_predict.shape[0]

        ratings_map = self._ratings.groupby('user_id').groups

        probs = np.zeros((n_pred, self.n_rating_levels))
        
        for qi, (_, (q, uid)) in enumerate(to_predict.loc[:,['business_id', 'user_id']].iterrows()):
            q_index = self._item_index.get_loc(q)

            if uid in ratings_map:
                ratings_uid = self._ratings.loc[ratings_map[uid],'stars']
                bids = self._ratings.loc[ratings_map[uid],'business_id']
                n_ratings = ratings_uid.shape[0]
                _, bid_indices = self._item_index.reindex(bids)
                bid_indices = list(bid_indices)
                bid_indices.append(q_index)
                ratings_ = np.zeros((n_ratings+1, self.n_rating_levels))
                ratings_[np.arange(n_ratings), self.rating_levels.reindex(ratings_uid)[1]] = 1.
            else:
                n_ratings = 0
                bid_indices = [q_index]
                ratings_ = np.zeros((1, self.n_rating_levels))
                
            if self.conditional:
                r = self._implicit_map[uid]
            else:
                r = None

            passes = self.iter_passes(ratings_, bid_indices, r)
            v0, h0 = passes.next()
            v1, _ =  passes.next()

            probs[qi,:] = v1[n_ratings,:]

        ## i = n_visible
        ## q = n_visible
        ## l = n_ratings
        ## k = n_ratings
        ## j = n_hidden

        ## TODO: fix this entire function. it is so ugly and broken
        # pi = np.zeros((n_pred, self.n_rating_levels))
        # gammas = np.zeros((n_pred, self.n_rating_levels))

        # def inner(uid, q, k, j):
        #     q_index = self._ite_mindex.get_loc(q)
        #     k_index = self.rating_levels.get_loc(k)
        #     _, bid_indices = self._item_index.reindex(self._ratings.loc[ratings_map[uid],'business_id'])

        #     user_ratings = self._ratings.loc[ratings_map[uid]]

        #     v_il = zip(self.rating_levels.reindex(user_ratings.loc[:,'stars'])[1], itertools.repeat(j),bid_indices)
        #     W_l = [self.weights_[il] for il in v_il]
            
        #     if q in set(user_ratings.loc[:,'business_id']) and user_ratings.loc[user_ratings.loc[:,'business_id']==q,'stars'] == k:
        #         W_k = self.weights_[k_index,j,q_index]
        #     else:
        #         W_k = 0.

        #     return 1 + np.exp(np.sum(W_l) + W_k + self.h_bias_[j])
            
        # for qi, (_, (q, uid)) in enumerate(to_predict.loc[:,['business_id', 'user_id']].iterrows()):
        #     q_index = self._item_index.get_loc(q)
        #     for k in self.rating_levels:
        #         k_index = self.rating_levels.get_loc(k)
        #         if uid not in ratings_map:
        #             pi[qi, k_index] = np.prod(np.exp(1.+self.h_bias_))
        #             gammas[qi, k_index] = 1.
        #             continue
        #         pi[qi, k_index] = np.prod([inner(uid, q, k, j) for j in range(self.n_hidden)])
        #         if q in set(self._ratings.loc[ratings_map[uid],'business_id']):
        #             gammas[qi, k_index] = np.exp(self.v_bias_[q_index, k_index])
        #         else:
        #             gammas[qi, k_index] = 1.
        
        # probs = gammas * pi

        ## normalize and compute expectation
        probs /= np.sum(probs, axis=1).reshape(n_pred, 1)

        if method == 'exp':
            Y = np.sum(np.array(self.rating_levels).reshape(1, self.n_rating_levels) * probs, axis=1)
        elif method == 'map':
            Y = np.argmax(probs, axis=1)
        elif method == 'prob':
            Y = probs

        return Y

if __name__ == '__main__':
    (bus_data, review_data, user_data, checkin_data) = eda.get_train_data()
    (bus_data_test, review_data_test, user_data_test, checkin_data_test) = eda.get_test_data()
    (bus_data_final, review_data_final, user_data_final, checkin_data_final) = eda.get_final_data()

    r = RBM(epochs=50, conditional=True, rating_levels=pd.Index([1,2,3,4,5]), learning_rate=0.01, lam=0.001, n_hidden=50, momentum=0.9)
    r.fit(pd.concat([bus_data, bus_data_test, bus_data_final]),
          review_data,
          pd.concat([review_data, review_data_test, review_data_final]))
    r_pred = r.predict(review_data_final, method='exp')
    pd.DataFrame({'review_id' : review_data_final.index, 'stars' : np.maximum(0, np.minimum(5, r_pred))}).to_csv('../rbm_submission_e50_h50.csv',index=False)

