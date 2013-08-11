import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from collections import defaultdict

import logging

class KorenNgbr(BaseEstimator, RegressorMixin):
    '''Nearest-Neighbors algorithm from [Koren2008]

    Parameters
    ----------
    gamma: learning rate

    lam: normalization constant for biases and weights

    n_iter: number of training epochs

    k: neighborhood size limit (not used)

    shrinkage: shrinkage constant for item similarities (not used)

    Attributes
    ----------
    `mu_` : global mean (float)

    `b_user_` : dict of user biases

    `b_item_` : dict of item biases

    `w_ij_` : dict of offset weights

    Notes
    -----
    See Factorization Meets the Neighborhood: a Multifaceted
    Collaborative Filtering Model (Koren, 2008) for a
    description of algorithm and parameters (Sec 3)
    '''
    
    def __init__(self, gamma=0.005, lam=0.002, n_iter=1,
                 k=500, shrinkage=50):
        self.gamma = gamma
        self.lam = lam
        self.n_iter = n_iter

        ## not implemented
        self.k = k
        self.shrinkage = shrinkage

        ## model parameters
        self._mu = None                   # global mean
        self._b_user = None               # user bias
        self._b_item = None               # item bias
        self._w_ij = None                 # baseline offsets

    def fit(self, X, y=None):
        '''
        Fit model.

        Parameters
        ----------
        X : array of business_data, review_data, user_data, checkin_data

        y : not used

        Returns
        -------
        self : instance of self
        '''
        (bus_data, review_data, user_data, checkin_data) = X

        self._b_user = pd.Series(0, index=user_data.index)
        self._b_item = pd.Series(0, index=bus_data.index)
        self._w_ij = pd.sparse.frame.SparseDataFrame(None, index=bus_data.index, columns=bus_data.index, default_fill_value=0)
        self._mu = review_data['stars'].mean()

        l = self.lam
        g = self.gamma
        mu = self._mu

        b_user = self._b_user
        b_item = self._b_item
        w_ij = self._w_ij
        
        print 'generating user prediction functions'
        
        def pred_clos(df):
            uid = df.ix[0, 'user_id']
            R_items = df.loc[:,'business_id']
            R_user_bias = mu + b_user.get(uid, default=0) + b_item.loc[R_items]
            offset = df.loc[:,'stars'] - R_user_bias
            R = df.shape[0] ** -0.5
            def f(bid):
                return mu+b_user.get(uid, default=0)+b_item.loc[bid]+R*np.dot(offset, w_ij[bid].loc[R_items])
            return f
        self._preds = review_data.groupby('user_id').agg(pred_clos)

        print 'starting training'
        
        ii = 1
        for _ in xrange(self.n_iter):
            for i, (uid, bid, stars) in review_data[['user_id', 'business_id', 'stars']].iterrows():
                if ii % 100 == 0: print "on review %d" % ii
                ii += 1
                err = stars - self._preds.loc[uid, 0](bid)
                self._b_user.loc[uid] += g * (err - l * self._b_user.loc[uid])
                self._b_item.loc[bid] += g * (err - l * self._b_item.loc[bid])
                R = review_data[['business_id', 'stars']][review_data['user_id']==uid]
                for _, (bid2, rat) in R.iterrows():
                    base_rat = mu + self._b_user.loc[uid] + self._b_item.loc[bid2]
                    self._w_ij[bid, bid2] += g * (R.shape[0] ** -0.5 * err * (rat - base_rat) - l * self._w_ij[bid, bid2])
        return self

    def predict(self, X):
        '''
        Make prediction from fitted model.

        Parameters
        ----------
        X : array of business_data, review_data, user_data, checkin_data

        Returns
        -------
        y : vector of predicted ratings
        '''
        return X.apply(lambda row: self._preds.loc[row['user_id']](row['business_id']), axis=1)
