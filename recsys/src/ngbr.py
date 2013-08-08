import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from collections import defaultdict

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
        n_user = user_data.shape[0]
        n_item = bus_data.shape[0]
        ## self._b_user = pd.Series(np.zeros(n_user), index=user_data.index)
        ## self._b_item = pd.Series(np.zeros(n_item), index=bus_data.index)
        ## self._w_ij = sp.sparse.coo_matrix(None, shape=(n_user, n_item))
        self._b_user = defaultdict(lambda: 0)
        self._b_item = defaultdict(lambda: 0)
        self._w_ij = defaultdict(lambda: 0)
        l = self.lam
        g = self.gamma
        mu = review_data['stars'].mean()
        self._mu = mu
        self._review_data = review_data[['user_id', 'business_id', 'stars']]
        for _ in xrange(self.n_iter):
            for i, (uid, bid) in review_data[['user_id', 'business_id']] \
              .iterrows():
                err = self._pred([uid, bid])
                self._b_user[uid] += g * (err - l * self._b_user[uid])
                self._b_item[bid] += g * (err - l * self._b_item[bid])
                R = review_data[['business_id', 'stars']][review_data['user_id']==uid]
                for _, (bid2, rat) in R.iterrows():
                    base_rat = mu + self._b_user[uid] + self._b_item[bid2]
                    self._w_ij[(bid, bid2)] += g * (R.shape[0] ** -0.5 * err * (rat - base_rat) - l * self._w_ij[(bid, bid2)])
        return self

    def _pred(self, p):
        uid, bid = p
        rhat = self._mu + self._b_user[uid] + self._b_item[bid]
        R = self._review_data[['business_id', 'stars']][self._review_data['user_id']==uid]
        rhat += (R.shape[0]**-0.5)*R.apply(lambda r: (r['stars']-(self._mu+self._b_user[uid]+self._b_item[r['business_id']])) * self._w_ij[(bid,r['business_id'])],axis=1).sum()
        return rhat

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
        return X[['user_id', 'business_id']].apply(self._pred, axis=1)
