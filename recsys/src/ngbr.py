import numpy as np
import pandas as pd
import scipy as sp

from sklearn.base import BaseEstimator, RegressorMixin

import logging
import time
import itertools

class KorenIntegrated(BaseEstimator, RegressorMixin):
    '''Integrated model from [Koren2008]

    Parameters
    ----------
    gam1: bias learning rate

    gam2: latent factor learning rate

    gam3: weight learning rate

    lam6: bias regularization constant

    lam7: latent factor regularization constant

    lam8: weight regularization constant

    n_iter: number of training epochs

    n_factors: number of latent factors to use

    k: neighborhood size limit (not used)

    shrinkage: shrinkage constant for item similarities (not used)

    Attributes
    ----------
    `mu_` : global mean (float)

    `b_user_` : dict of user biases

    `b_item_` : dict of item biases

    `w_ij_` : dict of offset weights

    `p_` : matrix of user factors

    `q_` : matrix of item factors

    `y_` : implicit user factors

    `c_` : matrix of implicit item effects

    Notes
    -----
    See Factorization Meets the Neighborhood: a Multifaceted
    Collaborative Filtering Model (Koren, 2008) for a
    description of algorithm and parameters (Sec 5)
    '''
    
    def __init__(self, gam1=0.007, gam2=0.007, gam3=0.001,
                 lam6=0.005, lam7=0.015, lam8=0.015,
                 n_iter=1, n_factors=50,
                 k=300, shrinkage=50,
                 model_type='integrated'):
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.lam6 = lam6
        self.lam7 = lam7
        self.lam8 = lam8
        self.n_iter = n_iter
        self.n_factors = n_factors

        self.model_types = {
            'integrated' : ['p', 'y', 'c', 'w'],
            'svd++' : ['p', 'y'],
            'factor' : ['p'],
            'nsvd' : ['y'],
            'neighbors' : ['w', 'c']
        }

        if model_type not in self.model_types:
            raise RuntimeError('model_type must be in %s' % model_type_allowed)
        self.model_type = model_type

        for feature in set(itertools.chain(*self.model_types.values())):
            self.__setattr__('_use_' + feature, False)
        
        ## not implemented
        self.k = k
        self.shrinkage = shrinkage
        self.similarity = None

        ## model parameters
        self.mu_ = None                   # global mean
        self.b_user_ = None               # user bias
        self.b_item_ = None               # item bias
        self.w_ij_ = None                 # baseline offsets
        self.p_ = None
        self.q_ = None
        self.y_ = None
        self.c_ = None

    def fit(self, X, y=None):
        '''
        Fit model.

        Parameters
        ----------
        X : array of business_data, review_data, review_data_implicit, user_data, checkin_data

        y : not used

        Returns
        -------
        self : instance of self
        '''
        (bus_data, review_data, review_data_implicit, user_data, checkin_data) = X

        for feature in self.model_types[self.model_type]:
            self.__setattr__('_use_' + feature, True)
            
        all_user_index = user_data.index \
          .union(pd.Index(review_data['user_id'].unique())) \
          .union(pd.Index(review_data_implicit['user_id'].unique())) \
          .unique()
        all_user_index = pd.Index(all_user_index)
          
        self.mu_ = review_data['stars'].mean()
        
        user_means = review_data.groupby('user_id')['stars'].agg(np.mean)
        self.b_user_ = pd.Series(all_user_index.map(lambda x: user_means.get(x, 0) - self.mu_), index=all_user_index)

        bus_means = review_data.groupby('business_id')['stars'].agg(np.mean)
        self.b_item_ = pd.Series(bus_data.index.map(lambda x: bus_means.get(x, 0) - self.mu_), index=bus_data.index)

        if self._use_w or self._use_c:
            self._w_ij_index = bus_data.index
        if self._use_w:
            self.w_ij_ = self.w_ij_ or sp.sparse.lil_matrix((bus_data.shape[0], bus_data.shape[0]))
        if self._use_c:
            self.c_ = self.c_ or sp.sparse.lil_matrix((bus_data.shape[0], bus_data.shape[0]))
        if self._use_p:
            self.p_ = self.p_ or pd.DataFrame(0.001*np.random.randn(all_user_index.shape[0], self.n_factors), index=all_user_index, columns=range(self.n_factors))
        if self._use_p or self._use_y:
            self.q_ = self.q_ or pd.DataFrame(0.001*np.random.randn(bus_data.index.shape[0], self.n_factors), index=bus_data.index, columns=range(self.n_factors))
        if self._use_y:
            self.y_ = self.y_ or pd.DataFrame(0.001*np.random.randn(bus_data.index.shape[0], self.n_factors), index=bus_data.index, columns=range(self.n_factors))

        self._review_data = review_data
        self._review_data_implicit = review_data_implicit
        self._review_map = review_data.groupby('user_id').groups
        self._review_implicit_map = review_data_implicit.groupby('user_id').groups

        print 'starting training'
        t2 = time.clock()

        
        for iiter in xrange(self.n_iter):
            irev = 1
            for i, (uid, bid, stars) in review_data[['user_id', 'business_id', 'stars']].iterrows():
                if irev % 100000 == 0: print "on iter %d review %d" % (iiter, irev)
                irev += 1
                err = stars - self._pred(uid, bid)
                invroot_R_mag = len(self._review_map[uid]) ** -0.5
                invroot_N_mag = len(self._review_implicit_map[uid]) ** -0.5
                ## general
                self.b_user_.loc[uid] += self.gam1 * (err - self.lam6 * self.b_user_.loc[uid])
                self.b_item_.loc[bid] += self.gam1 * (err - self.lam6 * self.b_item_.loc[bid])
                
                ## latent
                N_items = self._review_data_implicit.loc[self._review_implicit_map[uid],'business_id']
                if self._use_p and self._use_y:
                    self.q_.loc[bid,:] += self.gam2 * (err * (self.p_.loc[uid,:] +
                        invroot_N_mag * self.y_.loc[N_items].sum(axis=0)) - self.lam7 * self.q_.loc[bid,:])
                elif self._use_p:
                    self.q_.loc[bid,:] += self.gam2 * (err * self.p_.loc[uid,:] - self.lam7 * self.q_.loc[bid,:])
                elif self._use_y:
                    self.q_.loc[bid,:] += self.gam2 * (err * invroot_N_mag * self.y_.loc[N_items].sum(axis=0)
                                                              - self.lam7 * self.q_.loc[bid,:])
                if self._use_p:
                    self.p_.loc[uid,:] += self.gam2 * (err * self.q_.loc[bid,:] - self.lam7 * self.p_.loc[uid,:])
                if self._use_y:
                    self.y_.loc[bid,:] += self.gam2 * (err * invroot_N_mag * self.q_.loc[bid,:] - self.lam7 * self.y_.loc[bid,:])
                    
                ## neighborhood
                if self._use_w or self._use_c:
                    xi = self._w_ij_index.get_loc(bid)

                if self._use_w:
                    w_yi = self._w_ij_index.reindex(review_data.loc[self._review_map[uid],'business_id'])[1]
                    base_rat = self.mu_ + self.b_user_.loc[uid] + self.b_item_.loc[review_data.loc[self._review_map[uid],'business_id']]
                    self.w_ij_[xi,w_yi] = self.w_ij_[xi,w_yi] + self.gam3 * np.subtract(invroot_R_mag * err *
                        np.subtract(review_data.loc[self._review_map[uid],'stars'], base_rat), self.lam8 * self.w_ij_[xi,w_yi].todense())
                if self._use_c:
                    c_yi = self._w_ij_index.reindex(review_data_implicit.loc[self._review_implicit_map[uid],'business_id'])[1]
                    self.c_[xi,c_yi] = self.c_[xi,c_yi] + self.gam3 * np.subtract(invroot_N_mag * err, self.lam8 * self.c_[xi,c_yi].todense())

            self.predicted_ = self.predict(review_data)
            print 'train MSE = %f' % (np.sqrt(np.mean(np.power(review_data.loc[:,'stars'] - self.predicted_, 2))))

        t3 = time.clock()
        print 'finished training in %dm' % ((t3 - t2) / 60.)
        
        return self

    def predict(self, X):
        '''
        Make prediction from fitted model.

        Parameters
        ----------
        X : review_data format dataframe

        Returns
        -------
        y : vector of predicted ratings
        '''
        return X.apply(lambda row: self._pred(row['user_id'], row['business_id']), axis=1)

    def _pred(self, uid, bid):
        '''
        Make prediction from fitted model.

        Parameters
        ----------
        uid : `user_id`

        bid : `business_id`

        Returns
        -------
        y : predicted rating
        '''
        review_data_implicit_user = self._review_data_implicit.loc[self._review_implicit_map[uid]]
        N_items = review_data_implicit_user.loc[:,'business_id']
        invroot_N_mag = review_data_implicit_user.shape[0] ** -0.5

        if self._use_w or self._use_c:
            xi = self._w_ij_index.get_loc(bid)

        latent = 0
        neighborhood = 0
        neighborhood_implicit = 0
        
        if uid in self._review_map:
            general = self.mu_+self.b_user_.loc[uid]+self.b_item_.loc[bid]
            if self._use_w:
                review_data_user = self._review_data.loc[self._review_map[uid]]
                R_items = review_data_user.loc[:,'business_id']
                invroot_R_mag = review_data_user.shape[0] ** -0.5
                b_u = self.mu_ + np.add(self.b_user_.loc[uid], self.b_item_.loc[R_items])
                w_yi = self._w_ij_index.reindex(R_items)[1]
                neighborhood = (invroot_R_mag*self.w_ij_[xi,w_yi].dot(np.subtract(review_data_user.loc[:,'stars'], b_u)))[0]
        else:
            general = self.mu_+self.b_item_.loc[bid]

        if self._use_y and self._use_p:
            latent = np.dot(self.q_.loc[bid,:], np.add(self.p_.loc[uid,:], invroot_N_mag * self.y_.loc[N_items].sum(axis=0)))
        elif self._use_y:
            latent = np.dot(self.q_.loc[bid,:], invroot_N_mag * self.y_.loc[N_items].sum(axis=0))
        elif self._use_p:
            latent = np.dot(self.q_.loc[bid,:], self.p_.loc[uid,:])
        
        if self._use_c:
            c_yi = self._w_ij_index.reindex(N_items)[1]
            neighborhood_implicit = invroot_N_mag * self.c_[xi,c_yi].sum()
            
        return general + latent + neighborhood + neighborhood_implicit
