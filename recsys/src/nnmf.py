import numpy as np
import pandas as pd
import scipy as sp

import itertools
import pdb

from sklearn.base import BaseEstimator, RegressorMixin

def nnqo(A, b, eps=0.001):
    '''Non-negative Quadratic Optimization
    Minimize x.T * A * x - 2*b.T*x st x >= 0
    Used by BellKor for ALS training
    [BellKorICDM2007] Fig. 1
    '''
    k0, k1 = A.shape
    k2, k3 = b.shape

    if k0 != k1 or k1 != k2 or k3 != 1:
        raise RuntimeError('bad shapes')

    x = np.matrix(eps*np.ones(b.shape))
    while True:
        r = A * x - b
        r = np.where(x==0, np.maximum(0, r), r)
        if np.linalg.norm(r, ord=2) < eps:
            return x
        alpha = r.T * r / (r.T * A * r)
        alpha = np.min(np.where(r < 0, -np.divide(x,r), alpha))
        x += np.multiply(alpha, r)

class NNMF(BaseEstimator, RegressorMixin):
    '''Non-Negative Matrix Factorization from [BellKorICDM] and [BellKorKDD]

    Parameters
    ----------
    item_sim : none (no item similarity adjustment)
               mse (mse between ratings)
               corr (pearson correlation coef)
               simu (simultaneously learn user similarities)
               edit (edit distance of movie titles)
               supp (support-based)
               cont (content-based)
    
    estimation_method : inc (incrementally compute factors)
                        simu (simultaneously compute all factors by ALS)
    
    Attributes
    ----------
    `p_` : user factors

    `q_` : item factors

    Notes
    -----
    See Bell & Koren ICDM 2007 (5.1, 5.2)
        Bell, Koren, Volinsky KDD 2007 (4.3, 4.4)

    Solve /Q[u]\ * p_u = /r_u\
          \Lam /         \ 0 /
    As    Qp = r
    By NNLS:
    minimize (Qp-r)^2 over p st p >= 0
           = (Qp-r).T * (Qp-r)
           = p.T * Q.T * Q * p - 2*r.T * Q * p + r.T * r
    So let A = Q.T * Q, b = Q.T * r in QP standard form

    TODO
    ----
    - Date-weighted similarities
    - Content-aware item similarities
    - Content-aware user similarities    
    '''
    def __init__(self, n_factors=100, lam=0.3, item_sim='none', n_iter=20, eps=0.001,
                 alpha=25, estimation_method='simu'):
        
        self.n_factors = n_factors
        self.lam = lam

        item_sim_valid_vals = ['none', 'mse', 'corr', 'simu', 'edit', 'supp', 'cont']
        if item_sim not in item_sim_valid_vals:
            raise RuntimeError('item_sim must be in ' + str(item_sim_valid_vals))
        self.item_sim = item_sim

        est_meth_valid_vals = ['inc', 'simu']
        if estimation_method not in est_meth_valid_vals:
            raise RuntimeError('estimation_method must be in ' + str(est_meth_valid_vals))
        self.estimation_method = estimation_method
        
        self.n_iter = n_iter
        self.eps = eps
        self.alpha = alpha

    def fit(self, users, items, ratings):
        self._ratings = ratings
        self._users = ratings.groupby('user_id').groups
        self._businesses = ratings.groupby('business_id').groups

        all_users_index = users.index.union(ratings.loc[:,'user_id']).unique()
        self.p_ = pd.DataFrame(self.eps,
                               index=all_users_index,
                               columns=range(self.n_factors))
        self.q_ = pd.DataFrame(self.eps,
                               index=items.index,
                               columns=range(self.n_factors))

        if self.estimation_method == 'simu':
            for ii in range(self.n_iter):
                for uid, u_r in self._users.iteritems():
                    stars = ratings.loc[u_r, 'stars']
                    bids = ratings.loc[u_r, 'business_id']
                    n_u = len(u_r)
                    L = self.lam * n_u

                    r_u = np.vstack([np.matrix(stars).T, np.matrix(np.zeros(self.n_factors)).T])
                    q_u = np.vstack([np.matrix(self.q_.loc[bids]), L*np.eye(self.n_factors)])
                    
                    self.p_.loc[uid] = np.squeeze(np.asarray(nnqo(q_u.T * q_u, q_u.T * r_u, eps=self.eps)))

                for bid, b_r in self._businesses.iteritems():
                    stars = ratings.loc[b_r, 'stars']
                    uids = ratings.loc[b_r, 'user_id']
                    n_i = len(b_r)
                    L = self.lam * n_i
                    
                    r_u = np.vstack([np.matrix(stars).T, np.matrix(np.zeros(self.n_factors)).T])
                    p_u = np.vstack([np.matrix(self.p_.loc[uids]), L*np.eye(self.n_factors)])

                    self.q_.loc[uid] = np.squeeze(np.asarray(nnqo(p_u.T * p_u, p_u.T * r_u, eps=self.eps)))
                    
        elif self.estimation_method == 'inc':
            for f in range(self.n_factors):
                print "factor %d" % f

                uid_bid_pairs = ratings.ix[:,['user_id','business_id']].itertuples(index=False)
                
                r_u = ratings.loc[:,'stars']
                p_u = self.p_.ix[ratings.loc[:,'user_id'],:f-1]
                q_i = self.q_.ix[ratings.loc[:,'business_id'],:f-1]
                
                n_ui = ratings.groupby(['user_id', 'business_id'])['stars'].agg(len).loc[uid_bid_pairs]
                res_ui = np.multiply(n_ui, np.subtract(r_u, p_u.dot(q_i.T).sum(axis=1))) / (n_ui + self.alpha * f) #n_ui's index left over
                
                last_err = 0
                this_err = self._err(ratings)
                print this_err
                while last_err == 0 or (this_err / last_err < (1 - self.eps) or this_err > last_err):
                    for uid, u_r in self._users.iteritems():
                        bids = ratings.loc[u_r,'business_id']
                        r = res_ui.loc[itertools.izip(itertools.repeat(uid), bids)]
                        r.index = r.index.droplevel('business_id')
                        q = self.q_.ix[bids,f]
                        self.p_.loc[uid,f] = np.array(np.divide(r.dot(q), q.T.dot(q)))
                    for bid, b_r in self._businesses.iteritems():
                        uids = ratings.loc[b_r,'user_id']
                        r = res_ui.loc[itertools.izip(uids, itertools.repeat(bid))]
                        r.index = r.index.droplevel('user_id')
                        p = self.p_.ix[uids,f]
                        self.q_.loc[bid,f] = np.array(np.divide(r.dot(p), p.T.dot(p)))
                    last_err = this_err
                    this_err = self._err(ratings)
                    print this_err
                pdb.set_trace()
                print this_err, last_err
                    
        return self

    def _err(self, ratings):
        ## for _, row in ratings.iterrows():
        ##     row.loc['stars']-np.dot(self.p_.loc[row.loc['user_id']], self.q_.loc[row.loc['business_id']])
        return ratings.apply(lambda row: (row.loc['stars']-np.dot(self.p_.loc[row.loc['user_id']], self.q_.loc[row.loc['business_id']]))**2, axis=1).mean()

    def _no_sim(self, row, sims=None):
        return self.p_.loc[row.loc['user_id']]
    
    def _adaptive_factor(self, row, sims):
        '''Adaptive factor learning: KBV KDD 2008 Sec. 4.4'''
        uid, bid = row.loc[['user_id', 'business_id']]
        res = ratings.loc[self.users[uid]].set_index('user_id')
        p = self.p_.loc[uid].copy()
        n = len(res.shape[0])
        
        for l in range(self.n_factors):
            s = sims.loc[bid, res.loc[:,'business_id']]
            Q = self.q_.loc[res.loc[:,'business_id']]
            p[l] = np.dot(s, Q) / np.dot(s, Q ** 2)
            for j in res.index:
                res.loc[res] -= p[l]
                res.loc[res] = (n * res.loc[res]) / (n + self.alpha * l)
                
        return p                
    
    def predict(self, X):
        if self.item_sim == 'none':
            self._get_p = self._no_sim
            sims = None
        elif self.item_sim == 'mse' or self.item_sim == 'corr':
            self._get_p = self._adaptive_factor
            sims = sp.sparse.dok_matrix((self.q_.index.shape[0], self.q_.index.shape[0]))
            for ibid in self._businesses:
                for jbid in self._businesses:
                    i = self.q_.index.get_loc(ibid)
                    j = self.q_.index.get_loc(jbid)
                    if self.item_sim == 'mse':
                        sims[i,j] = np.mean((self._reviews.loc[self._businesses[ibid],'stars']-self._reviews.loc[self._businesses[jbid],'stars']) ** 2) ** -6
                    elif self.item_sim == 'corr':
                        sims[i,j] = np.corrcoef(self._reviews.loc[self._businesses[ibid],'stars'], self._reviews.loc[self._businesses[jbid],'stars'])[0,1]
        else:
            ## 'simu', 'edit', 'supp', 'cont'
            raise NotImplementedError

        # for _, row in X.iterrows():
        #     np.dot(self._get_p(row, sims), self.q_.loc[row.loc['business_id']])
            
        return X.apply(lambda row: np.dot(self._get_p(row, sims), self.q_.loc[row.loc['business_id']]), axis=1)
