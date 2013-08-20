import numpy as np

import itertools

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
    item_sim : none (no similarity adjustment)
               mse (mse between ratings)
               corr (pearson correlation coef)
               simu (simultaneously learn user similarities)
               edit (edit distance of movie titles)
               supp (support-based)
    
    estimation_method : inc (incrementally compute factors)
                        simu (simultaneously compute all factors by ALS)
    
    Attributes
    ----------
    `p_` : user factors

    `q_` : item factors

    Notes
    -----
    See Bell & Koren ICDM 2007
        Bell, Koren, Volinsky KDD 2007

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

        item_sim_valid_vals = ['none', 'mse', 'corr', 'simu', 'edit', 'supp']:
        if item_sim not in item_sim_valid_vals
            raise RuntimeException('item_sim must be in ' + str(item_sim_valid_vals))
        self.item_sim = item_sim

        est_meth_valid_vals = ['inc', 'simu']
        if estimation_method not in est_meth_valid_vals:
            raise RuntimeException('estimation_method must be in ' + str(est_meth_valid_vals))
        self.estimation_method = estimation_method
        
        self.n_iter = n_iter
        self.eps = eps
        self.alpha = alpha

    def fit(self, users, items, ratings):
        self.p_ = pd.DataFrame(0, index=users.index, columns=range(self.n_factors))
        self.q_ = pd.DataFrame(0, index=items.index, columns=range(self.n_factors))

        if self.estimation_method == 'simu':
            users = ratings.groupby('user_id').groups
            businesses = ratings.groupby('business_id').groups
            
            for ii in range(self.n_iter):
                for uid in users:
                    bids, stars = ratings.loc[users[uid], ['business_id', 'stars']]
                    n_u = len(users[uid])
                    L = self.lam * n_u
                    
                    r_u = np.vstack([np.matrix(stars), np.zeros(n_u)])
                    q_u = np.vstack([np.matrix(self.q_.loc[bids]), L*np.eye(n_u)])
                    p_u = np.vstack([np.matrix(self.p_.loc[uid]), np.zeros((n_u, n_u))])
                    
                    p_u_ = nnqo(q_u.T * q_u, q_u.T * r, eps=self.eps)
                    self.p_.loc[uid] = p_u_[:n_u]
                for bid in businesses:
                    uids, stars = ratings.loc[businesses[bid], ['user_id', 'stars']]
                    n_i = len(businesses[bid])
                    L = self.lam * n_i
                    
                    r_u = np.vstack([np.matrix(stars), np.zeros(n_u)])
                    q_u = np.vstack([np.matrix(self.q_.loc[bid]), L*np.eye(n_u)])
                    p_u = np.vstack([np.matrix(self.p_.loc[uids]), np.zeros((n_u, n_u))])
                    
                    q_u_ = nnqo(p_u.T * p_u, p_u.T * r, eps=self.eps)
                    self.q_.loc[uid] = q_u_[:n_i]
                    
        elif self.estimation_method == 'inc':
            users = ratings.groupby('user_id').groups
            businesses = ratings.groupby('business_id').groups

            for f in range(self.n_factors):
                r_u = ratings.loc[:,'stars']
                p_u = self.p_.loc[ratings.loc[:,'business_id']]
                q_i = self.q_.loc[ratings.loc[:,'user_id']]

                uid_bid_pairs = ratings.ix[:,['user_id','business_id']].itertuples(index=False)
                n_ui = ratings.groupby(['user_id', 'business_id'].agg(len)).loc[uid_bid_pairs]
                res_ui = (np.subtract(r_u, (p_u * q_i).sum(axis=1)) * n_ui) / (n_ui + self.alpha * f) #n_ui's index left over
                res_iu = res_ui.copy()
                res_iu.index = res_ui.index.reorder_levels([1,0])
                
                last_err = 0
                while last_err == 0 or self._err() / last_err < (1 - self.eps):
                    for uid in users:
                        bids = ratings.loc[users[uid],'business_id']
                        r = res_ui.loc[uid].loc[bids]
                        q = q_i.loc[bids]
                        self.p_.loc[uid,f] = np.dot(r, q) / np.dot(q, q)
                    for bid in businesses:
                        uids = ratings.loc[businesses[bid],'user_id']
                        r = res_iu.loc[bid].loc[uids]
                        p = p_u.loc[uids]
                        self.q_.loc[bid,f] = np.dot(r, p) / np.dot(p, p)

        ## TODO: define error functions, do NeighborhoodAdaptiveFactors, item_sims, prediction rule
                        
        if self.item_sim == 'none':
            return self
        else:
            self._err = None
            raise NotImplementedError
            
## calc user and item factors, then recompute user factors given similarity scores

    def predict(self, X):
        raise NotImplementedError
