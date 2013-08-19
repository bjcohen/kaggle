import numpy as np

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
    '''
    Solve /Q[u]\ * p_u = /r_u\
          \Lam /         \ 0 /
    As    Qp = r
    By NNLS:
    minimize (Qp-r)^2 over p st p >= 0
           = (Qp-r).T * (Qp-r)
           = p.T * Q.T * Q * p - 2*r.T * Q * p + r.T * r
    So let A = Q.T * Q, b = Q.T * r in QP standard form
    '''
    def __init__(self, n_factors=100, lam=0.3, item_sim='none', n_iter=20, eps=0.001):
        self.n_factors = n_factors
        self.lam = lam
        item_sim_valid_vals = ['none', 'mse', 'corr', 'simu', 'edit', 'supp']:
        if item_sim not in item_sim_valid_vals
            raise RuntimeException('item_sim must be in ' + str(item_sim_valid_vals))
        self.item_sim = item_sim
        self.n_iter = n_iter
        self.eps = eps

    def fit(self, users, items, ratings):
        self.p_ = pd.DataFrame(0, index=users.index, columns=range(self.n_factors))
        self.q_ = pd.DataFrame(0, index=items.index, columns=range(self.n_factors))

        users = ratings.groupby('user_id').groups
        businesses = ratings.groupby('')
        for ii in range(self.n_iter):
            for uid in users:
                bids, stars = ratings[users[uid], ['business_id', 'stars']]
                n_u = len(users[uid])
                L = self.lam * n_u

                r_u = np.vstack([np.matrix(stars), np.zeros(n_u)])
                q_u = np.vstack([np.matrix(self.q_.loc[bids]), L*np.eye(n_u)])
                p_u = np.vstack([np.matrix(self.p_.loc[uid]), np.zeros((n_u, n_u))])

                p_u_ = nnqo(q_u.T * q_u, q_u.T * r, eps=self.eps)
                
                ## q-phase
            

    def predict(self, X): pass
