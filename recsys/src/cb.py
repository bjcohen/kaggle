import numpy as np
import scipy as sp
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import sklearn

from sklearn.metrics.pairwise import cosine_similarity
from scipy import corrcoef
import scipy.sparse
import scipy.sparse.linalg

import eda

def pd_to_sp(x):
    return sp.sparse.csr_matrix(np.array(x).astype(np.float))

def r_features(review_data):
    '''
    used: stars, date, votes, text
    dropped in test: stars, date, votes, text
    '''
    return review_data.loc[:,['stars', 'date', 'votes', 'text']]

def u_features(user_data):
    '''
    used: review_count, average_stars, votes
    dropped in test: average_stars, votes
    not used: name
    '''
    return np.matrix(user_data.loc[:,['review_count', 'average_stars', 'votes_useful', 'votes_funny', 'votes_cool']]).astype(np.float)

def b_features(bus_data, checkin_data):
    '''
    used: stars, review_count, open, categories, city
    dropped in test: stars
    unused: full_address, latitude, longitude, name, state
    '''
    cats_dv = DictVectorizer()
    cats = cats_dv.fit_transform(bus_data.loc[:,'categories'].map(lambda x: {c:1 for c in x}))
    cities_dv = DictVectorizer()
    cities = cities_dv.fit_transform(bus_data.loc[:,'city'].map(lambda x: {x:1}))

    n = bus_data.shape[0]

    checkin_data['bindex'] = bus_data.index.reindex(checkin_data.index)[1]
    datamat = np.vstack([np.array([[v] + map(int, k.split('-')) + [x.loc['bindex']] for k, v in x.loc['checkin_info'].iteritems()])
                         for _, x in checkin_data.iterrows()])
    
    hours = sp.sparse.coo_matrix((datamat[:,0], (datamat[:,3], datamat[:,1])), shape=(n, 24)).tocsr()
    days = sp.sparse.coo_matrix((datamat[:,0], (datamat[:,3], datamat[:,2])), shape=(n, 7)).tocsr()
    
    return sp.sparse.hstack([pd_to_sp(bus_data.loc[:,['stars','review_count','open']]), cats, cities, hours, days], format='csr')

class PairwisePreferenceRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1., loss='user'):
        self.lam = lam
        self.loss = loss
        self.w_ = None

    def fit(self, x, z, r, u_groups=None):
        C = x.shape[1]
        D = z.shape[1]
        if self.loss == 'global':
            w1 = self.lam * sp.sparse.eye(C*D).tocsr()
            w2 = sp.sparse.csr_matrix((1, C*D))
            for i in range(r.shape[0]):
                xu = x[r[i,1],:]
                zi = z[r[i,2],:]
                w1 = w1 + sp.sparse.kron(zi.T * zi, xu.T * xu)
                w2 = w2 + r[i,0] * sp.sparse.kron(zi, xu)
            self.w_ = sp.sparse.linalg.inv(w1) * w2.T
        elif self.loss == 'user':
            A = self.lam / 2 * sp.sparse.eye(C*D)
            B = sp.sparse.csr_matrix((1, C*D))
            ii = 0
            for u, rs in u_groups:
                if ii % 100000 == 0: print 'on user %d' % ii
                ii += 1
                zbar = 1. / len(rs) * sp.sparse.csr_matrix(z[r[rs,2]].sum(axis=0))
                xu = x[r[u,1],:]
                for i in rs:
                    zi = z[r[i,2],:]
                    A = A + sp.sparse.kron(zi.T * (zi - zbar), xu.T * xu)
                    B = B + r[i,0] * sp.sparse.kron(zi - zbar, xu)
            self.w_ = sp.sparse.linalg.inv(A) * B.T
        else:
            raise Exception
        return self

    def predict(self, x, z, r):
        s = np.zeros(r.shape[0])
        for i in range(r.shape[0]):
            s[i] = (sp.sparse.kron(x[r[i,0]], z[r[i,1]]) * self.w_).todense()
        return s

if __name__ == '__main__':
    (bus_data, review_data, user_data, checkin_data) = eda.get_train_data()
    (bus_data_test, review_data_test, user_data_test, checkin_data_test) = eda.get_test_data()
    (bus_data_final, review_data_final, user_data_final, checkin_data_final) = eda.get_final_data()

    review_data_mat = np.array(review_data.loc[:,['stars','user_id','business_id']])
    review_data_mat[:,1] = user_data.index.reindex(review_data_mat[:,1])[1]
    review_data_mat[:,2] = bus_data.index.reindex(review_data_mat[:,2])[1]

    u_feat = u_features(user_data)
    b_feat = b_features(bus_data, checkin_data)

    review_data_mat_s = review_data_mat
    u_groups = ((user_data.index.get_loc(k), review_data.index.reindex(v)[1]) for (k, v) in review_data.groupby('user_id').groups.iteritems() if k in user_data.index)
    
    ppr = PairwisePreferenceRegression(lam = 1.)
    ppr.fit(u_feat, b_feat, review_data_mat_s, u_groups)
    ppr.predict(u_feat, b_feat, review_data_mat_s[:,[1,2]])
