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

    ## cats, cities
    return sp.sparse.hstack([pd_to_sp(bus_data.loc[:,['stars','review_count','open']]), hours, days], format='csr')

class PairwisePreferenceRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1., loss='user'):
        self.lam = lam
        self.loss = loss
        self.w_ = None

    def fit(self, x, z, r, u_groups=None):
        _sparse = sp.sparse.issparse(x) or sp.sparse.issparse(z)

        C = x.shape[1] + 1
        D = z.shape[1]
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        r = r.copy()
        self.r_mean_ = r[:,0].mean()
        r[:,0] -= self.r_mean_
        if self.loss == 'global':
            if _sparse:
                self.A_ = self.lam * sp.sparse.eye(C*D).tocsr()
                self.B_ = sp.sparse.csr_matrix((1, C*D))
            else:
                self.A_ = self.lam * np.eye(C*D)
                self.B_ = np.matrix(np.zeros((1, C*D)))
            for i in range(r.shape[0]):
                xu = x[r[i,1],:]
                zi = z[r[i,2],:]
                if _sparse:
                    self.A_ = self.A_ + sp.sparse.kron(zi.T * zi, xu.T * xu)
                    self.B_ = self.B_ + r[i,0] * sp.sparse.kron(zi, xu)
                else:
                    self.A_ = self.A_ + np.kron(zi.T * zi, xu.T * xu)
                    self.B_ = self.B_ + r[i,0] * np.kron(zi, xu)
            
        elif self.loss == 'user':
            if u_groups is None:
                raise RuntimeError('need groups for user loss function')
            if _sparse:
                self.A_ = self.lam / 2 * sp.sparse.eye(C*D)
                self.B_ = sp.sparse.csr_matrix((1, C*D))
            else:
                self.A_ = self.lam / 2 * np.eye(C*D)
                self.B_ = np.matrix(np.zeros((1, C*D)))
            ii = 0
            for u, rs in u_groups:
                if ii % 10000 == 0: print 'on user %d' % ii
                ii += 1
                if _sparse:
                    zbar = sp.sparse.csr_matrix(z[r[rs,2]].mean(axis=0))
                else:
                    zbar = z[r[rs,2]].mean(axis=0)
                xu = x[r[u,1],:]
                for i in rs:
                    zi = z[r[i,2],:]
                    if _sparse:
                        self.A_ = self.A_ + sp.sparse.kron(zi.T * (zi - zbar), xu.T * xu)
                        self.B_ = self.B_ + r[i,0] * sp.sparse.kron(zi - zbar, xu)
                    else:
                        self.A_ = self.A_ + np.kron(zi.T * (zi - zbar), xu.T * xu)
                        self.B_ = self.B_ + r[i,0] * np.kron(zi - zbar, xu)
        else:
            raise RuntimeError('only valid values of loss are "user" and "global"')
        
        self.A_ = self.A_ / r.shape[0]
        self.B_ = self.B_ / r.shape[0]
        if _sparse:
            self.w_ = sp.sparse.linalg.inv(self.A_) * self.B_.T
        else:
            self.w_ = np.linalg.inv(self.A_) * self.B_.T
        return self

    def predict(self, x, z, r):
        _sparse = sp.sparse.issparse(x) or sp.sparse.issparse(z)

        x = np.hstack([x, np.ones((x.shape[0], 1))])
        s = np.zeros(r.shape[0])
        for i in range(r.shape[0]):
            if _sparse:
                s[i] = (sp.sparse.kron(x[r[i,0]], z[r[i,1]]) * self.w_).todense()
            else:
                s[i] = np.kron(x[r[i,0]], z[r[i,1]]) * self.w_
        return s + self.r_mean_

if __name__ == '__main__':
    (bus_data, review_data, user_data, checkin_data) = eda.get_train_data()
    (bus_data_test, review_data_test, user_data_test, checkin_data_test) = eda.get_test_data()
    (bus_data_final, review_data_final, user_data_final, checkin_data_final) = eda.get_final_data()

    user_data_all = pd.concat([user_data, user_data_test, user_data_final]).groupby(level=0).last()
    bus_data_all = pd.concat([bus_data, bus_data_test, bus_data_final]).groupby(level=0).last()
    checkin_data_all = pd.concat([checkin_data, checkin_data_test, checkin_data_final]).groupby(level=0).last()
    
    review_data_mat = np.array(review_data.loc[:,['stars','user_id','business_id']])
    review_data_mat[:,1] = user_data_all.index.reindex(review_data_mat[:,1])[1]
    review_data_mat[:,2] = bus_data_all.index.reindex(review_data_mat[:,2])[1]
    review_data_mat = review_data_mat.astype(np.int64)

    review_data_mat_final = np.array(review_data_final.loc[:,['user_id','business_id']])
    review_data_mat_final[:,0] = user_data_all.index.reindex(review_data_mat_final[:,0])[1]
    review_data_mat_final[:,1] = bus_data_all.index.reindex(review_data_mat_final[:,1])[1]
    review_data_mat_final = review_data_mat_final.astype(np.int64)

    u_feat = u_features(user_data_all.fillna(0))
    b_feat = b_features(bus_data_all.fillna(0), checkin_data_all.fillna({})).todense()

    u_groups = [(user_data_all.index.get_loc(k), review_data.index.reindex(v)[1]) for (k, v) in review_data.groupby('user_id').groups.iteritems() if k in user_data_all.index]
    
    ppr = PairwisePreferenceRegression(lam = 1000., loss='user')
    ppr.fit((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat, u_groups)
    ppr_pred = ppr.predict((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat[:,[1,2]])
    mean_squared_error(review_data['stars'], np.minimum(5, np.maximum(0, ppr_pred)))
    pd.DataFrame({'review_id' : review_data.index, 'stars' : np.maximum(0, np.minimum(5, ppr_pred))}).to_csv('../ppr_fitted_user.csv', index=False)

    ppr_pred_final= ppr.predict((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat_final)
    pd.DataFrame({'review_id' : review_data_final.index,'stars':np.maximum(0,np.minimum(5,ppr_pred_final))}).to_csv('../ppr_submission_user.csv', index=False)

    ppr = PairwisePreferenceRegression(lam = 1000., loss='global')
    ppr.fit((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat, u_groups)
    ppr_pred = ppr.predict((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat[:,[1,2]])
    mean_squared_error(review_data['stars'], np.minimum(5, np.maximum(0, ppr_pred)))
    pd.DataFrame({'review_id' : review_data.index, 'stars' : np.maximum(0, np.minimum(5, ppr_pred))}).to_csv('../ppr_fitted_global.csv', index=False)

    ppr_pred_final= ppr.predict((u_feat - u_feat.mean(axis=0)) / u_feat.std(axis=0), (b_feat-b_feat.mean(axis=0)) / b_feat.std(axis=0), review_data_mat_final)
    pd.DataFrame({'review_id' : review_data_final.index,'stars':np.maximum(0,np.minimum(5,ppr_pred_final))}).to_csv('../ppr_submission_global.csv', index=False)
    
