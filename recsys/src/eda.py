import pandas as pd
import numpy as np

import json
import os
import operator
import datetime

import ngbr
import rbm

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_squared_error

def read_streaming_json(fn):    
    with open(fn) as f:
        data = [json.loads(l) for l in f]
        df = pd.DataFrame(data)
    return df

def write_submission(fn, ids, votes):
    df = pd.DataFrame({'id' : ids, 'votes' : votes})
    df.to_csv(fn, sep=',', header=True, index=False)

def get_train_data():
    business_fn = 'yelp_training_set_business.json'
    review_fn = 'yelp_training_set_review.json'
    user_fn = 'yelp_training_set_user.json'
    checkin_fn = 'yelp_training_set_checkin.json'

    train_directory = os.path.join('..', 'data')

    business_data = read_streaming_json(os.path.join(train_directory, business_fn))
    business_data.set_index('business_id', inplace=True)
    review_data = read_streaming_json(os.path.join(train_directory, review_fn))
    review_data.set_index('review_id', inplace=True)
    user_data = read_streaming_json(os.path.join(train_directory, user_fn))
    user_data.set_index('user_id', inplace=True)
    checkin_data = read_streaming_json(os.path.join(train_directory, checkin_fn))
    checkin_data.set_index('business_id', inplace=True)

    preprocess(business_data, review_data, user_data, checkin_data)

    return (business_data, review_data, user_data, checkin_data)
    
def get_test_data():
    business_test_fn = 'yelp_test_set_business.json'
    review_test_fn = 'yelp_test_set_review.json'
    user_test_fn = 'yelp_test_set_user.json'
    checkin_test_fn = 'yelp_test_set_checkin.json'

    test_directory = os.path.join('..', 'data')

    business_data_test = read_streaming_json(os.path.join(test_directory, business_test_fn))
    business_data_test.set_index('business_id', inplace=True)
    review_data_test = read_streaming_json(os.path.join(test_directory, review_test_fn))
    user_data_test = read_streaming_json(os.path.join(test_directory, user_test_fn))
    user_data_test.set_index('user_id', inplace=True)
    checkin_data_test = read_streaming_json(os.path.join(test_directory, checkin_test_fn))
    checkin_data_test.set_index('business_id', inplace=True)

    preprocess(business_data_test, review_data_test, user_data_test, checkin_data_test)

    return (business_data_test, review_data_test, user_data_test, checkin_data_test)
    
def get_final_data():
    business_final_fn = 'final_test_set_business.json'
    review_final_fn = 'final_test_set_review.json'
    user_final_fn = 'final_test_set_user.json'
    checkin_final_fn = 'final_test_set_checkin.json'

    final_directory = os.path.join('..', 'data')

    business_data_final = read_streaming_json(os.path.join(final_directory, business_final_fn))
    business_data_final.set_index('business_id', inplace=True)
    review_data_final = read_streaming_json(os.path.join(final_directory, review_final_fn))
    user_data_final = read_streaming_json(os.path.join(final_directory, user_final_fn))
    user_data_final.set_index('user_id', inplace=True)
    checkin_data_final = read_streaming_json(os.path.join(final_directory, checkin_final_fn))
    checkin_data_final.set_index('business_id', inplace=True)

    preprocess(business_data_final, review_data_final, user_data_final, checkin_data_final)

    return (business_data_final, review_data_final, user_data_final, checkin_data_final)

def preprocess(business_data, review_data, user_data, checkin_data):
    del business_data['neighborhoods']

    vote_types = ['useful', 'funny', 'cool']

    if 'votes' in review_data and 'votes' in user_data:
        for v in vote_types:
            review_data['votes_' + v] = review_data['votes'].map(operator.itemgetter(v))
            user_data['votes_' + v] = user_data['votes'].map(operator.itemgetter(v))

        del review_data['votes']
        del user_data['votes']

    if 'date' in review_data:
        review_data['date'] = review_data['date'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    del business_data['type']
    del review_data['type']
    del user_data['type']
    del checkin_data['type']

if __name__ == '__main__':
    (bus_data, review_data, user_data, checkin_data) = get_train_data()
    (bus_data_test, review_data_test, user_data_test, checkin_data_test) = get_test_data()
    (bus_data_final, review_data_final, user_data_final, checkin_data_final) = get_final_data()

    review_data['age'] = review_data['date'].map(lambda x: (datetime.datetime(2013, 1, 19) - x).days)

    # business: full_address, lat, lng, name, state
    # user: name
    # model_mat = review_data[['business_id', 'user_id', 'stars', 'date', 'votes_useful', 'votes_funny', 'votes_cool', 'text']] \
    #   .rename(columns={'stars' : 'stars_review',
    #                    'votes_useful' : 'votes_useful_review',
    #                    'votes_funny' : 'votes_funny_review',
    #                    'votes_cool' : 'votes_cool_review'}) \
    #   .join(bus_data[['stars', 'review_count', 'open', 'categories', 'city']].rename(columns={'review_count' : 'review_count_bus'}), on='business_id') \
    #   .join(user_data[['review_count', 'average_stars', 'votes_useful', 'votes_funny', 'votes_cool']], on='user_id') \
    #   .join(checkin_data[['checkin_info']], on='business_id')

    # review: stars, date, votes, text
    # business: stars
    # user: average_stars, votes
    # model_mat_test = review_data_test \
    #   .join(bus_data_test[['review_count', 'open', 'categories', 'city']].rename(columns={'review_count' : 'review_count_bus'}), on='business_id') \
    #   .join(user_data_test[['review_count']], on='user_id') \
    #   .join(checkin_data_test[['checkin_info']], on='business_id')
      
    # model_mat['date'] = datetime.datetime(2013, 1, 19) - model_mat['date']

    # model_mat['votes_useful_avg'] = model_mat['votes_useful'] / model_mat['review_count_user']
    # model_mat['votes_funny_avg'] = model_mat['votes_funny'] / model_mat['review_count_user']
    # model_mat['votes_cool_avg'] = model_mat['votes_cool'] / model_mat['review_count_user']

    ## cross-validation
    # cv_result = cross_val_score(model, model_mat.drop('stars', 1),
    #                             model_mat['stars'], score_func=mean_squared_error)

    ## fit and return model
    # scaler = StandardScaler()
    # model_mat = scaler.fit_transform(model_mat)
    # model.fit(model_mat.drop('stars', 1), model_mat['stars'])

    # reload(ngbr)
    # kn = ngbr.KorenNgbr()
    # kn.fit((pd.concat([bus_data, bus_data_test]), review_data, pd.concat([user_data, user_data_test]),
    #         pd.concat([checkin_data, checkin_data_test])))

    # reload(ngbr)
    # ki = ngbr.KorenIntegrated(n_iter=1, model_type='integrated')
    # ki.fit((pd.concat([bus_data, bus_data_test]), review_data.iloc[:100], review_data.iloc[:200], pd.concat([user_data, user_data_test]),
    #         pd.concat([checkin_data, checkin_data_test])))
    # ki_pred = ki.predict(review_data.iloc[100:200])
    # np.sqrt(mean_squared_error(review_data.ix[100:200,'stars'], ki_pred))

    # reload(rbm)
    # r = rbm.RBM(epochs=50, conditional=True, rating_levels=pd.Index([1,2,3,4,5]), learning_rate=0.01, lam=0.001, n_hidden=200, momentum=0.9)
    # r.fit(bus_data.iloc[range(500) + [7637, 3038]], review_data.iloc[:2], review_data.iloc[:2])
    # r_pred_proba = r.predict(review_data.iloc[:2], method='prob')

    reload(rbm)
    r = rbm.RBM(epochs=50, conditional=True, rating_levels=pd.Index([1,2,3,4,5]), learning_rate=0.01, lam=0.001, n_hidden=100, momentum=0.9)
    r.fit(pd.concat([bus_data, bus_data_test, bus_data_final]),
          review_data,
          pd.concat([review_data, review_data_test, review_data_final]))
    r_pred = r.predict(review_data_final, method='exp')
    pd.DataFrame({'review_id' : review_data_final.index, 'stars' : np.maximum(0, np.minimum(5, r_pred))}).to_csv('../rbm_submission_e50_h100.csv',index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='integrated', gam3=0.002)
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test, review_data_final]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_test)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../integrated_submission_i10_f100.csv', index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='svd++')
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test, review_data_final]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_final)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../svdpp_submission_i10_f100.csv', index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='factor')
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test, review_data_final]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_final)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../factor_submission_i10_f100.csv', index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='nsvd')
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_final)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../nsvd_submission_i10_f100.csv', index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='neighbors')
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test, review_data_final]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_final)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../neighbors_submission_i10.csv', index=False)

    reload(ngbr)
    ki = ngbr.KorenIntegrated(n_iter=10, n_factors=100, model_type='timeneighbors')
    ki.fit((pd.concat([bus_data, bus_data_test, bus_data_final]),
            review_data,
            pd.concat([review_data, review_data_test, review_data_final]),
            pd.concat([user_data, user_data_test]),
            pd.concat([checkin_data, checkin_data_test, checkin_data_final])))
    ki_pred = ki.predict(review_data_final)
    pd.DataFrame({'review_id' : ki_pred.index, 'stars' : np.maximum(0, np.minimum(5, ki_pred))}).to_csv('../timeneighbors_submission_i10.csv', index=False)
    
    # reload(nnmf)
    # mf = nnmf.NNMF(item_sim='none', estimation_method='simu')
    # mf.fit(user_data, bus_data, review_data.iloc[:100])
    # mf_pred = mf.predict(review_data.iloc[:100])
    
    ## TODO: use IdLookupTable.csv to verify order
