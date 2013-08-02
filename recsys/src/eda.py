import pandas as pd
import numpy as np

import json
import os
import operator

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
    review_data.set_index(['business_id', 'user_id'], inplace=True)
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
    review_data_test.set_index(['business_id', 'user_id'], inplace=True)
    user_data_test = read_streaming_json(os.path.join(test_directory, user_test_fn))
    user_data_test.set_index('user_id', inplace=True)
    checkin_data_test = read_streaming_json(os.path.join(test_directory, checkin_test_fn))
    checkin_data_test.set_index('business_id', inplace=True)

    preprocess(business_data_test, review_data_test, user_data_test, checkin_data_test)

    return (business_data_test, review_data_test, user_data_test, checkin_data_test)
    
def preprocess(business_data, review_data, user_data, checkin_data):
    del business_data['neighborhoods']

    vote_types = ['useful', 'funny', 'cool']

    if 'votes' in review_data and 'votes' in user_data:
        for v in vote_types:
            review_data['votes_' + v] = review_data['votes'].map(operator.itemgetter(v))
            user_data['votes_' + v] = user_data['votes'].map(operator.itemgetter(v))

        del review_data['votes']
        del user_data['votes']

    review_data['date'] = review_data['date'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

def train_model(model, train_data, cv=True):
    (business_data, review_data, user_data, checkin_data) = train_data

    ## build model matrix
    r_pp = review_data[['stars', 'date', 'votes_useful', 'votes_funny', 'votes_cool', 'text']] \
      .rename(columns={'votes_useful' : 'votes_useful_review', 'votes_funny' : 'votes_funny_review', 'votes_cool' : 'votes_cool_review'})
    b_pp = business_data[['latitude', 'longitude', 'stars', 'review_count', 'open', 'categories', 'city']] # name, full_address, state
    u_pp = user_data[['review_count', 'average_stars', 'votes_useful', 'votes_funny', 'votes_cool']] # name
    c_pp = checkin_data[['checkin_info']]

    ## TODO: fix this for multiindex
    model_mat = r_pp \
      .merge(b_pp, left_index=True, right_index=True, how='left', suffixes=('_rev', '_bus')) \
      .merge(u_pp, left_index=True, right_index=True, how='left', suffixes=('_bus', '_user')) \
      .merge(c_pp, left_index=True, right_index=True, how='left')
      
    model_mat['date'] = (datetime.datetime(2013, 1, 19) - model_mat['date']).map(operator.attrgetter("days"))

    model_mat['votes_useful_avg'] = model_mat['votes_useful'] / model_mat['review_count_user']
    model_mat['votes_funny_avg'] = model_mat['votes_funny'] / model_mat['review_count_user']
    model_mat['votes_cool_avg'] = model_mat['votes_cool'] / model_mat['review_count_user']

    ## cross-validation
    if cv:
        cv_result = cross_val_score(model, model_mat.drop('votes_useful', 1), model_mat['votes_useful'], cv=5, score_func=RMSLE)
    else:
        cv_result = None

    ## fit and return model
    scaler = StandardScaler()
    model_mat = scaler.fit_transform(model_mat)
    model.fit(model_mat.drop('votes_useful', 1), model_mat['votes_useful'])

    return (model, scaler, model.score(model_mat.drop('votes_useful', 1), model_mat['votes_useful']), cv_result)