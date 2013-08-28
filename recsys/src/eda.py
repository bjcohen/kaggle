import pandas as pd
import numpy as np

import json
import os
import operator
import datetime

import ngbr
import rbm

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
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
    review_data_final.set_index('review_id', inplace=True)
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
