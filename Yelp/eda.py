'''
business: type, business_id, name, [neighborhoods], full_address, city, state, latitude, longitude, stars, review_count, [categories], open
review: type, business_id, user_id, stars, text, date, {votes (useful, funny, cool)}
user: type, user_id, name, review_count, average_stars, {votes (useful, funny, cool)}
checkin: type, business_id, [checkin_info] (x-y# of checkins from x:00 to x+1:00 on y'th day, Sunday is 0)

Goal: predict # of useful votes a review will get

Loss Fct: RMSLE
Top Scores: 0.44, Benchmark: 0.644
'''

import pandas as pd
import numpy as np
import scipy as sp
import os.path
import itertools
import operator
import json
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
import datetime
from pattern import en
import nltk
import codecs
import re
from collections import defaultdict

class ArgumentException(Exception): pass

class HurdleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, selector = None, predictor = None):
        self.selector = selector
        self.predictor = predictor
    
    def fit(self, X, y):
        self.selector.fit(X, y > 0)
        self.predictor.fit(X, np.log(y + 1))
        return self

    def predict(self, X):
        sel = self.selector.predict(X)
        pred = np.zeros_like(sel, dtype=np.dtype('float'))
        pred[sel == 1] = np.exp(self.predictor.predict(X[sel == 1])) - 1
        return pred

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        '''Return observations that are nonzero'''
        return self.selector.predict(X)

class RoundDownTransformer(BaseEstimator):
    def __init__(self, trans):
        self.trans = trans
    def fit(self, X, y):
        self.trans.fit(X, y)
        return self
    def transform(self, X):
        return self.trans.transform(X)
    def fit_transform(self, X, y):
        return self.trans.fit_transform(X, y)
    def predict(self, X):
        pred = self.predict(X)
        pred[pred < 1] = 0
        return pred

class ColSelector(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_list = None, mats = None):
        if len(transformer_list) != len(mats):
            raise ArgumentException()
        self.transformer_list = transformer_list
        self.mats = mats

    def fit(self, X, y):
        for t, m in zip(self.transformer_list, self.mats):
            t.fit(m, y)

    def transform(self, X):
        return np.hstack([t.transform(m) for t, m in zip(self.transformer_list, self.mats)])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureUnion, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out

def read_streaming_json(fn):    
    with open(fn) as f:
        data = [json.loads(l) for l in f]
        df = pd.DataFrame(data)
    return df

def RMSLE(pred, act):
    return np.sqrt(np.mean((np.log(np.maximum(0, pred) + 1) - np.log(np.maximum(0, act) + 1)) ** 2))

def RMS(pred, act):
    return np.sqrt(np.mean((pred - act) ** 2))

def t1t2_stats(pred, act):
    n = pred.shape[0]
    corr = sum((pred > 0) == (act > 0))
    t1 = sum((pred > 0) & (act == 0))
    t2 = sum((pred == 0) & (act > 0))
    return (float(corr) / n, float(t1) / n, float(t2) / n)

def write_submission(fn, ids, votes):
    df = pd.DataFrame({'id' : ids, 'votes' : votes})
    df.to_csv(fn, sep=',', header=True, index=False)

def get_train_data():
    business_fn = 'yelp_training_set_business.json'
    review_fn = 'yelp_training_set_review.json'
    user_fn = 'yelp_training_set_user.json'
    checkin_fn = 'yelp_training_set_checkin.json'

    train_directory = 'yelp_training_set'

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

    test_directory = 'yelp_test_set'

    business_data_test = read_streaming_json(os.path.join(test_directory, business_test_fn))
    business_data_test.set_index('business_id', inplace=True)
    review_data_test = read_streaming_json(os.path.join(test_directory, review_test_fn))
    review_data_test.set_index('review_id', inplace=True)
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

def dummify(mat, cols):
    if isinstance(cols, basestring):
        cols = [cols]                     # pass in the a single column name
    for col in cols:
        if isinstance(col, tuple):
            colname, variates = col       # pass in a list of the columns to use
        else:
            colname = col
            if isinstance(mat[colname][0], basestring):          # each item is the base item
                variates = set(mat[colname])
            else:                                # each item is a list
                variates = set([c for cs in mat[colname] for c in list(cs)])
        for _v in variates:
            mat[colname + '_' + _v.replace(' ', '')] = mat[colname].map(lambda x: 1 if _v in x else 0)
        del mat[colname]

def train_model(model, train_data, cv=True):
    (business_data, review_data, user_data, checkin_data) = train_data

    ## build model matrix
    r_pp = review_data[['business_id', 'user_id', 'stars', 'date', 'votes_useful', 'text']].rename(columns={'votes_useful' : 'votes_useful_review'})
    b_pp = business_data[['latitude', 'longitude', 'stars', 'review_count', 'open', 'categories', 'city']] # name, full_address, state
    u_pp = user_data[['review_count', 'average_stars', 'votes_useful', 'votes_funny', 'votes_cool']] # name
    c_pp = checkin_data[['checkin_info']]

    model_mat = r_pp \
      .merge(b_pp, left_on='business_id', right_index=True, how='left', suffixes=('_rev', '_bus')) \
      .merge(u_pp, left_on='user_id', right_index=True, how='left', suffixes=('_bus', '_user')) \
      .merge(c_pp, left_on='business_id', right_index=True, how='left').drop(['business_id', 'user_id'], 1)
      
    model_mat['date'] = (datetime.datetime(2013, 1, 19) - model_mat['date']).map(operator.attrgetter("days"))

    def agg_by_day(d):
        aggregated = defaultdict(lambda: 0)
        for k in d:
            k1, k2 = k.split('-')
            aggregated['hour%s' % k1] += d[k]
            aggregated['day%s' % k2] += d[k]
        return aggregated
    model_mat['checkin_info'] = model_mat['checkin_info'].fillna({}).map(agg_by_day)
    for day in range(7): model_mat['checkin_info_day_%d' % day] = model_mat['checkin_info'].map(lambda x: x['day%d' % day])
    for hour in range(24): model_mat['checkin_info_hour_%d' % hour] = model_mat['checkin_info'].map(lambda x: x['hour%d' % hour])
            
    ## text
    sentiment = model_mat['text'].map(en.sentiment)
    model_mat['polarity'] = sentiment.map(operator.itemgetter(0))
    model_mat['subjectivity'] = sentiment.map(operator.itemgetter(1))
    model_mat['text_len'] = model_mat['text'].map(len)

    ## categories
    dummify(model_mat, [('categories', ['Art Supplies', 'Champagne Bars', 'Auto Glass Services', 'Adult', 'Lounges', 'Body Shops', 'African', 'Health Markets'])])
    dummify(model_mat, [('city', ['Phoenix', 'Tempe', 'Scottsdale', 'Mesa', 'Chandler', 'Gilbert', 'Glendale', 'Tolleson', 'Surprise', 'Peoria', 'Buckeye', 'Avondale', 'Goodyear', 'Carefree', 'Laveen', 'Guadalupe', 'Yuma', 'Ahwatukee'])])

    model_mat['stars_rev_polarity'] = abs(model_mat['stars_rev'] - 3)
    model_mat['text_ques'] = model_mat['text'].map(lambda x: x.count('?'))
    model_mat['text_excl'] = model_mat['text'].map(lambda x: x.count('!'))

    model_mat['text_complain'] = model_mat['text'].map(lambda x: len(re.findall('complain', x)))
    model_mat['text_recommend'] = model_mat['text'].map(lambda x: len(re.findall('recommend', x)))
    model_mat['text_suggest'] = model_mat['text'].map(lambda x: len(re.findall('suggest', x)))
    model_mat['text_enjoy'] = model_mat['text'].map(lambda x: len(re.findall('enjoy', x)))
    model_mat['text_love'] = model_mat['text'].map(lambda x: len(re.findall('love', x)))

    word_feat = json.load(file('words2'))
    for w in word_feat:
        model_mat['text_' + w] = model_mat['text'].map(lambda x: len(re.findall(w, x)))
    
    model_mat['text_paragraphs'] = model_mat['text'].map(lambda x: x.count('\n'))
    
    model_mat['votes_useful_avg'] = model_mat['votes_useful'] / model_mat['review_count_user']
    model_mat['votes_funny_avg'] = model_mat['votes_funny'] / model_mat['review_count_user']
    model_mat['votes_cool_avg'] = model_mat['votes_cool'] / model_mat['review_count_user']

    model_mat['user_anon'] = model_mat['votes_useful'].isnull()

    model_mat['review_count_user'].fillna(-1, inplace=True)
    model_mat['average_stars'].fillna(-1, inplace=True)
    model_mat['votes_useful'].fillna(-1, inplace=True)
    model_mat['votes_funny'].fillna(-1, inplace=True)
    model_mat['votes_cool'].fillna(-1, inplace=True)
    model_mat['votes_useful_avg'].fillna(-1, inplace=True)
    model_mat['votes_funny_avg'].fillna(-1, inplace=True)
    model_mat['votes_cool_avg'].fillna(-1, inplace=True)

    pos_text = pd.read_csv('pos_text.csv', index_col='review_id')
    model_mat['pos_nnp'] = pos_text.text.fillna('').map(lambda x: len(re.findall('NNP', x)))
    
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

def predict_model(model, train_data, test_data):
    (business_data, review_data, user_data, checkin_data) = train_data
    (business_data_test, review_data_test, user_data_test, checkin_data_test) = test_data

    b_pp = business_data[['latitude', 'longitude', 'stars', 'review_count', 'open', 'categories', 'city']]
    u_pp = user_data[['review_count', 'average_stars', 'votes_useful', 'votes_funny', 'votes_cool']]
    c_pp = checkin_data[['checkin_info']]
    
    r_t = review_data_test[['business_id', 'user_id', 'stars', 'date', 'text']]
    b_t = business_data_test[['latitude', 'longitude', 'stars', 'review_count', 'open', 'categories', 'city']]
    u_t = user_data_test[['review_count', 'average_stars']]
    c_t = checkin_data_test[['checkin_info']]

    b_all = pd.concat([b_pp, b_t])[b_pp.columns]
    u_all = pd.concat([u_pp, u_t])[u_pp.columns]
    c_all = pd.concat([c_pp, c_t])[c_pp.columns]

    mm_t = r_t \
        .merge(b_all, left_on='business_id', right_index=True, how='left', suffixes=('_rev', '_bus')) \
        .merge(u_all, left_on='user_id', right_index=True, how='left', suffixes=('_bus', '_user')) \
        .merge(c_all, left_on='business_id', right_index=True, how='left').drop(['business_id', 'user_id'], 1)
        
    mm_t['date'] = (datetime.datetime(2013, 3, 12) - mm_t['date']).map(operator.attrgetter("days"))
    def agg_by_day(d):
        aggregated = defaultdict(lambda: 0)
        for k in d:
            k1, k2 = k.split('-')
            aggregated['hour%s' % k1] += d[k]
            aggregated['day%s' % k2] += d[k]
        return aggregated
    mm_t['checkin_info'] = mm_t['checkin_info'].fillna({}).map(agg_by_day)
    for day in range(7): mm_t['checkin_info_day_%d' % day] = mm_t['checkin_info'].map(lambda x: x['day%d' % day])
    for hour in range(24): mm_t['checkin_info_hour_%d' % hour] = mm_t['checkin_info'].map(lambda x: x['hour%d' % hour])

    sentiment = mm_t['text'].map(en.sentiment)
    mm_t['polarity'] = sentiment.map(operator.itemgetter(0))
    mm_t['subjectivity'] = sentiment.map(operator.itemgetter(1))
    mm_t['text_len'] = mm_t['text'].map(len)

    dummify(mm_t, [('categories', ['Art Supplies', 'Champagne Bars', 'Auto Glass Services', 'Adult', 'Lounges', 'Body Shops', 'African', 'Health Markets'])])
    dummify(mm_t, [('city', ['Phoenix', 'Tempe', 'Scottsdale', 'Mesa', 'Chandler', 'Gilbert', 'Glendale', 'Tolleson', 'Surprise', 'Peoria', 'Buckeye', 'Avondale', 'Goodyear', 'Carefree', 'Laveen', 'Guadalupe', 'Yuma', 'Ahwatukee'])])

    mm_t['stars_rev_polarity'] = abs(mm_t['stars_rev'] - 3)
    mm_t['text_ques'] = mm_t['text'].map(lambda x: x.count('?'))
    mm_t['text_excl'] = mm_t['text'].map(lambda x: x.count('!'))

    mm_t['text_complain'] = mm_t['text'].map(lambda x: len(re.findall('complain', x)))
    mm_t['text_recommend'] = mm_t['text'].map(lambda x: len(re.findall('recommend', x)))
    mm_t['text_suggest'] = mm_t['text'].map(lambda x: len(re.findall('suggest', x)))
    mm_t['text_enjoy'] = mm_t['text'].map(lambda x: len(re.findall('enjoy', x)))
    mm_t['text_love'] = mm_t['text'].map(lambda x: len(re.findall('love', x)))

    word_feat = json.load(file('words2'))
    for w in word_feat:
        mm_t['text_' + w] = mm_t['text'].map(lambda x: len(re.findall(w, x)))

    mm_t['text_paragraphs'] = mm_t['text'].map(lambda x: x.count('\n'))
    
    mm_t['votes_useful_avg'] = mm_t['votes_useful'] / mm_t['review_count_user']
    mm_t['votes_funny_avg'] = mm_t['votes_funny'] / mm_t['review_count_user']
    mm_t['votes_cool_avg'] = mm_t['votes_cool'] / mm_t['review_count_user']

    mm_t['user_anon'] = mm_t['votes_useful'].isnull()

    mm_t['review_count_user'].fillna(-1, inplace=True)
    mm_t['average_stars'].fillna(-1, inplace=True)
    mm_t['votes_useful'].fillna(-1, inplace=True)
    mm_t['votes_funny'].fillna(-1, inplace=True)
    mm_t['votes_cool'].fillna(-1, inplace=True)
    mm_t['votes_useful_avg'].fillna(-1, inplace=True)
    mm_t['votes_funny_avg'].fillna(-1, inplace=True)
    mm_t['votes_cool_avg'].fillna(-1, inplace=True)

    pos_text = pd.read_csv('pos_text_test.csv', index_col='review_id')
    mm_t['pos_nnp'] = pos_text.text.fillna('').map(lambda x: len(re.findall('NNP', x)))
    
    return model.predict(mm_t.drop('review_id', 1))

def output_blei_corpus(df, fn):
    vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
    features = vectorizer.fit_transform(df['text']).tocsr()
    with codecs.open(fn + '.documents', 'w', encoding='utf-8') as f:
        for i in range(features.shape[0]):
            n = features[i].indices.shape[0]
            f.write(str(n))
            for j in range(n):
                f.write(u" %d:%d" % (j, features[i,features[i].indices[j]]))
            f.write('\n')
    with codecs.open(fn + '.vocab', 'w', encoding='utf-8') as f:
        for w in vectorizer.vocabulary_:
            f.write(w + '\n')
        

if __name__ == '__main__':
    pass
    ## train_data = get_train_data()
    ## test_data = get_test_data()
    ## model = RandomForestRegressor(compute_importances=True)
    ## model = train_model(model, train_data, False)
    ## predicted = predict_model(model, train_data, test_data)
    ## write_submission("rf_submission.csv", test_data[1]['review_id'], predicted)

    ## text_cols_reg = ColSelector([text_clf, RandomForestRegressor(compute_importances=True)], [review_data['text'], model_mat.drop('votes_useful', 1)])

    text_clf = Pipeline([('features', FeatureUnion([('word_features', TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))),
                                                    ('pos_features', CountVectorizer(preprocessor=lambda x: ' '.join(map(operator.itemgetter(1), nltk.tag.pos_tag(nltk.tokenize.wordpunct_tokenize(x)))), ngram_range=(1, 2)))])),
                         ('scaler', StandardScaler(with_mean=False)),
                         ('clf', SGDClassifier())])
    
    ## text_clf = Pipeline([('tfidf', TfidfVectorizer(max_df=.5, ngram_range=(1,2), sublinear_tf=True, use_idf=True), ('clf', MultinomialNB(alpha=1e-3))])
    ## gs_clf = GridSearchCV(text_clf, {'tfidf__ngram_range': [(1, 2)], 'tfidf__use_idf': [True], 'tfidf__sublinear_tf': [True, False], 'tfidf__max_df': [.3, .4, .5], 'clf__alpha': [1e-3]}, verbose=True)
    ## gs_clf.fit(review_data.text, review_data.votes_useful > 0)
    
    ## zip(pca.explained_variance_, model_mat.columns[np.fliplr(np.argsort(np.abs(pca.components_)))], np.fliplr(np.sort(np.abs(pca.components_))))

    # pos_text = review_data['text'].map(lambda x: ' '.join(map(operator.itemgetter(1), nltk.tag.pos_tag(nltk.tokenize.wordpunct_tokenize(x)))))
    # tfidf_vec = TfidfVectorizer(ngram_range = (1, 2))
    # pos_feat = tfidf_vec.fit_transform(pos_text)
    # tfidf_stopwords_vec = TfidfVectorizer(stop_words = nltk.corpus.stopwords.words('english'), max_features = 20000)
    # text_feat = tfidf_stopwords_vec.fit_transform(review_data['text'])
    
    # pca_rfr = Pipeline([('pca', RandomizedPCA()), ('rfr', RandomForestRegressor())])

    # pca_rfr_cv = GridSearchCV(pca_rfr, {'pca__n_components' : [50, 100, 200], 'pca__whiten' : [True, False], 'rfr__n_estimators' : [10, 100], 'rfr__min_samples_split' : [2, 5, 10]}, loss_func=RMSLE)

    # pca_rfr_cv.fit(pos_feat, review_data['votes_useful'])

    # gbr.fit(model_mat.drop(['votes_useful_review', 'text', 'checkin_info'], 1).fillna(0), model_mat.votes_useful_review)
    # gbr_pred = gbr.predict(mm_t.drop(['text', 'checkin_info'], 1).fillna(0))
    gbr = GradientBoostingRegressor(alpha = 0.7, loss = 'ls', max_depth = 11, verbose = 2)
