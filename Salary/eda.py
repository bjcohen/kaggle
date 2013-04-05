import pandas as pd
import numpy as np
import scipy as sp
import gensim
from gensim.corpora.bleicorpus import BleiCorpus
from gensim.models.ldamodel import LdaModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import logging

def import_data():
    data_train = pd.read_csv('Train.csv')
    fd_corpus = BleiCorpus('full_description', 'full_description.vocab')
    title_corpus = BleiCorpus('titles', 'titles.vocab')
    return (data_train, fd_corpus, title_corpus)

def generate_lda_features(fd_corpus, title_corpus):
    try:
        with open('fd_lda.model', 'r') as m: pass
        fd_lda = LdaModel.load('fd_lda.model')
    except IOError as e:
        fd_lda = LdaModel(corpus=fd_corpus, id2word=fd_corpus.id2word, num_topics=25, update_every=1, chunksize=10000, passes=1)
        fd_lda.save('fd_lda.model')
    
    (fd_docsums, _) = fd_lda.inference(fd_corpus)

    try:
        with open('title_lda.model', 'r') as m: pass
        title_lda = LdaModel.load('title_lda.model')
    except IOError as e:
        title_lda = LdaModel(corpus=title_corpus, id2word=title_corpus.id2word, num_topics=25, update_every=1, chunksize=10000, passes=1)
        title_lda.save('title_lda.model')

    (title_docsums, _) = title_lda.inference(title_corpus)

    return (fd_lda, fd_docsums, title_lda, title_docsums)
    
def generate_plain_features(data_train):
    cv = CountVectorizer(max_features=100)
    title_features = cv.fit_transform(data_train.Title.fillna(''))
    location_features = 
    return (title_features, location_features)

def construct_model_matrix(title_features, location_features, fd_docsums):
    fd_docsums_sparse = sp.sparse.coo_matrix(fd_docsums)
    model_matrix = sp.sparse.hstack([title_features, location_features, fd_docsums_sparse])
    model_matrix = model_matrix.tocsr()
    return model_matrix

def train_model(model_matrix, data_train):
    # RidgeRegression SVR(linear)
    # SVR(rbf) EnsembleRegressors
    # SGD Regressor
    X = normalize(model_matrix, axis=0)
    Y = data_train.SalaryNormalized

    sgdl = SGDRegressor(verbose=1)
    sgdl.fit(X, Y)
    predicted = sgdl.predict(X)
    
    classifier = RandomForestRegressor(n_estimators=50, verbose=2, n_jobs=-1, min_samples_split=30)
    classifier.fit(model_matrix, data_train.SalaryNormalized)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    (data_train, fd_corpus, title_corpus) = import_data()
    (fd_lda, fd_docsums, title_lda, title_docsums) = generate_lda_features(fd_corpus, title_corpus)
    title_features = generate_plain_features(data_train)
    model_matrix = construct_model_matrix(title_features, location_features, fd_docsums)

