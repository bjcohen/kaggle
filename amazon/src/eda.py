import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier, RandomizedLogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, NuSVC

from operator import itemgetter
from itertools import imap, combinations
from subprocess import call
import os

def df_to_libsvm(df, target, filename, columns = None, colmap = None):
    if columns is None:
        columns = list(df.columns)
    if target in columns:
        columns.remove(target)
    if colmap is None:
        ncols = 0
        colmap = {}
        for c in columns:
            colmap.update({c + '_' + str(name) : index  for (index, name) in enumerate(set(df[c]), ncols)})
            ncols += len(set(df[c]))
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            f.write(str(row[target]))
            f.write(' ')
            f.write(' '.join(['%s:1' % colmap[col + '_' + str(value)] for col, value in row.iteritems() if col != target]))
            f.write('\n')
    return colmap

def sparse_to_libsvm(data, target, filename):
    with open(filename, 'w') as f:
        for i in xrange(data.shape[0]):
            s = [str(target[i])] + map(lambda x: str(x) + ':1', data[i].indices)
            f.write(' '.join(s) + '\n')

def write_grouping(df, target, filename = 'grouping.libfm'):
    with open(filename, 'w') as f:
        cols = list(df.columns)
        if target in cols:
            cols.remove(target)
        for i, col in enumerate(cols):
            for j in range(len(set(df[col]))): f.write('%d\n' % i)

def run_fest_test(festpath='/Users/bjcohen/dev/fest', **kwargs):
    '''
    -c <int>  : committee type:
                1 bagging
                2 boosting (default)
                3 random forest
    -d <int>  : maximum depth of the trees (default: 1000)
    -e        : report out of bag estimates (default: no)
    -n <float>: relative weight for the negative class (default: 1)
    -p <float>: parameter for random forests: (default: 1)
                (ratio of features considered over sqrt(features))
    -t <int>  : number of trees (default: 100)
    '''
    idstr = ''.join(map(lambda (f,v): f+str(v), kwargs.items()))
    ret = call([os.path.join(festpath, 'festlearn'),
                ' '.join(map(lambda (f,v): '-'+f+str(v), kwargs.items())),
                os.path.join('..', 'data', 'train_3way_-27000.libsvm'),
                os.path.join('..', 'data', 'fest_%s_-27000.model' % idstr)])
    if ret != 0: raise Exception()
    ret = call([os.path.join(festpath, 'festclassify'),
                os.path.join('..', 'data', 'train_3way_-27000.libsvm'),
                os.path.join('..', 'data', 'fest_%s_-27000.model' % idstr),
                os.path.join('..', 'data', 'pred_fest_train_-27000_%s' % idstr)])
    if ret != 0: raise Exception()
    ret = call([os.path.join(festpath, 'festclassify'),
                os.path.join('..', 'data', 'train_3way_27000-.libsvm'),
                os.path.join('..', 'data', 'fest_%s_-27000.model' % idstr),
                os.path.join('..', 'data', 'pred_fest_train_27000-_%s' % idstr)])
    if ret != 0: raise Exception()
    tr_score = auc_score(ACTION[:27000],
                         pd.read_table('../data/pred_fest_train_-27000_%s' % idstr, header=None))
    te_score = auc_score(ACTION[27000:],
                         pd.read_table('../data/pred_fest_train_27000-_%s' % idstr, header=None))
    return (tr_score, te_score)

if __name__ == '__main__':
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv', index_col='id')

    # compute 2- and 3-way combinations of features
    train_data_nway = train_data.drop('ACTION', 1)
    test_data_nway = test_data.copy()

    nway_dict = {}
    
    def nway_lookup(x):
        if x not in nway_dict:
            nway_dict[x] = len(nway_dict)
        return nway_dict[x]

    for nway in [2, 3]:
        for fs in combinations(test_data.columns, nway):
            train_data_nway['|'.join(fs)] = map(nway_lookup, [','.join(map(str, list(row))) for _, row in train_data[list(fs)].iterrows()])
            test_data_nway['|'.join(fs)] = map(nway_lookup, [','.join(map(str, list(row))) for _, row in test_data[list(fs)].iterrows()])

    # convert features from min(feature_code)..max(feature_code) to 0..num_feature_codes
    n_train = train_data_nway.shape[0]
    for col in train_data_nway.columns:
        factvals, _ = pd.factorize(np.hstack([train_data_nway[col], test_data_nway[col]]))
        train_data_nway[col], test_data_nway[col] = factvals[:n_train], factvals[n_train:]

    # do one-hot encoding on features
    ohe = OneHotEncoder()
    ohe.fit(np.vstack([train_data_nway, test_data_nway]))

    model_mat_train = ohe.transform(train_data_nway)
    model_mat_test = ohe.transform(test_data_nway)
    ACTION = np.array(train_data.ACTION)

    # find column indices of features that are in both sets and subset model matrices
    indices = np.concatenate([np.intersect1d(train_data_nway.ix[:,i], test_data_nway.ix[:,i]) + ohe.feature_indices_[i]
                              for i in range(train_data_nway.shape[1])])
    indices.sort()
    model_mat_train = model_mat_train[:,indices]
    model_mat_test = model_mat_test[:,indices]
    
    ### model training
    
    rfe = RFE(LogisticRegression(penalty='l2', dual=True, C=1., fit_intercept=True, #approx 2mm starting out
                                 intercept_scaling=10., class_weight='auto', verbose=1))

    ## logistic regression
    
    rfe.set_params(n_features_to_select=50000, step=.1)
    rfe.fit(model_mat_train[:27000], ACTION[:27000])

    lr = LogisticRegression(penalty='l2', dual=True, C=10.,
                            intercept_scaling=10., class_weight='auto')
    lr.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    pred = lr.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], pred[:,1])

    lr = LogisticRegression(penalty='l2', dual=True, C=10.,
                            intercept_scaling=10., class_weight='auto')
    lr.fit(model_mat_train[:27000], ACTION[:27000])
    pred = lr.predict_proba(model_mat_train[27000:])
    auc_score(ACTION[27000:], pred[:,1])
    
    lr.fit(model_mat_train[:, np.where(rfe.support_)[0]], ACTION)
    pred = lr.predict_proba(model_mat_test[:,np.where(rfe.support_)[0]])
    pd.DataFrame({'Id' : test_data.index, 'Action' : pred[:,1]}).to_csv('../lr2_submission.csv', header=True, index=False)

    ## svms
    svc = SVC(C=1., kernel='rbf', probability=True, class_weight='auto', verbose=2)
    svc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    pred = svc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], pred[:,1])

    nusvc = NuSVC(nu=.11, kernel='rbf', degree=3, probability=True, cache_size=1024, verbose=2)
    nusvc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    svc_pred = nusvc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], svc_pred[:,1])

    nusvc = NuSVC(nu=.11, kernel='rbf', degree=3, probability=True, cache_size=1024, verbose=2)
    nusvc.fit(model_mat_train[:27000], ACTION[:27000])
    svc_pred = nusvc.predict_proba(model_mat_train[27000:])
    auc_score(ACTION[27000:], svc_pred[:,1])

    nusvc.fit(model_mat_train[:, np.where(rfe.support_)[0]], ACTION)
    svc_pred = nusvc.predict_proba(model_mat_test[:,np.where(rfe.support_)[0]])
    pd.DataFrame({'Id' : test_data.index, 'Action' : svc_pred[:,1]}).to_csv('../nusvc_submission.csv', header=True, index=False)

    ## random forest
    
    rfe.set_params(n_features_to_select=10000, step=.1)
    rfe.fit(model_mat_train, ACTION)
    
    rfc = RandomForestClassifier(max_depth=100, verbose=2)
    rfc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]].todense(), ACTION[:27000])
    pred = rfc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]].todense())
    auc_score(ACTION[27000:], pred[:,1])

    gbc = GradientBoostingClassifier(loss='deviance', learning_rate=.1, n_estimators=100, max_depth=10, verbose=2)
    gbc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]].todense(), ACTION[:27000])
    pred = gbc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]].todense())
    auc_score(ACTION[27000:], pred[:,1])

    ## nearest-neighbors
    
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    knn_pred = knn.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], knn_pred[:,1])

    ##

    skb = SelectKBest(chi2, k=500000)
    model_mat_train_reduced = skb.fit_transform(model_mat_train, ACTION)
    model_mat_test_reduced = skb.transform(model_mat_test)

    lr = LogisticRegression(penalty='l2', dual=True, C=10., intercept_scaling=10., class_weight='auto')
    lr.fit(model_mat_train_reduced, ACTION)
    pred = lr.predict_proba(model_mat_test_reduced)
    pd.DataFrame({'Id' : test_data.index, 'Action' : pred[:,1]}).to_csv('../skb_lr_submission.csv', header=True, index=False)
