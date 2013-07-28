import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier, RandomizedLogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import Ward
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, NuSVC

from operator import itemgetter
from itertools import imap, combinations

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

def write_grouping(df, target, filename = 'grouping.libfm'):
    with open(filename, 'w') as f:
        cols = list(df.columns)
        if target in cols:
            cols.remove(target)
        for i, col in enumerate(cols):
            for j in range(len(set(df[col]))): f.write('%d\n' % i)

if __name__ == '__main__':
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv', index_col='id')

    train_data_nway = train_data.drop('ACTION', 1)
    test_data_nway = test_data.copy()

    nway_dict = {}
    
    def nway_lookup(x):
        if x not in nway_dict:
            nway_dict[x] = len(nway_dict)
        return nway_dict[x]
    
    for nway in [2, 3]:
        for fs in combinations(test_data.columns, nway):
            train_data_nway[','.join(fs)] = map(nway_lookup, [','.join(map(str, list(row))) for _, row in train_data[list(fs)].iterrows()])
            test_data_nway[','.join(fs)] = map(nway_lookup, [','.join(map(str, list(row))) for _, row in test_data[list(fs)].iterrows()])
            
    ohe = OneHotEncoder()
    ohe.fit(np.vstack([train_data_nway, test_data_nway]))

    model_mat_train = ohe.transform(train_data_nway)
    model_mat_test = ohe.transform(test_data_nway)
    ACTION = np.array(train_data.ACTION)

    ### model training
    
    rfe = RFE(LogisticRegression(penalty='l2', dual=True, C=1., fit_intercept=True, #approx 2mm starting out
                                 intercept_scaling=10., class_weight='auto'))

    ## logistic regression
    
    rfe.set_params(n_features_to_select=50000, step=.1)
    rfe.fit(model_mat_train, ACTION)

    lr = LogisticRegression(penalty='l2', dual=True, C=2.,
                            intercept_scaling=50., class_weight='auto')
    lr.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    pred = lr.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], pred[:,1])

    lr.fit(model_mat_train[:, np.where(rfe.support_)[0]], ACTION)
    pred = lr.predict_proba(model_mat_test[:,np.where(rfe.support_)[0]])
    pd.DataFrame({'Id' : test_data.index, 'Action' : pred[:,1]}).to_csv('../lr_submission.csv', header=True, index=False)

    ## svms
    svc = SVC(C=1., kernel='rbf', probability=True, class_weight='auto', verbose=2)
    svc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    pred = svc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], pred[:,1])

    nusvc = NuSVC(nu=.5, kernel='rbf', probability=True, cache_size=1024, class_weight='auto', verbose=2)
    nusvc.fit(model_mat_train[:27000, np.where(rfe.support_)[0]], ACTION[:27000])
    pred = nusvc.predict_proba(model_mat_train[27000:,np.where(rfe.support_)[0]])
    auc_score(ACTION[27000:], pred[:,1])

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
    
