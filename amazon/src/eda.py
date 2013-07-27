import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import Ward

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

    sgdc = SGDClassifier(loss='log', penalty='l1', n_iter=20, verbose=2)
    rfe = RFE(sgdc, n_features_to_select=50000 ,step=.1)   #approx 2mm starting out

    ACTION = np.array(train_data.ACTION)
    
    rfe.fit(model_mat_train, ACTION)

    sgdc.fit(model_mat_train[:27000,rfe.support_], ACTION[:27000])
    auc_score(train_data.ACTION[27000:], sgdc.predict_proba(model_mat_train[27000:,rfe.support_])[:,1])
