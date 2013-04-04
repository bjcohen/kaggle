import os
import pandas as pd
from dateutil.parser import parse

## TODO: Machine_Appendix.csv?

def get_paths():
    data_path = os.path.join(os.curdir, 'data')
    submission_path = os.path.join(os.curdir, 'submissions')
    return data_path, submission_path

def get_train_df():
    data_path, _ = get_paths()
    train = pd.read_csv(os.path.join(data_path, 'Train.csv'), converters={'saledate' : parse})
    return train

def get_test_df(test_set=False):
    data_path, _ = get_paths()
    if test_set:
        test_path = 'Test.csv'
    else:
        test_path = 'Valid.csv'
    test = pd.read_csv(os.path.join(data_path, test_path), converters={'saledate' : parse})
    return test

def get_train_test_df(test_set=False):
    return get_train_df(), get_test_df(test_set) # change to get_test_df once final test set released

def write_submission(name, predictions):
    _, submission_path = get_paths()
    test = get_test_df()
    test = test.join(pd.DataFrame({'SalePrice' : predictions}))
    test[['SalesID', 'SalePrice']].to_csv(os.path.join(submission_path, name), index=False)
