import util as u
import pandas as pd
import operator
import numpy as np
from sklearn import ensemble, cross_validation, metrics
import re

## todo: gradient boosted model?

if __name__ == '__main__':
    train, test = u.get_train_test_df(test_set=False)

    ## define columns to use for features
    ## not used: SalesID, MachineID, ModelID, Saledate, fiModelDesc
    ## ProductClassDesc, ProductGroupDesk
    ## fiBaseModel, fiSecondaryDesc, fiModelSeries, fiModelDescriptor
    categorical = {'datasource', 'auctioneerID', 'UsageBand', 'ProductSize', 'ProductGroup', 'Drive_System',
                   'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
                   'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier',
                   'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type',
                   'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
                   'Differential_Type', 'Steering_Controls'}
    numerical = {'YearMade', 'MachineHoursCurrentMeter', 'Undercarriage_Pad_Width', 'Stick_Length'}
    

    
    def get_date_dataframe(date_column):
        '''Create a data frame with columns for each date constituent'''
        return pd.DataFrame({
            'SaleYear' : [d.year for d in date_column],
            'SaleMonth' : [d.month for d in date_column],
            'SaleDay' : [d.day for d in date_column]
        }, index=date_column.index)
    
    ## for col in columns:
    ##     if train[col].dtype == np.dtype('object'):
    ##         s = np.unique(train[col].fillna(-1).values)
    ##         mapping = pd.Series([x[0] for x in enumerate(s)], index=s)
    ##         train_feat = train_feat.join(train[col].map(mapping).fillna(-1))
    ##         test_feat = test_feat.join(test[col].map(mapping).fillna(-1))
    ##     else:
    ##         train_feat = train_feat.join(train[col].fillna(0))
    ##         test_feat = test_feat.join(test[col].fillna(0))

    def stick_length_converter(s):
        '''parse Stick_Length column'''
        if s == 'None or Unspecified' or (not isinstance(s, str) and np.isnan(s)): return None
        match = re.match('(\d+)\' (\d+)', s)
        return float(match.group(1)) + float(match.group(2)) / 12.

    def undercarriage_pad_width_converter(s):
        '''Parse Undercarriage_Pad_Width column'''
        if s == 'None or Unspecified' or (not isinstance(s, str) and np.isnan(s)): return None
        match = re.match('([\d\.]+) inch', s)
        return float(match.group(1))
        
    train['Stick_Length'] = train['Stick_Length'].apply(stick_length_converter)
    test['Stick_Length'] = test['Stick_Length'].apply(stick_length_converter)    

    train['Undercarriage_Pad_Width'] = train['Undercarriage_Pad_Width'].apply(undercarriage_pad_width_converter)
    test['Undercarriage_Pad_Width'] = test['Undercarriage_Pad_Width'].apply(undercarriage_pad_width_converter)    

    ## start features with date features
    train_feat = get_date_dataframe(train['saledate'])
    test_feat = get_date_dataframe(test['saledate'])

    ## add features using the types defined above
    for col in categorical:
        v = train[col].fillna(-1).unique()
        colnames = [col + str(n) for n in v if n != -1]
        train_feat = train_feat.join(pd.DataFrame(dict(zip(colnames, [train[col].map(lambda x: 1 if x==s else 0) for s in v if s != -1]))))
        test_feat = test_feat.join(pd.DataFrame(dict(zip(colnames, [train[col].map(lambda x: 1 if x==s else 0) for s in v if s != -1]))))            

    for col in numerical:
        train_feat = train_feat.join(train[col].fillna(0))
        test_feat = test_feat.join(test[col].fillna(0))

    ## train random forest
    rf = ensemble.RandomForestRegressor(n_estimators=50, n_jobs=2, compute_importances=True)
    # rf.fit(train_feat, train['SalePrice'])
    # predictions = rf.predict(test_feat)
    # imp = sorted(zip(train_feat.columns, rf.feature_importances_), key=operator.itemgetter(1), reverse=True)
    # for f in imp:
    #    print f

    ## train gradient boosted forest
    gb = ensemble.GradientBoostingRegressor(n_estimators=100)

    ## compute CV scores
    def rmsle(x1, x2):
       return np.sqrt(np.mean(np.power(np.log(np.array(x1)+1) - np.log(np.array(x2)+1), 2)))

    ## rmsle_scorer = metrics.Scorer(rmsle, greater_is_better=False)
    
    rf_scores = cross_validation.cross_val_score(rf, train_feat, train['SalePrice'], score_func=rmsle)
    gb_scores = cross_validation.cross_val_score(gb, train_feat, train['SalePrice'], score_func=rmsle)

    ## write submission: .25 = ok, .21 = winning
    # u.write_submission('my_random_forest_benchmark.csv', predictions)
