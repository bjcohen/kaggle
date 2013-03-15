import pandas as pd
import numpy as np
from pymc import MCMC

def import_data():
    data_train = pd.read_csv('Train.csv')
    data_location = pd.read_csv('Location_Tree.csv')
    return (data_train, data_location)

def create_location_trie(data):
    root = {'_parent' : None}
    lookup = {}
    for _, row in data.iterrows():
        s = row[0]
        curr = root
        parent = None
        for l in s.split('~'):
            if l == parent:
                l = l + '~' + l
            if l not in curr:
                curr[l] = {'_parent' : parent}
                lookup[l] = curr[l]
            curr = curr[l]
            parent = l
    return (root, lookup)

def get_hierarchy(name, trie, trie_lookup):
    curr = name
    path = []
    while curr is not None:
        path.append(curr)
        curr = trie_lookup[curr]['_parent']
    
if __name__ == '__main__':
    (data_train, data_location) = import_data()
    (trie, trie_lookup) = create_location_trie(data_location)

    loc_table = {}
    for k in trie_lookup:
        loc_table[k] = []

    tr = {'Highlands' : 'Highland',
          'Shetlands' : 'Shetland',
          'Gatwick Airport' : 'London Gatwick Airport'}
    for i, row in data_train.iterrows():
        loc = row['LocationNormalized']
        if loc in tr:
            loc = tr[loc]
        if loc in loc_table:
            loc_table[loc].append(i)
        elif row['LocationRaw'] in loc_table:
            loc_table[row['LocationRaw']].append(i)
        else:
            print "bad location: %s" % loc

    prune_list = []
    for k in loc_table.iterkeys():
        if len(loc_table[k]) == 0:
            prune_list.append(k)

    for k in prune_list:
        del loc_table[k]
        
