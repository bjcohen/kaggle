import pandas as pd
from collections import Counter, OrderedDict
import sys
from string import maketrans, zfill

class AutoIncDict:
    def __init__(self):
        self.__d = OrderedDict()
    def __getitem__(self, key):
        if key not in self.__d:
            self.__d[key] = len(self.__d)
        return self.__d[key]
    def __getattr__(self, attr):
        try:
            return self.__d.__getattribute__(attr)
        except AttributeError:
            raise AttributeError("'AutoIncDict' object has no attribute '%s'" % attr)

def intersperse(iterable, sep):
    i = iter(iterable)
    yield i.next()
    for x in i:
        yield sep
        yield x

## TODO: stemming, filtering
def write_corpus(data, doc_visitor, vocab_visitor):
    vocab = AutoIncDict()
    punc = ",.!:;'\"/\\()*?-_%[]{}‘’"
    trans_mask = maketrans(punc, ''.zfill(len(punc)).translate(maketrans('0',' ')))
    skip = {'****k****k', '', '****k'}
    def proc_data(data):
        for (i, desc) in data:
            desc = desc.translate(trans_mask).lower().split(' ')
            ln = str(i)
            for word, cnt in Counter(desc).iteritems():
                word = word.strip()
                if word in skip: continue
                ln += " %i:%i" % (vocab[word], cnt)
            yield ln + '\n'
    doc_visitor(proc_data(data))
    vocab_visitor(intersperse(vocab.iterkeys(), '\n'))
    
if __name__ == '__main__':
    # location_tree = pd.read_csv('Location_Tree.csv')
    data_train = pd.read_csv('Train.csv')
    # data_valid = pd.read_csv('Valid.csv')
    with open('documents.dat', 'w+') as doc, open('vocab.dat', 'w+') as vocab:
        write_corpus(data_train['FullDescription'].iteritems(), doc.writelines, vocab.writelines)

