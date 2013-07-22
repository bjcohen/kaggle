import pandas as pd
from collections import Counter, OrderedDict
import sys
from string import maketrans, zfill
from stemming.porter import stem
from nltk.corpus import stopwords

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

def write_corpus(data, doc_visitor, vocab_visitor):
    vocab = AutoIncDict()
    punc = ",.!:;'\"/\\()*?-_%[]{}‘’&"
    trans_mask = maketrans(punc, ''.zfill(len(punc)).translate(maketrans('0',' ')))
    skip = set(stopwords.words('english'))
    def proc_data(data):
        for (i, desc) in data:
            if pd.isnull(desc): desc = 'none'
            desc = desc.translate(trans_mask).lower().split(' ')
            ln_list = []
            for word, cnt in Counter(desc).iteritems():
                word = word.strip()
                word = stem(word)
                if word in skip: continue
                if word == '': continue
                ln_list.append("%i:%i" % (vocab[word], cnt))
            yield str(len(ln_list)) + ' ' + ' '.join(ln_list) + '\n'
    doc_visitor(proc_data(data))
    vocab_visitor(intersperse(vocab.iterkeys(), '\n'))
    
if __name__ == '__main__':
    # location_tree = pd.read_csv('Location_Tree.csv')
    data_train = pd.read_csv('Train.csv')
    with open('full_description', 'w+') as doc, open('full_description.vocab', 'w+') as vocab:
       write_corpus(data_train.FullDescription.iteritems(), doc.writelines, vocab.writelines)
    with open('titles', 'w+') as doc, open('titles.vocab', 'w+') as vocab:
        write_corpus(data_train.Title.iteritems(), doc.writelines, vocab.writelines)
