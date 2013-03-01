DATA = .
R = I:\R\R-2.15.2\bin\R.exe
python = python

.PHONY : corpus

corpus: documents.

documents.dat: # also vocab.dat
	python gen_corpus.py
