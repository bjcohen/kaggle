import numpy as np
import pandas as pd

import theano
import time
import logging
import os

from sklearn.preprocessing import OneHotEncoder
from theano import sparse, shared
from DBN import DBN

def train_dbn(datasets, finetune_lr=0.1, pretraining_epochs=100,
              pretrain_lr=0.01, k=1, training_epochs=1000,
              batch_size=10):
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    logging.info('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=16961,
              hidden_layers_sizes=[2000, 2000, 2000],
              n_outs=2)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    logging.info('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    logging.info('... pre-training the model')
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            logging.info('Pre-training layer %i, epoch %d, cost ' % (i, epoch))
            logging.info(np.mean(c))

    end_time = time.clock()
    logging.warn('The pretraining code for file ' +
                 os.path.split(__file__)[1] +
                 ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    logging.info('... getting the finetuning functions')
    train_fn, validate_model, test_model, validate_auc, test_auc = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    logging.info('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_auc = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                validation_auc_score = validate_auc()
                logging.info('epoch %i, minibatch %i/%i, validation error %f %%, validation auc %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., validation_auc_score * 100.))

                # if we got the best validation score until now
                if validation_auc_score > best_validation_auc:

                    #improve patience if loss improvement is good enough
                    if (validation_auc_score > best_validation_auc /
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_auc = validation_auc_score
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    test_auc_score = test_auc()
                    logging.info(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%, auc of best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100., test_auc_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    logging.info(('Optimization complete with best validation auc score of %f %%,'
           'with test auc %f %% (zero-one %f %%)') %
                 (best_validation_auc * 100., test_auc_score * 100., test_score * 100.))
    logging.warn('The fine tuning code for file ' +
                 os.path.split(__file__)[1] +
                 ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return dbn
    
if __name__ == '__main__':
    logging.basicConfig(level=0)
    
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv', index_col='id')

    ohe = OneHotEncoder()
    ohe.fit(np.vstack([train_data.drop('ACTION', 1), test_data]))

    train_model_mat = ohe.transform(train_data.drop('ACTION', 1))
    test_model_mat = ohe.transform(test_data)

    train_model_mat = train_model_mat.astype(theano.config.floatX)
    test_model_mat = test_model_mat.astype(theano.config.floatX)
    
    n_train = 22000
    n_valid = 5000
    n_test = 5769

    train_i = np.zeros(n_train)
    valid_i = np.zeros(n_valid) + 1
    test_i = np.zeros(n_test) + 2

    perm = np.random.permutation(np.hstack([train_i, valid_i, test_i]))

    train_set_x = sparse.shared(train_model_mat[np.where(perm == 0)[0]])
    train_set_y = shared(train_data.ACTION[perm == 0].astype('int32'))
    valid_set_x = sparse.shared(train_model_mat[np.where(perm == 1)[0]])
    valid_set_y = shared(train_data.ACTION[perm == 1].astype('int32'))
    test_set_x  = sparse.shared(train_model_mat[np.where(perm == 2)[0]])
    test_set_y  = shared(train_data.ACTION[perm == 2].astype('int32'))
    
    dbn = train_dbn([(train_set_x, train_set_y),
                     (valid_set_x, valid_set_y),
                     (test_set_x, test_set_y)],
                     batch_size = 10, pretraining_epochs = 100, training_epochs = 1000)

    pred_set_x = sparse.shared(test_model_mat)
    
    pred_proba, _ = dbn.build_prediction_functions(pred_set_x, batch_size = 100)

    pred_set_y = pred_proba()[:,1]
