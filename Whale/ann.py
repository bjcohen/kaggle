import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import time
import os

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        '''
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)        
        '''
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        dtype = theano.config.floatX
        
        W_bound = np.sqrt(3./fan_in)
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=dtype)
        self.W = theano.shared(W_values, name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((filter_shape[0],), dtype=dtype), name='b', borrow=True)
        conv_out = conv.conv2d(input, self.W, filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """        
        dtype = theano.config.floatX
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=dtype), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=dtype), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        dtype = theano.config.floatX
        if W is None:
            W_bnd = np.sqrt(6. / (n_in + n_out))
            W_values = np.asarray(rng.uniform(low=-W_bnd, high=W_bnd, size=(n_in, n_out)), dtype=dtype)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

            if b is None:
                b_values = np.zeros((n_out,), dtype=dtype)
                b = theano.shared(value=b_values, name='b', borrow=True)

            self.W = W
            self.b = b

            lin_output = T.dot(input, self.W) + self.b
            self.output = (lin_output if activation is None else activation(lin_output))
            self.params = [self.W, self.b]
                
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output,
                                                     n_in=n_hiddden, n_out=n_out)
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = abs(self.hiddenLayer.W**2).sum() + abs(self.logRegressionLayer.W**2).sum()
        self.neg_log_likelihoot = self.logRegressionLayer.neg_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hidddenLayer.params + self.logRegressionLayer.params

class SGDTrain(object):
    def __init__(self, x, y, data, model, cost, errors, learning_rate=0.1,
                 L1_reg=0., L2_reg=0.0001, n_epochs=1000, batch_size=20, n_hidden=500):

        train_set_x, train_set_y = data[0]
        valid_set_x, valid_set_y = data[1]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar()
        
        # cost = cost + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

        validate_model = theano.function(inputs=[index], outputs=errors,
                                         givens={
                                             x: valid_set_x[index*batch_size:(index+1)*batch_size],
                                             y: valid_set_y[index*batch_size:(index+1)*batch_size]
                                             })

        params = [o for m in model for o in m.params]
        grads = T.grad(cost, params)
        updates = [(param, param-learning_rate * grad) for (param, grad) in zip(params, grads)]

        train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                      givens={
                                             x: train_set_x[index*batch_size:(index+1)*batch_size],
                                             y: train_set_y[index*batch_size:(index+1)*batch_size]
                                      })
        # start training model
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience / 2)

        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        start_time = time.clock()

        epoch = 0
        done_looping = False

        print 'starting training'
        
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                iter = epoch * n_train_batches + minibatch_index
                if iter % 100 == 0: print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)
                
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
    
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                if patience <= iter:
                        done_looping = True
                        break
                        
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i') %
               (best_validation_loss * 100., best_iter))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
