import numpy as np
import logging


def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(x): return x

def bernoulli(p):
    return np.random.rand(*p.shape) < p

class RBM(object):
    def __init__(self, num_visible, num_hidden, binary = True, scale = 0.001):
        self.weights = scale * np.random.randn(num_hidden, num_visible)
        self.h_bias = 2 * scale * np.random.randn(num_hidden)
        self.v_bias = scale * np.random.randn(num_visible)
        
        self._visible = binary and sigmoid or identity

    @property
    def num_hidden(self):
        return len(self.h_bias)

    @property
    def num_visible(self):
        return len(self.v_bias)

    def hidden_expectation(self, visible, bias = 0.):
        return sigmoid(np.dot(self.weights, visible.T).T + self.h_bias + bias)

    def visible_expectation(self, hidden, bias = 0.):
        return self._visible(np.dot(hidden, self.weights) + self.v_bias + bias)

    def iter_passes(self, visible):
        while True:
            hidden = self.hidden_expectation(visible)
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden))

    def reconstruct(self, visible, passes=1):
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i+1 == passes:
                return visible

class RBMTrainer(object):
    def __init__(self, rbm, momentum=0., l2=0., target_sparsity=None):
        self.rbm = rbm
        self.momentum = momentum
        self.l2 = l2
        self.target_sparsity = target_sparsity

        self.grad_weights = np.zeros(rbm.weights.shape, float)
        self.grad_vis = np.zeros(rbm.v_bias.shape, float)
        self.grad_hid = np.zeros(rbm.h_bias.shape, float)

    def learn(self, visible, learning_rate=0.2):
        gradients = self.calculate_gradients(visible)
        self.apply_gradients(*gradients, learning_rate=learning_rate)

    def calculate_gradients(self, visible_batch):
        passes = self.rbm.iter_passes(visible_batch)
        v0, h0 = passes.next()
        v1, h1 = passes.next()

        gw = (np.dot(h0.T, v0) - np.dot(h1.T, v1)) / len(visible_batch)
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)
        if self.target_sparsity is not None:
            gh = self.target_sparsity - h0.mean(axis=0)

        logging.debug('dispacement: %.3g, hidden std: %.3g', np.linalg.norm(gv), h0.std(axis=1).mean())

        return gw, gv, gh

    def apply_gradients(self, weights, visible, hidden, learning_rate=0.2):
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (g - l2 * target)
            target += learning_rate * g
            _g[:] = g

        update('v_bias', visible, self.grad_vis)
        update('h_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)

if __name__ == '__main__':
    batch_size = 200
    
    rbm = RBM(15626, 2000, binary = True)
    trainer = RBMTrainer(rbm)

    # for i in range(train_model_mat.shape[0] / batch_size):
    #     logging.info('Training batch %d' % (i+1))
    #     batch = train_model_mat[i*batch_size:(i+1)*batch_size]
    #     trainer.learn(batch.todense())
        
    
