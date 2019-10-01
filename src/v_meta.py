import chainer
from chainer import Variable, functions as F
import numpy as np
from functools import partial
from copy import deepcopy
import chainerrl
from chainerrl.agents import A2C

class A2C_Vmeta(A2C):
    def __init__(self, outerstepsize=0.1, innerstepsize=0.02, innerepochs=1, meta_batch_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_meta = deepcopy(self.model.v)
        self.innerstepsize = innerstepsize
        self.innerepochs = innerepochs
        self.meta_batch_size = meta_batch_size

    def _compute_returns(self, *args, **kwargs):
        if self.meta_phaze:
            super()._compute_returns(*ags, **kwargs)
        else:
            pass

    def meta_batch_train(self, inds):
        values = self.v_meta(Variable(self.states[inds].reshape([-1] + list(self.obs_shape))))
        loss = F.mean((self.returns[inds] - values) ** 2)
        self.v_meta.cleargrads()
        loss.backward()
        for params in self.v_meta.params():
            params.data -= self.innerstepsize * params.grad

    def meta_train(self):
        before = deepcopy(self.v_meta)
        batch_iter = chainer.iterators.SerialIterator(np.arange(self.states.shape[0]-1), self.meta_batch_size)
        for _ in range(self.innerepochs):
            while batch_iter.epoch == 0:
                inds = batch_iter.__next__()
                self.meta_batch_train(inds)
        for params_before, params in zip(before.params(), self.v_meta.params()):
            params.data = params_before.data + (params.data - params_before.data)

    def reset_params(self):
        for params, params_meta in zip(self.model.v.params(), self.v_meta.params()):
            params.data = params_meta.data.copy()


    def update(self):
        self.meta_phaze = True
        with chainer.no_backprop_mode():
            _, next_value = self.model.pi_and_v(self.states[-1])
            next_value = next_value.array[:, 0]
        self._compute_returns(next_value)
        self.meta_train()
        self.meta_phaze = False
        self.reset_params()
        self.update()