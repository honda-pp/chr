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
        self.outerstepsize = outerstepsize

    def _compute_returns(self, *args, **kwargs):
        if self.meta_phaze:
            super()._compute_returns(*args, **kwargs)
        else:
            pass

    def meta_batch_train(self, inds, model):
        values = model(Variable(self.states[inds].reshape([-1] + list(self.obs_shape))))
        loss = F.mean((self.returns[inds] - values) ** 2)
        model.cleargrads()
        loss.backward()
        for params in model.params():
            params.data -= self.innerstepsize * params.grad

    def meta_train(self, model):
        mb_iter = chainer.iterators.SerialIterator(np.arange(self.states.shape[0]-1), self.meta_batch_size)
        for _ in range(self.innerepochs):
            while mb_iter.epoch == 0:
                inds = mb_iter.__next__()
                self.meta_batch_train(inds, model)

    def meta_update(self, model):
        model_cp = deepcopy(model)
        self.meta_train(model_cp)
        for params_cp, params in zip(model_cp.params(), self.v_meta.params()):
            params.data = params_cp.data + self.outerstepsize * (params.data - params_cp.data)

    def sync_params(self, model_base, model):
        for params, params_base in zip(model.params(), model_base.params()):
            params.data = params_base.data.copy()

    def update(self):
        self.meta_phaze = True
        with chainer.no_backprop_mode():
            _, next_value = self.model.pi_and_v(self.states[-1])
            next_value = next_value.array[:, 0]
        self._compute_returns(next_value)
        self.meta_update(self.v_meta)
        self.meta_phaze = False
        self.sync_params(self.v_meta, self.model.v)
        super().update()