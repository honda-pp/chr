import chainer
from chainer import Variable, functions as F
import numpy as np
from functools import partial
from copy import deepcopy
import chainerrl
from chainerrl.agents import A2C

class A2C_Vmeta(A2C):
    def __init__(self, outerstepsize=0.1, innerstepsize=0.02, innerepochs=1, meta_batch_size=4, v_learn_epochs=4, gpu=0, *args, **kwargs):
        super().__init__(gpu=gpu, *args, **kwargs)
        self.v_meta = deepcopy(self.model.v)
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.v_meta.to_gpu(device=gpu)
        self.innerstepsize = innerstepsize
        self.innerepochs = innerepochs
        self.meta_batch_size = meta_batch_size
        self.outerstepsize = outerstepsize
        self.v_learn_epochs = v_learn_epochs
    """
    def _compute_returns(self, *args, **kwargs):
        if self.meta_phaze:
            super()._compute_returns(*args, **kwargs)
        else:
            pass
    """

    def meta_batch_train(self, inds, model):
        values = model(Variable(self.states[inds].reshape([-1] + list(self.obs_shape))))
        values = values.reshape((self.update_steps, self.num_processes))
        loss = F.mean((self.returns[inds] - values) ** 2)
        model.cleargrads()
        loss.backward()
        for params in model.params():
            params.data -= self.innerstepsize * params.grad
        return loss.data

    def meta_train(self, model, batch=True):
        if batch:
            loss = 0
            for _ in range(self.innerepochs):
                loss += self.meta_batch_train(self.xp.arange(self.states.shape[0]-1), model)
            return loss / self.innerepochs
        else:
            mb_iter = chainer.iterators.SerialIterator(np.arange(self.states.shape[0]-1), self.meta_batch_size)
            loss = 0
            n = 0
            for _ in range(self.innerepochs):
                while mb_iter.epoch == 0:
                    inds = self.xp.array(mb_iter.__next__())
                    loss += self.meta_batch_train(inds, model)
                    n +- 1
            return loss / n

    def meta_update(self, model):
        model_cp = deepcopy(model)
        loss = self.meta_train(model_cp)
        for params_cp, params in zip(model_cp.params(), model.params()):
            params.data = params_cp.data + self.outerstepsize * (params.data - params_cp.data)
        return loss

    def sync_params(self, model_base, model):
        for params, params_base in zip(model.params(), model_base.params()):
            params.data = params_base.data.copy()

    def update(self):
        #self.meta_phaze = True
        with chainer.no_backprop_mode():
            _, next_value = self.model.pi_and_v(self.states[-1])
            next_value = next_value.array[:, 0]
        self._compute_returns(next_value)
        self.meta_update(self.v_meta)
        #self.meta_phaze = False
        self.sync_params(self.v_meta, self.model.v)
        value_loss = 0
        for _ in range(self.v_learn_epochs):
            value_loss += self.meta_batch_train(self.xp.arange(self.states.shape[0]-1), self.model.v)
        value_loss /= self.v_learn_epochs
        
        #super().update()
        pout, values = \
            self.model.pi_and_v(chainer.Variable(
                self.states[:-1].reshape([-1] + list(self.obs_shape))))

        actions = chainer.Variable(
            self.actions.reshape([-1] + list(self.action_shape)))
        dist_entropy = F.mean(pout.entropy)
        action_log_probs = pout.log_prob(actions)

        values = values.reshape((self.update_steps, self.num_processes))
        action_log_probs = action_log_probs.reshape(
            (self.update_steps, self.num_processes))
        advantages = self.returns[:-1] - values
        action_loss = \
            - F.mean(advantages.array * action_log_probs)

        self.model.cleargrads()

        (action_loss * self.pi_loss_coef -
         dist_entropy * self.entropy_coeff).backward()

        self.optimizer.update()
        self.states[0] = self.states[-1]

        self.t_start = self.t

        # Update stats
        self.average_actor_loss += (
            (1 - self.average_actor_loss_decay) *
            (float(action_loss.array) - self.average_actor_loss))
        self.average_value += (
            (1 - self.average_value_decay) *
            (float(value_loss) - self.average_value))
        self.average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(dist_entropy.array) - self.average_entropy))
