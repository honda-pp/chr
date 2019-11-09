import os
from copy import deepcopy
import numpy as np
import numpy.random as rnd
import chainer
from chainer import serializers, Variable, functions as F
from chainerrl.misc.batch_states import batch_states
from chainerrl import links
from v_meta import A2C_Vmeta

class Meta_Opt(A2C_Vmeta):
    def __init__(self, outerstepsize=0.1, innerstepsize=0.02, innerepochs=1, meta_batch_size=4, v_learn_epochs=4, epochs=4000, check_interval=40,
                ndim_obs=4, hidden_sizes=(64, 64), t_v_learn_epochs=30, gpu=0, gamma=0.9, f_num=2000, num_processes=14, update_step=5, 
                use_gae=False, tau=0.95, batch_states=batch_states, outdir="t_models", *args, **kwargs):
        self.model = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        self.gpu = gpu
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            self.converter = chainer.cuda.to_gpu
        else:
            self.converter = lambda x: x
        self.meta = deepcopy(self.model)
        self.innerstepsize = innerstepsize
        self.innerepochs = innerepochs
        self.meta_batch_size = meta_batch_size
        self.outerstepsize = outerstepsize
        self.v_learn_epochs = v_learn_epochs
        self.epochs = epochs
        self.check_interval = check_interval
        self.num_processes = num_processes
        self.gamma = gamma
        self.update_steps = update_step
        self.use_gae = use_gae
        self.tau = tau
        self.xp = self.model.xp
        self.phi = lambda x: x
        self.batch_states = batch_states
        self.t_v_learn_epochs = t_v_learn_epochs
        self.f_num = f_num
        self.a_files = ["npy/action"+str(i)+".npy" for i in range(f_num)]
        self.s_files = ["npy/state"+str(i)+".npy" for i in range(f_num)]
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outdir = outdir


    def _flush_storage(self, obs_shape):
        obs_shape = obs_shape[1:]

        self.states = self.xp.zeros(
            [self.update_steps + 1, self.num_processes] + list(obs_shape),
            dtype='f')
        self.rewards = self.xp.zeros(
            (self.update_steps, self.num_processes), dtype='f')
        self.value_preds = self.xp.zeros(
            (self.update_steps + 1, self.num_processes), dtype='f')
        self.returns = self.xp.zeros(
            (self.update_steps + 1, self.num_processes), dtype='f')
        self.masks = self.xp.ones(
            (self.update_steps, self.num_processes), dtype='f')

        self.obs_shape = obs_shape

    def set_data(self, states, masks, rewards, inds):
        t_inds = inds[:self.update_steps+1]
        if len(t_inds) != self.update_steps + 1:
            t_inds = np.append(t_inds, [t_inds[-1]+1])
        self.states = self.converter(states[t_inds])
        t_inds = inds[:self.update_steps]
        self.masks = self.converter(masks[t_inds])
        self.rewards = self.converter(rewards[t_inds])

    def gen_task(self, ind=None):
        if ind is None:
            ind = rnd.randint(0, self.f_num, 1).item()
        states = np.load(self.s_files[ind])
        actions = np.load(self.a_files[ind])
        masks = np.ones([*actions.shape])
        masks[-1] = 0
        rewards = np.ones([*actions.shape])
        t_last = masks.shape[0]
        terminal = np.zeros(actions.shape[1], dtype='i')
        for t, ac in enumerate(actions):
            if -100 in ac:
                terminal[ac==-100] += 1
                if 0 in terminal:
                    states[t:, ac==-100]= states[:-t, ac==-100].copy()
                    masks[t-1, ac==-100] = 0
                    actions[t:,ac==-100] = actions[:-t,ac==-100].copy()
                else:
                    masks[t-1:] = 0
                    """
                    states = states[:t]
                    masks = masks[:t]
                    rewards = rewards[:t]
                    """
                    t_last = t
                    break
        t_start = rnd.randint(0, t_last - self.update_steps, 1).item()
        return states, masks, rewards, t_start, t_last, ind

    def meta_update(self, model):
        """
        wip
        """
        states, masks, rewards, t_start, _, ind = self.gen_task()
        inds = np.arange(self.update_steps+1) + t_start
        self.set_data(states, masks, rewards, inds)
        with chainer.no_backprop_mode():
            next_value = model(Variable(
                    self.converter(self.states[-1].reshape([-1] + list(self.obs_shape)))
                    ))
        next_value = chainer.cuda.to_cpu(next_value.array[:,0])
        self._compute_returns(next_value)
        return super().meta_update(model), ind

    def pre_train(self, name="vmeta"):
        for e in range(self.epochs):
            loss, ind = self.meta_update(self.model)
            print(e, ind, loss.item(), end=' ')
            if (e + 1) % self.check_interval == 0:
                states, masks, rewards, t_start, t_last, _ = self.gen_task(ind)
                self.sync_params(self.model, self.meta)
                inds = np.arange(self.update_steps+1) + t_start
                self.set_data(states, masks, rewards, inds)
                with chainer.no_backprop_mode():
                    next_value = model(Variable(
                            self.converter(self.states[-1].reshape([-1] + list(self.obs_shape)))
                            ))
                next_value = chainer.cuda.to_cpu(next_value.array[:,0])
                self._compute_returns(next_value)
                for _ in range(self.v_learn_epochs):
                    self.meta_batch_train(inds, self.meta)
                self.v_pef_check(self.meta, states, masks, rewards, t_last)
            else:
                print()
        serializers.save_npz(self.outdir+"/"+name+".npz", self.model)


    
    def set_value_preds(self):
        """
        wip
        """
        pass

    def _compute_returns(self, next_value):
        if self.use_gae:
            """
            wip
            self.value_preds[-1] = next_value
            self.set_value_preds(self)
            gae = 0
            for i in reversed(range(self.update_steps)):
                delta = self.rewards[i] + \
                    self.gamma * self.value_preds[i + 1] * self.masks[i] - \
                    self.value_preds[i]
                gae = delta + self.gamma * self.tau * self.masks[i] * gae
                self.returns[i] = gae + self.value_preds[i]
            """
            raise NotImplementedError()
        else:
            super()._compute_returns(next_value)
    

    def v_pef_check(self, model, states, masks, rewards, t_last):
        """
        wip
        """
        with chainer.no_backprop_mode():
            returns = np.zeros((t_last + 1, self.num_processes), dtype='f')
            next_value = model(Variable(
                        self.converter(states[t_last].reshape([-1] + list(self.obs_shape)))
                        ))
            next_value = chainer.cuda.to_cpu(next_value.array[:,0])
            returns[-1] = next_value
            if self.use_gae:
                """
                wip
                """
                raise NotImplementedError()
                gae = 0
                for i in reversed(range(t_last)):
                    delta = rewards[i] + \
                        self.gamma * self.value_preds[i + 1] * masks[i] - \
                        self.value_preds[i]
                    gae = delta + self.gamma * self.tau * masks[i] * gae
                    returns[i] = gae + self.value_preds[i]
            else:
                for i in reversed(range(t_last)):
                    returns[i] = rewards[i] + \
                        self.gamma * returns[i + 1] * masks[i]
            values = self.model(Variable(
                            self.converter(states[:t_last].reshape([-1] + list(self.obs_shape)))
                            )).reshape(t_last, self.num_processes)                        
            values = chainer.cuda.to_cpu(values.array)
            error = F.mean((returns[:-1] - values) ** 2)
            print('error', error.item())
        


    def learn_v_target(self, t):
        states, masks, rewards, _, t_last, _ = self.gen_task(t)
        self.meta_phaze = True
        for e in range(self.t_v_learn_epochs):
            phaze = rnd.randint(0, t_last, 1).item()
            inds = np.arange(0, t_last) + phaze
            inds %= t_last
            losses = np.zeros(t_last//self.update_steps, dtype='f')
            for i in range(0, t_last // self.update_steps):
                #t_inds = inds[i*self.update_steps:(i+1)*self.update_steps]
                self.set_data(states, masks, rewards, inds[i*self.update_steps:])
                with chainer.no_backprop_mode():
                    next_values = self.model(Variable(self.states[-1].reshape([-1] + list(self.obs_shape))))
                self._compute_returns(next_values.array[:,0])
                losses[i] = self.meta_train(self.model, batch=True).item()
            print(e, end=' ')
            for loss in losses:
                print(loss, end=' ')
            self.v_pef_check(self.model, states, masks, rewards, t_last)
        serializers.save_npz(self.outdir+"/v_t"+str(t)+'.npz', self.model)


if __name__=="__main__":
    import argparse
    from config import agp
    parser = argparse.ArgumentParser()
    parser.add_argument("--outerstepsize", type=float, default=0.1)
    parser.add_argument("--innerstepsize", type=float, default=0.02)
    parser.add_argument("--innerepochs", type=int, default=1)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument('--v_learn_epochs', type=int, default=1)
    parser.add_argument('--t_v_learn_epochs', type=int, default=20)
    parser.add_argument('--t', type=int, default=0)
    args = agp(parser, outdir='pre-meta')
    args = agp(parser, name='vmeta')

    meop = Meta_Opt(outerstepsize=args.outerstepsize, innerepochs=args.innerepochs, innerstepsize=args.innerstepsize, 
             t_v_learn_epochs=args.t_v_learn_epochs, gpu=args.gpu, outdir=args.outdir)
    meop._flush_storage([20,4])
    #meop.learn_v_target(args.t)
    meop.pre_train(args.name)
    