from copy import deepcopy
import numpy as np
import numpy.random as rnd
import chainer
from chainer import serializers, Variable
from chainerrl.misc.batch_states import batch_states
from chainerrl import links
from v_meta import A2C_Vmeta

class Meta_Opt(A2C_Vmeta):
    def __init__(self, outerstepsize=0.1, innerstepsize=0.02, innerepochs=1, meta_batch_size=4, v_learn_epochs=1, 
                ndim_obs=4, hidden_sizes=(64, 64), t_v_learn_epochs=30, gpu=0, gamma=0.9, f_num=2000, num_processes=4, update_step=5, 
                use_gue=False, tau=0.95, batch_states=batch_states, *args, **kwargs):
        self.model = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        self.gpu = gpu
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self
        self.meta = deepcopy(self.model)
        self.innerstepsize = innerstepsize
        self.innerepochs = innerepochs
        self.meta_batch_size = meta_batch_size
        self.outerstepsize = outerstepsize
        self.v_learn_epochs = v_learn_epochs
        self.num_processes = num_processes
        self.gamma = gamma
        self.update_steps = update_step
        self.use_gue = use_gue
        self.tau = tau
        self.xp = self.model.xp
        self.phi = lambda x: x
        self.batch_states = batch_states
        self.t_v_learn_epochs = t_v_learn_epochs
        self.f_num = f_num
        self.a_files = ["npy/action"+str(i)+".npy" for i in range(f_num)]
        self.s_files = ["npy/state"+str(i)+".npy" for i in range(f_num)]

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
        self.states = states[t_inds]
        t_inds = inds[:self.update_steps]
        self.masks = masks[t_inds]
        self.rewards = rewards[t_inds]

    def gen_task(self, ind=None):
        if ind is None:
            ind = rnd.randint(0, self.f_num, 1).item()
        states = np.load(self.s_files[ind])
        actions = np.load(self.a_files[ind])
        masks = np.ones([*actions.shape, 1])
        masks[0] = 0
        rewards = np.ones([*actions.shape, 1])
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
        return states, masks, rewards, t_start, t_last
        
    def learn_v_target(self, t):
        states, masks, rewards, _, t_last = self.gen_task(t)
        for e in range(self.t_v_learn_epochs):
            phaze = rnd.randint(0, t_last, 1).item()
            inds = np.arange(0, t_last) + phaze
            inds %= t_last
            losses = np.zeros(t_last//self.update_steps, dtype='f')
            for i in range(0, t_last // self.update_steps):
                #t_inds = inds[i*self.update_steps:(i+1)*self.update_steps]
                self.set_data(states, masks, rewards, inds[i*self.update_steps:])
                next_values = self.model(Variable(self.states[-1].reshape([-1] + list(self.obs_shape))))
                self._compute_returns(next_values)
                losses[i] = self.meta_train(self.model, batch=True)
            print(e, losses)
        serializers.save_npz('t_models/v_t'+str(t)+'.npz', self.model)


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
    args = agp(parser)

    meop = Meta_Opt(outerstepsize=args.outerstepsize, innerepochs=args.innerepochs, innerstepsize=args.innerstepsize, 
             t_v_learn_epochs=args.t_v_learn_epochs, gpu=args.gpu)
    meop.learn_v_target(args.t)