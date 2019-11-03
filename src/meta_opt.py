from copy import deepcopy
import numpy as np
import numpy.random as rnd
import chainer
from chainerrl.misc.batch_states import batch_states
from v_meta import A2C_Vmeta

class Meta_Opt(A2C_Vmeta):
    def __init__(self, model, outerstepsize=0.1, innerstepsize=0.02, innerepochs=1, meta_batch_size=4, v_learn_epochs=1, gpu=0, gamma=0.9, 
                f_num=1000, num_processes=4, update_step=5, use_gue=False, tau=0.95, batch_states=batch_states, *args, **kwargs):
        self.model = model
        self.gpu = gpu
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self
        self.meta = deepcopy(model)
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

    def set_data(self):
        pass

    def gen_task(self):
        ind = rnd.randint(0, self.f_num, 1).item()
        states = np.load(self.s_files[ind])
        actions = np.load(self.a_files[ind])
        masks = np.ones([actions.shape[0], 1])
        masks[actions==-100] = 0
        rewards = masks.copy()
        for t, ac in enumerate(actions):
            if -100 in ac:
                if ac[ac!=-100].sum() != 0:
                    pass
                else:
                    states = states[:t]
                    masks = masks[:t]
                    rewards = rewards[:t]
                    break

        return states, masks, rewards
        