import chainer
#from chainer import Variable, functions as F
import numpy as np
#import chainerrl
#from chainerrl.agents import A2C
from chainer import serializers
from v_meta import A2C_Vmeta
#from copy import deepcopy
import os

class off_learn(A2C_Vmeta):
    def __init__(self, epochs=20000, outdir="trained/", base_path="dt/", gpu=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_ind = 1000
        self.base_path = base_path
        self.epochs = epochs
        if gpu is not None and gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            self.converter = chainer.cuda.to_gpu
        else:
            self.converter = lambda x: x
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.outdir = outdir
        self.name = "i-e" + str(self.innerepochs) + "i-s" + str(self.innerstepsize) + "o-s" + str(self.outerstepsize) + "v-e" + str(self.v_learn_epochs)
        self.xp = self.model.xp
        
    def set_data(self, data_ind, t_ind):
        all_states = np.load(self.base_path+"state"+str(data_ind)+".npy")
        all_rewards = np.load(self.base_path+"reward"+str(data_ind)+".npy")
        all_masks = np.load(self.base_path+"mask"+str(data_ind)+".npy")[1:]
        all_actions = np.load(self.base_path+"action"+str(data_ind)+".npy")
        self.states[:] = self.converter(all_states[t_ind:t_ind+self.update_steps+1])
        self.rewards[:] = self.converter(all_rewards[t_ind:t_ind+self.update_steps])
        self.masks[:] = self.converter(all_masks[t_ind:t_ind+self.update_steps])
        #self.actions[:] = all_actions[t_ind:t_ind+self.update_steps]
    def gen_task(self):
        data_ind = np.random.randint(0, self.max_ind)
        t_ind = np.random.randint(0, 195)
        self.set_data(data_ind, t_ind)
    def update(self):
        with chainer.no_backprop_mode():
            _, next_value = self.model.pi_and_v(self.states[-1])
            next_value = next_value.array[:, 0]
        self._compute_returns(next_value)
        self.meta_update(self.v_meta)
        self.sync_params(self.v_meta, self.model.v)
        for _ in range(self.v_learn_epochs):
            loss = self.meta_batch_train(self.xp.arange(self.states.shape[0]-1), self.model.v)
        return loss / self.v_learn_epochs
    def __call__(self):
        data_ind = 0
        states = self.batch_states(np.load(self.base_path+"state"+str(data_ind)+".npy")[0], self.xp, self.phi)
        actions = np.load(self.base_path+"reward"+str(data_ind)+".npy")[0]
        self._flush_storage(states.shape, actions)
        for e in range(self.epochs):
            self.gen_task()
            loss = self.update()
            print(e, loss)
        self.sync_params(self.v_meta, self.model.v)
        serializers.save_npz(self.outdir+"/"+self.name+".npz", self.model)




class A2C_not_learn(A2C_Vmeta):
    def __init__(self, model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_path is not None:
            serializers.load_npz(model_path, self.model)
        
    def meta_update(self):
        pass