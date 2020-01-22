import chainer
from chainer import Variable, functions as F
import numpy as np
import chainerrl
from chainerrl.agents import A2C
from chainer import serializers
from v_meta import A2C_Vmeta
from copy import deepcopy

class off_learn(A2C_Vmeta):
    def __init__(self, base_path, data_ind, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_states = np.load(base_path+"state"+str(data_ind)+".npy")
        self.all_rewards = np.load(base_path+"reward"+str(data_ind)+".npy")
        self.all_masks = np.load(base_path+"mask"+str(data_ind)+".npy")
        self.all_actions = np.load(base_path+"action"+str(data_ind)+".npy")
        
    def set_data(self, t):
        pass

"""
class A2C_not_learn(A2C):
    def __init__(self, model_path, compare, *args, **kwargs):
        super().__init__(*args, **kwargs)
        serializers.load_npz(model_path, self.model)
        self.compare = compare # Compare_meta()
    
    def update(self):
        self.compare.update()
        
    def batch_act_and_train(self, batch_obs):
        return super().batch_act_and_train(batch_obs)
    

class Compare_meta(A2C_Vmeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.not_meta_v = deepcopy(self.model.v)
        self.not_meta_opt = deepcopy(self.optimizer)
        self.not_meta_opt.setup(self.not_meta_v)

    def update(self):
        super().update()

        values = self.not_meta_v(chainer.Variable(self.states[:-1].reshape([-1] + list(self.obs_shape))))
        values = values.reshape((self.update_steps, self.num_processes))
        advantages = self.returns[:-1] - values
        value_loss = F.mean(advantages * advantages)
        self.not_meta_v.cleargrads()
        (value_loss * self.v_loss_coef).backward()
        self.not_meta_opt.update()
"""