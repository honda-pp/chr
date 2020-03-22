import chainer
import numpy as np
from v_meta import A2C_Vmeta

class A2C_V_On_Meta(A2C_Vmeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_holder = Task_holder(*args, **kwargs)

class Task_holder(A2C_Vmeta):
    def __init__(self, capacity=5*10**5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.capacity = capacity
        self.init_task_holder()

    def init_task_holder(self):
        self.task_states = np.zeros([self.capacity, *self.states.shape], dtype='f')
        self.task_actions = np.zeros([self.capacity, *self.actions.shape], dtype=self.actions.dtype)
        self.task_masks = np.ones([self.capacity, *self.masks.shape], dtype='f')
        self.task_rewards = np.zeros([self.capacity, *self.rewards.shape], dtype='f')

    def set_task(self, ind):
        self.states = chainer.dataset.concat_examples(self.task_states[ind], device=self.gpu)
        self.actions = chainer.dataset.concat_examples(self.task_actions[ind], device=self.gpu)
        self.masks = chainer.dataset.concat_examples(self.task_masks[ind], device=self.gpu)
        self.rewards = chainer.dataset.concat_examples(self.task_rewards[ind], device=self.gpu)

    def push_task(self, states, actions, masks, rewards):
        self.task_states[self.counter] = chainer.cuda.to_cpu(states)
        self.task_actions[self.counter] = chainer.cuda.to_cpu(actions)
        self.task_masks[self.counter] = chainer.cuda.to_cpu(masks)
        self.task_rewards[self.counter] = chainer.cuda.to_cpu(rewards)
        self.counter += 1
        if self.counter >= self.capacity:
            self.counter = 0