import chainer
from chainer import functions as F
import gym
import numpy as np
from functools import partial

import chainerrl
from chainerrl.agents import A2C
class A2C_Vcheet(A2C):
    def __init__(self, v_learn_length, env_name, num_v_env, max_episode_len=None, batch_size=32, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_learn_length = v_learn_length
        self.num_v_env = num_v_env
        self.max_episode_len = max_episode_len
        self.episode_len = np.zeros(num_v_env, dtype='i')
        self.bathc_size = batch_size
        seeds = np.arange(num_v_env) + (seed + 1) * self.num_processes
        def make_env(env_name, i):
            env = gym.make(env_name)
            env.seed(seeds[0])
            return chainerrl.wrappers.CastObservationToFloat32(env)
        self.env = chainerrl.envs.MultiprocessVectorEnv(
            [partial(
                make_env, env_name, i
            ) for i in range(self.num_v_env)]
        )
        self.v_ini_state = self.env.reset()
    def v_learn(self):
        obss = self.v_ini_state
        #statevar = self.batch_states(obss, self.xp, self.phi)
        #states = np.zeros(
        #    [self.v_learn_length + 1, self.num_v_env] + list(statevar.shape[1:]), dtype='f'
        #    )
        
        states = np.zeros(
            [self.v_learn_length + 1, self.num_v_env] + list(obss.shape[1:]), dtype='f'
            )
        rewards = np.zeros(
            (self.v_learn_length, self.num_v_env), dtype='f'
            )
        masks = np.zeros(
            (self.v_learn_length, self.num_v_env), dtype='f'
            )
        value_preds = np.zeros(
            (self.v_learn_length + 1, self.num_v_env), dtype='f'
            )
        #states[0] = chainer.cuda.to_cpu(statevar)
        states[0] = chainer.cuda.to_cpu(obss)
        for t in range(self.v_learn_length):
            
            actions, value = self.batch_act_and_critic(obss)
            obss, rs, dones, infos = self.env.step(actions)
            self.episode_len += 1
            
            #statevar = self.batch_states(obss, self.xp, self.phi)
            states[t+1] = obss
            value_preds[t] = value[:,0]
            masks[t] = np.logical_not(dones)
            rewards[t] = np.array(rs, dtype='f')

            resets = (self.max_episode_len == self.episode_len)
            resets = np.logical_or(
                    resets, [info.get('needs_reset', False) for info in infos])

            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)
            self.episode_len[end] = 0
            obss = self.env.reset(not_end)

        self.v_batch_observe_and_train(states, rewards, masks, value_preds)

    def batch_act_and_critic(self, batch_obs):
        statevar = self.batch_states(batch_obs, self.xp, self.phi)
        with chainer.no_backprop_mode():
            pout, valu = self.model.pi_and_v(statevar)
            action = pout.sample().array
        return chainer.cuda.to_cpu(action), chainer.cuda.to_cpu(valu.array)
    
    def _v_compute_returns(self, rewards, masks, value_preds):
        returns = np.zeros(
            (self.v_learn_length + 1, self.num_v_env), dtype='f'
        )
        if self.use_gae:
            gae = 0
            for i in reversed(range(self.v_learn_length)):
                delta = self.rewards[i] + \
                    self.gamma * value_preds[i + 1] * masks[i] - \
                    value_preds[i]
                gae = delta + self.gamma * self.tau * masks[i] * gae
                returns[i] = gae + value_preds[i]
        else:
            returns[-1] = value_preds[-1]
            for i in reversed(range(self.v_learn_length)):
                returns[i] = rewards[i] + \
                    self.gamma * returns[i + 1] * masks[i]
        return returns

    def v_batch_observe_and_train(self, states, rewards, masks, value_preds):
        with chainer.no_backprop_mode():
            _, next_value = self.batch_act_and_critic(states[-1])
        value_preds[-1] = next_value[:,0]
        returns = self._v_compute_returns(rewards, masks, value_preds)
        states = states.reshape([-1] + list(self.obs_shape))
        returns = returns[:-1].reshape(-1, 1)

        batch_iter = chainer.iterators.SerialIterator(np.arange(self.v_learn_length * self.num_v_env), self.bathc_size)
        while batch_iter.epoch == 0:
            batch_ind = batch_iter.__next__()
            batch_states = states[batch_ind]
            batch_returns = returns[batch_ind]
            if chainer.cuda.available and self.xp is chainer.cuda.cupy:
                batch_states = self.xp.to_gpu(batch_states)
                batch_returns = self.xp.to_gpu(batch_returns)
            _, values = self.model.pi_and_v(batch_states)
            value_loss = F.mean((batch_returns - values) ** 2)
            self.model.cleagrads()
            value_loss.backward()
            self.optimizer.update()



    def batch_observe_and_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.t - self.t_start == self.update_steps:
            self.v_learn()
        return super().batch_observe_and_train(batch_obs, batch_reward, batch_done, batch_reset)
