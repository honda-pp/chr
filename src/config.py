import argparse
import functools

import chainer
from chainer import functions as F
import gym
import numpy as np

import logging

import chainerrl
from chainerrl.agents import a2c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies
from chainerrl import v_function


class A2CFFSoftmax(chainer.ChainList, a2c.A2CModel):
    """An example of A2C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64, 64)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

def agp(parser=argparse.ArgumentParser(), env='CartPole-v0', seed=0, outdir=None, profile='store_true', steps=2*10**5, 
        update_steps=5, log_interval=10**3, eval_interval=10**3, eval_n_runs=14, reward_scale_factor=1e-2, gamma=0.99, 
        lr=7e-4, gpu=1,num_envs=8
    ):    
    parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default=outdir)
    parser.add_argument('--profile', action=profile)
    parser.add_argument('--steps', type=int, default=steps)
    parser.add_argument('--update-steps', type=int, default=update_steps)
    parser.add_argument('--log-interval', type=int, default=log_interval)
    parser.add_argument('--eval-interval', type=int, default=eval_interval)
    parser.add_argument('--eval-n-runs', type=int, default=eval_n_runs)
    parser.add_argument('--reward-scale-factor', type=float, default=reward_scale_factor)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-5)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=gamma,
                        help='discount factor')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter')
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer alpha')
    parser.add_argument('--gpu', '-g', type=int, default=gpu,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--num-envs', type=int, default=num_envs)
    return  parser.parse_args()