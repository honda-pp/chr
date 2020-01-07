from config import *
import logging, time, os
import numpy as np
from chainerrl.experiments import evaluator
#from chainer import serializers

outnum = 0


def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
    ):
    global outnum
    logger = logger or logging.getLogger(__name__)
    outdir = logger.name
    #serializers.save_npz(outdir+'/model'+str(outnum), agent.model)
    agent.save(outdir+'/model'+str(outnum))
    outnum += 1    
    return evaluator.batch_run_evaluation_episodes(env, agent, n_steps, n_episodes, max_episode_len=max_episode_len, logger=logger)


