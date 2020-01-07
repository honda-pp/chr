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

evaluator.batch_run_evaluation_episodes = batch_run_evaluation_episodes

def main():
    
    args = agp(log_interval=100, eval_interval=100, steps=2*10**5, outdir="target")
    
    logging.basicConfig(level=args.logger_level)

    misc.set_random_seed(args.seed)


    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    modeldir = args.outdir + "/models"
    os.mkdir(modeldir)
    logger = logging.getLogger(modeldir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)

        # Scale rewards observed by agents
        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
        if args.render and process_idx == 0 and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space


    model = A2CFFSoftmax(obs_space.low.size, action_space.n)

    optimizer = chainer.optimizers.RMSprop(args.lr,
                                           eps=args.rmsprop_epsilon,
                                           alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.max_grad_norm))
    if args.weight_decay > 0:
        optimizer.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a2c.A2C(model, optimizer, gamma=args.gamma,
                    gpu=args.gpu,
                    num_processes=args.num_envs,
                    update_steps=args.update_steps,
                    use_gae=args.use_gae,
                    tau=args.tau)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            log_interval=args.log_interval,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            logger=logger
        )


if __name__ == '__main__':
    main()