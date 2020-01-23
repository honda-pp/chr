from config import *

def mk_dt(agent, env, steps, outdir, number):
    num_envs = env.num_envs
    obss = env.reset()
    rewards = np.zeros([steps, num_envs], dtype='f')
    states = np.zeros([steps+1, num_envs, *obss[0].shape], dtype='f')
    masks = np.ones([steps+1, num_envs], dtype='f')
    masks[0] *= 0
    states[0] = agent.batch_states(obss, np, agent.phi)
    actions = np.zeros([steps, num_envs], dtype='i')
    for t in range(steps):
        action = agent.batch_act(obss)
        actions[t] = action
        obss, rs, dones, infos = env.step(action)
        rewards[t] = rs
        resets = np.zeros(num_envs, dtype=bool)
        resets = np.logical_or(
            resets, [info.get('needs_reset', False) for info in infos])
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)
        obss = env.reset(not_end)
        states[t+1] = agent.batch_states(obss, np, agent.phi)
        masks[t+1] = not_end
        
    np.save(outdir+"/state"+number+".npy", states)
    np.save(outdir+"/action"+number+".npy", actions)
    np.save(outdir+"/mask"+number+".npy", masks)
    np.save(outdir+"/reward"+number+".npy", rewards)
        

    
def main():
        
    args = agp(log_interval=100, eval_interval=100, steps=200, outdir="data")
    
    logging.basicConfig(level=args.logger_level)

    misc.set_random_seed(args.seed)


    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

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
        agent.load(args.env[0]+"models/model"+args.load)

    mk_dt(
        agent=agent,
        env=make_batch_env(test=False),
        steps=args.steps,
        outdir=args.outdir,
        number=args.load
    )


if __name__ == '__main__':
    main()