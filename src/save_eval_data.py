import logging, time
import numpy as np
from chainerrl.experiments import evaluator

def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.
    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.
    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_indices = np.zeros(num_envs, dtype='i')
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype='i')

    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    termination_conditions = False
    timestep = 0
    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = (episode_len == max_episode_len)
        resets = np.logical_or(
            resets, [info.get('needs_reset', False) for info in infos])

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode)[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(
                            episode_r[unfinished_index])
                        eval_episode_lens.append(
                            episode_len[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
            zip(eval_episode_lens, eval_episode_returns)):
        logger.info('evaluation episode %s length: %s R: %s',
                    i, epi_len, epi_ret)
    return [float(r) for r in eval_episode_returns]

evaluator.batch_run_evaluation_episodes = batch_run_evaluation_episodes


class Evaluator(evaluator.Evaluator):
    def evaluate_and_update_max_score(self, t, episodes):
        eval_stats = evaluator.eval_performance(
            self.env, self.agent, self.n_steps, self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in self.agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        evaluator.record_stats(self.outdir, values)
        if mean > self.max_score:
            self.logger.info('The best score is updated %s -> %s',
                             self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                evaluator.save_agent(self.agent, "best", self.outdir, self.logger)
        return mean