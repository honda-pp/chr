import logging
from chainerrl.experiments import train_agent_batch
from chainerrl.experiments import evaluator
from chainerrl.misc.makedirs import makedirs

def train_agent_batch_with_evaluation(agent,
                                      env,
                                      steps,
                                      eval_n_steps,
                                      eval_n_episodes,
                                      eval_interval,
                                      outdir,
                                      checkpoint_freq=None,
                                      max_episode_len=None,
                                      step_offset=0,
                                      eval_max_episode_len=None,
                                      return_window_size=100,
                                      eval_env=None,
                                      log_interval=None,
                                      successful_score=None,
                                      step_hooks=(),
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      ):
    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    train_agent_batch(
        agent, env, steps, outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        eval_interval=eval_interval,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger)


class Evaluator(evaluator.Evaluator):
    def a(self):
        pass