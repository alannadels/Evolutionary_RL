"""PPO training loop with composite reward wrappers and periodic evaluation callbacks."""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.envs.env_factory import make_env, is_minigrid
from src.rewards.composite import CompositeRewardWrapper
from src.rewards.weight_schedule import WeightSchedule
from src.training.evaluation import evaluate_policy


class TaskRewardCallback(BaseCallback):
    """Tracks task reward per episode during training."""

    def __init__(self):
        super().__init__()
        self.episode_task_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            self._current_episode_reward += info.get('task_reward', 0.0)
            if info.get('episode'):
                self.episode_task_rewards.append(self._current_episode_reward)
                self._current_episode_reward = 0.0
        return True


class PeriodicEvalCallback(BaseCallback):
    """Evaluates the policy on the clean task every eval_freq episodes.

    Mirrors Arditi et al. 2025: every 100 training episodes, run 50 clean
    eval episodes and record mean return. Produces learning curves over time.
    """

    def __init__(
        self,
        env_id: str,
        eval_seed: int,
        n_eval_episodes: int,
        fully_observable: bool,
        eval_freq_episodes: int = 100,
    ):
        super().__init__()
        self.env_id = env_id
        self.eval_seed = eval_seed
        self.n_eval_episodes = n_eval_episodes
        self.fully_observable = fully_observable
        self.eval_freq_episodes = eval_freq_episodes

        self._episode_count = 0
        self.learning_curve = []  # list of (episode, mean_return)

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if info.get('episode'):
                self._episode_count += 1
                if self._episode_count % self.eval_freq_episodes == 0:
                    mean_return = np.mean(evaluate_policy(
                        self.model,
                        self.env_id,
                        n_episodes=self.n_eval_episodes,
                        seed=self.eval_seed,
                        fully_observable=self.fully_observable,
                        deterministic=True,
                    ))
                    self.learning_curve.append({
                        'episode': self._episode_count,
                        'mean_return': float(mean_return),
                    })
        return True


def train_ppo(
    env_id: str,
    weight_schedule: WeightSchedule,
    total_timesteps: int,
    seed: int,
    eval_seed: int = 123,
    n_eval_episodes: int = 50,
    fully_observable: bool = True,
    eval_freq_episodes: int = 100,
    model_save_path: str = None,
    ppo_kwargs: dict = None,
) -> dict:
    """Train a PPO agent with a composite reward schedule and evaluate.

    Matches Arditi et al. 2025 evaluation protocol:
      - 10 seeds per condition (called externally)
      - Every eval_freq_episodes training episodes: 50-episode clean eval
      - Final evaluation: n_eval_episodes clean episodes

    Args:
        env_id: MiniGrid environment ID.
        weight_schedule: WeightSchedule defining time-varying reward weights.
        total_timesteps: Number of training timesteps.
        seed: Random seed for training (env + PPO).
        eval_seed: Random seed for evaluation.
        n_eval_episodes: Number of episodes for final evaluation.
        fully_observable: Whether to use FullyObsWrapper.
        eval_freq_episodes: How often (in training episodes) to run periodic eval.
        ppo_kwargs: Additional PPO hyperparameters.

    Returns:
        Dictionary with 'fitness', 'eval_returns', 'training_task_rewards',
        and 'learning_curve' (list of {episode, mean_return} dicts).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(env_id, fully_observable=fully_observable, seed=seed)
    if is_minigrid(env_id):
        env = CompositeRewardWrapper(env, weight_schedule)
    else:
        from src.rewards.taxi_composite import TaxiCompositeRewardWrapper
        env = TaxiCompositeRewardWrapper(env, weight_schedule)

    default_kwargs = dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    if ppo_kwargs:
        default_kwargs.update(ppo_kwargs)

    model = PPO("MlpPolicy", env, **default_kwargs)

    task_cb = TaskRewardCallback()
    eval_cb = PeriodicEvalCallback(
        env_id=env_id,
        eval_seed=eval_seed,
        n_eval_episodes=n_eval_episodes,
        fully_observable=fully_observable,
        eval_freq_episodes=eval_freq_episodes,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[task_cb, eval_cb],
        progress_bar=True,
    )

    # Final evaluation
    eval_returns = evaluate_policy(
        model, env_id, n_episodes=n_eval_episodes,
        seed=eval_seed, fully_observable=fully_observable,
    )
    fitness = np.mean(eval_returns)

    if model_save_path:
        import os
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)

    env.close()

    return {
        'fitness': fitness,
        'eval_returns': eval_returns,
        'training_task_rewards': task_cb.episode_task_rewards,
        'learning_curve': eval_cb.learning_curve,
    }
