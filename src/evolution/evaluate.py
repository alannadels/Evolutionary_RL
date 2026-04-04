"""Shared candidate evaluation for all evolutionary algorithms.

Each evolutionary algorithm calls evaluate_candidate() to assess a single
parameter vector by training PPO and returning mean eval fitness.
"""

import numpy as np

from src.rewards.weight_schedule import WeightSchedule
from src.training.ppo_trainer import train_ppo


def evaluate_candidate(
    raw_params: np.ndarray,
    env_id: str,
    total_timesteps: int,
    K: int,
    seeds: list,
    eval_seed: int,
    n_eval_episodes: int,
    fully_observable: bool,
    ppo_kwargs: dict = None,
) -> float:
    """Evaluate a single candidate parameter vector.

    Trains PPO with the schedule defined by raw_params and returns mean
    fitness across all seeds. Periodic evaluation is disabled (eval_freq_episodes
    set to a very large value) since only final fitness is needed during search.

    Args:
        raw_params: Raw pre-softplus parameters, shape (N_COMPONENTS * K,).
        env_id: MiniGrid environment ID.
        total_timesteps: PPO training timesteps.
        K: Number of weight schedule control points.
        seeds: List of training seeds to average over.
        eval_seed: Seed for evaluation episodes.
        n_eval_episodes: Number of evaluation episodes per seed.
        fully_observable: Whether to use FullyObsWrapper.
        ppo_kwargs: Additional PPO hyperparameters.

    Returns:
        Mean fitness (mean episodic return) across all seeds.
    """
    schedule = WeightSchedule(raw_params, total_timesteps, K)
    fitnesses = []
    for seed in seeds:
        result = train_ppo(
            env_id=env_id,
            weight_schedule=schedule,
            total_timesteps=total_timesteps,
            seed=seed,
            eval_seed=eval_seed,
            n_eval_episodes=n_eval_episodes,
            fully_observable=fully_observable,
            eval_freq_episodes=10 ** 9,  # disable periodic eval during search
            ppo_kwargs=ppo_kwargs,
        )
        fitnesses.append(result['fitness'])
    return float(np.mean(fitnesses))
