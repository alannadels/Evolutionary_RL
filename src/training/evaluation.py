"""Policy evaluation under the original sparse task reward (no composite wrapper)."""

import gymnasium as gym
import numpy as np
from src.envs.env_factory import make_env, is_minigrid


def _encode_taxi_obs(env, obs):
    """Convert Taxi integer observation to feature vector matching training."""
    taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(obs)
    features = np.zeros(11, dtype=np.float32)
    features[0] = taxi_row / 4.0
    features[1] = taxi_col / 4.0
    features[2 + pass_loc] = 1.0
    features[7 + dest_idx] = 1.0
    return features


def evaluate_policy(
    model,
    env_id: str,
    n_episodes: int = 30,
    seed: int = 123,
    fully_observable: bool = True,
    deterministic: bool = True,
) -> list:
    """Evaluate a trained policy under the original sparse task reward.

    Creates a clean environment (no composite wrapper) and runs the trained
    policy for n_episodes. Returns the list of episodic returns.

    Args:
        model: Trained SB3 model.
        env_id: Environment ID.
        n_episodes: Number of evaluation episodes.
        seed: Random seed for evaluation environment.
        fully_observable: Whether to use FullyObsWrapper (MiniGrid only).
        deterministic: Whether to use deterministic actions.

    Returns:
        List of episodic returns (sparse task reward).
    """
    is_taxi = not is_minigrid(env_id)

    if is_taxi:
        env = gym.make(env_id)
    else:
        env = make_env(env_id, fully_observable=fully_observable, seed=seed)

    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if is_taxi:
            obs = _encode_taxi_obs(env, obs)
        episode_return = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            if is_taxi:
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            if is_taxi:
                obs = _encode_taxi_obs(env, obs)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

    env.close()
    return returns
