"""Environment factory: creates MiniGrid and Gymnasium environments with appropriate wrappers."""

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import minigrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


def is_minigrid(env_id: str) -> bool:
    return env_id.startswith('MiniGrid')


def make_env(env_id: str, fully_observable: bool = True, seed: int = 42):
    """Create an environment with appropriate wrappers.

    For MiniGrid:
        MiniGrid env -> FullyObsWrapper -> ImgObsWrapper -> FlattenObservation

    For Taxi and other Gymnasium envs:
        Raw environment (no wrappers needed).
    """
    env = gym.make(env_id)
    if is_minigrid(env_id):
        if fully_observable:
            env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
    env.reset(seed=seed)
    return env
