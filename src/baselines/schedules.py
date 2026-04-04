"""Fixed baseline reward weight schedules for ablation experiments."""

import numpy as np
from src.rewards.weight_schedule import WeightSchedule


def extrinsic_only(total_timesteps: int, K: int = 5) -> WeightSchedule:
    """Baseline 1: Only task reward (no intrinsic signals).

    All intrinsic weights near zero via large negative softplus inputs.
    """
    params = np.full((3, K), -10.0)  # all near zero after softplus
    return WeightSchedule(params.flatten(), total_timesteps, K)


def fixed_equal(total_timesteps: int, K: int = 5) -> WeightSchedule:
    """Baseline 2: Equal intrinsic weights at all times.

    All params at 0 -> softplus(0) = ln(2) ~ 0.693 for each.
    """
    params = np.zeros(3 * K)
    return WeightSchedule(params, total_timesteps, K)


def developmental(total_timesteps: int, K: int = 5) -> WeightSchedule:
    """Baseline 3: Sequential peaks matching biological developmental order.

    agency -> novelty -> reactivity (then all fade, leaving pure task reward)
    """
    params = np.full((3, K), -5.0)  # base: near zero after softplus

    # K=5 gives control points at 0%, 25%, 50%, 75%, 100% of training
    # Agency peaks early
    params[0, 0] = 3.0
    params[0, 1] = 1.0

    # Novelty peaks second
    params[1, 1] = 3.0
    params[1, 2] = 1.0

    # Reactivity peaks third
    params[2, 2] = 3.0
    params[2, 3] = 1.0

    return WeightSchedule(params.flatten(), total_timesteps, K)


def reversed_developmental(total_timesteps: int, K: int = 5) -> WeightSchedule:
    """Baseline 4: Reversed developmental order.

    reactivity -> novelty -> agency (opposite of biological order)
    """
    params = np.full((3, K), -5.0)

    # Reactivity peaks early
    params[2, 0] = 3.0
    params[2, 1] = 1.0

    # Novelty peaks second
    params[1, 1] = 3.0
    params[1, 2] = 1.0

    # Agency peaks last then fades (mirrors developmental structure)
    params[0, 3] = 3.0
    params[0, 4] = 1.0

    return WeightSchedule(params.flatten(), total_timesteps, K)


BASELINE_SCHEDULES = {
    'extrinsic_only': extrinsic_only,
    'fixed_equal': fixed_equal,
    'developmental': developmental,
    'reversed': reversed_developmental,
}
