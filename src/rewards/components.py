"""Intrinsic reward component functions: agency, novelty, and reactivity."""

import numpy as np

# Intrinsic signal scaling coefficient. Intrinsic signals fire every step
# (~0.1-0.5 per step) while the task reward fires once per episode (~0.9).
# This coefficient brings intrinsic per-episode contribution to the same
# order of magnitude as the task reward, so the softplus weights from
# CMA-ES control relative importance meaningfully.
INTRINSIC_SCALE = 0.003


def compute_agency(prev_pos, prev_dir, prev_carrying, curr_pos, curr_dir, curr_carrying):
    """Agency reward: did the action cause a state change?

    Compares agent position, direction, and carrying status before and after
    the step. Returns scaled value if any changed, 0 otherwise.
    """
    pos_changed = prev_pos != curr_pos
    dir_changed = prev_dir != curr_dir
    carry_changed = prev_carrying != curr_carrying

    if isinstance(pos_changed, np.ndarray):
        pos_changed = pos_changed.any()

    return INTRINSIC_SCALE if (pos_changed or dir_changed or carry_changed) else 0.0


def compute_novelty(state_key, visit_counts: dict) -> float:
    """Novelty reward: count-based visitation bonus.

    Returns scaled 1/sqrt(N(s)) where N(s) is the visit count.
    The visit_counts dict is updated in place.
    """
    if state_key not in visit_counts:
        visit_counts[state_key] = 0
    visit_counts[state_key] += 1
    return INTRINSIC_SCALE / np.sqrt(visit_counts[state_key])


def compute_reactivity(agent_pos, goal_pos, d_max: float) -> float:
    """Reactivity reward: proximity-based attraction to goal.

    Returns scaled (1 - manhattan_distance(agent, goal) / d_max).
    If goal_pos is None (no goal found), returns 0.
    """
    if goal_pos is None or d_max <= 0:
        return 0.0
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    return INTRINSIC_SCALE * (1.0 - dist / d_max)
