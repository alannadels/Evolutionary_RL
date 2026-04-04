"""Composite reward wrapper combining task reward with time-varying intrinsic signals."""

import gymnasium as gym
import numpy as np
from src.rewards.components import compute_agency, compute_novelty, compute_reactivity
from src.rewards.weight_schedule import WeightSchedule


class CompositeRewardWrapper(gym.Wrapper):
    """Wraps a MiniGrid environment to provide composite developmental rewards.

    Intercepts step() to compute four reward components (agency, novelty,
    reactivity, direction) and combines them using time-varying weights
    from a WeightSchedule.
    """

    def __init__(self, env, weight_schedule: WeightSchedule):
        super().__init__(env)
        self.weight_schedule = weight_schedule
        self.global_step = 0
        self.visit_counts = {}
        self.goal_pos = None
        self.d_max = 1.0

        # Previous state for agency computation
        self._prev_pos = None
        self._prev_dir = None
        self._prev_carrying = None

    # Cell types that represent the terminal goal object across MiniGrid envs
    GOAL_TYPES = {'goal', 'ball', 'box'}

    def _find_goal(self):
        """Scan the grid to find the goal position.

        Searches for any cell type in GOAL_TYPES. This covers:
          - 'goal' (DoorKey and most standard envs)
          - 'ball' (KeyCorridor envs)
          - 'box'  (other fetch-style envs)
        """
        grid = self.env.unwrapped.grid
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is not None and cell.type in self.GOAL_TYPES:
                    self.goal_pos = (i, j)
                    self.d_max = (grid.width - 2) + (grid.height - 2)
                    return
        self.goal_pos = None

    def _get_agent_state(self):
        """Get current agent position, direction, and carrying status."""
        unwrapped = self.env.unwrapped
        pos = tuple(unwrapped.agent_pos)
        direction = unwrapped.agent_dir
        carrying = unwrapped.carrying is not None
        return pos, direction, carrying

    def _get_door_states(self):
        """Return a tuple of is_open booleans for all doors, ordered by position."""
        grid = self.env.unwrapped.grid
        doors = []
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is not None and cell.type == 'door':
                    doors.append(cell.is_open)
        return tuple(doors)

    def _get_state_key(self, pos, direction, carrying):
        """Create hashable state key for novelty counting.

        Includes position, facing direction, carrying status, and door states.
        This captures novel events beyond spatial exploration:
          - Turning in place (direction changes)
          - Picking up the key (carrying changes)
          - Opening a door (door state changes)
        """
        return (pos[0], pos[1], direction, carrying) + self._get_door_states()

    def _record_state(self):
        """Save current agent state for next step's agency computation."""
        self._prev_pos, self._prev_dir, self._prev_carrying = self._get_agent_state()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._find_goal()
        self._record_state()
        info['task_reward'] = 0.0
        info['reward_components'] = {
            'agency': 0.0, 'novelty': 0.0,
            'reactivity': 0.0, 'direction': 0.0
        }
        info['weights'] = self.weight_schedule.get_weights(self.global_step)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        curr_pos, curr_dir, curr_carrying = self._get_agent_state()

        # Compute 4 reward components
        r_agency = compute_agency(
            self._prev_pos, self._prev_dir, self._prev_carrying,
            curr_pos, curr_dir, curr_carrying
        )

        state_key = self._get_state_key(curr_pos, curr_dir, curr_carrying)
        r_novelty = compute_novelty(state_key, self.visit_counts)

        r_reactivity = compute_reactivity(
            curr_pos, self.goal_pos, self.d_max
        )

        r_direction = original_reward

        # Get weights for current global timestep
        # Additive formulation: task reward always present, intrinsic signals on top
        weights = self.weight_schedule.get_weights(self.global_step)

        # r_task (direction) is always at full strength
        # Intrinsic signals are supplementary with evolved weights
        composite = (
            r_direction
            + weights[0] * r_agency
            + weights[1] * r_novelty
            + weights[2] * r_reactivity
        )

        # Store info for logging and fitness evaluation
        info['task_reward'] = original_reward
        info['reward_components'] = {
            'agency': r_agency,
            'novelty': r_novelty,
            'reactivity': r_reactivity,
            'direction': r_direction,
        }
        info['weights'] = weights

        # Update state tracking
        self._record_state()
        self.global_step += 1

        return obs, composite, terminated, truncated, info
