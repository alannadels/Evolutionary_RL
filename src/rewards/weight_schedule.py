"""Piecewise-linear reward weight schedule parameterized over K control points."""

import numpy as np


def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


class WeightSchedule:
    """Piecewise linear weight schedule over K control points.

    Parameterizes 3 weight functions (agency, novelty, reactivity)
    as piecewise linear functions over K evenly spaced control points across
    the training horizon. Softplus ensures non-negative weights. Weights are
    NOT normalized to sum to 1 during training — CMA-ES discovers both the
    relative proportions and absolute magnitudes. Normalization to sum-to-1
    is applied only for reporting/visualization.
    """

    N_COMPONENTS = 3  # agency, novelty, reactivity

    def __init__(self, raw_params: np.ndarray, total_timesteps: int, K: int = 5):
        assert raw_params.shape == (self.N_COMPONENTS * K,), (
            f"Expected shape ({self.N_COMPONENTS * K},), got {raw_params.shape}"
        )
        self.K = K
        self.total_timesteps = total_timesteps
        self.raw = raw_params.reshape(self.N_COMPONENTS, K)
        self.control_points = np.linspace(0, total_timesteps, K)
        self._weights = softplus(self.raw)

    def get_weights(self, timestep: int) -> np.ndarray:
        """Get [alpha, beta, gamma] (agency, novelty, reactivity) at a given training timestep.

        Returns raw (unnormalized) non-negative weights for training.
        Linearly interpolates between the two nearest control points.
        """
        t = np.clip(timestep, 0, self.total_timesteps)

        if t <= self.control_points[0]:
            return self._weights[:, 0]
        if t >= self.control_points[-1]:
            return self._weights[:, -1]

        idx = np.searchsorted(self.control_points, t, side='right') - 1
        idx = min(idx, self.K - 2)

        t0 = self.control_points[idx]
        t1 = self.control_points[idx + 1]
        frac = (t - t0) / (t1 - t0)

        w0 = self._weights[:, idx]
        w1 = self._weights[:, idx + 1]
        return w0 + frac * (w1 - w0)

    def get_normalized_weights(self, timestep: int) -> np.ndarray:
        """Get weights normalized to sum to 1 (for reporting/visualization)."""
        w = self.get_weights(timestep)
        total = w.sum()
        if total > 0:
            return w / total
        return np.ones(self.N_COMPONENTS) / self.N_COMPONENTS

    def get_all_weights(self, n_points: int = 100, normalized: bool = True) -> tuple:
        """Get weight trajectories sampled at n_points for visualization.

        Args:
            n_points: Number of sample points.
            normalized: If True, normalize weights to sum to 1 at each point.

        Returns:
            timesteps: array of shape (n_points,)
            weights: array of shape (3, n_points)
        """
        timesteps = np.linspace(0, self.total_timesteps, n_points)
        if normalized:
            weights = np.array([self.get_normalized_weights(int(t)) for t in timesteps]).T
        else:
            weights = np.array([self.get_weights(int(t)) for t in timesteps]).T
        return timesteps, weights
