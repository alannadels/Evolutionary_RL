"""xNES evolutionary search for optimal reward weight schedules.

Exponential Natural Evolution Strategy (xNES).

Reference:
  Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010).
  Exponential natural evolution strategies. GECCO 2010.

  Wierstra, D., Forster, A., Peters, J., Togelius, J., Schmidhuber, J., &
  Schaul, T. (2014). Natural evolution strategies. JMLR, 15(1), 949-980.

Algorithm summary:
  Maintains a search distribution N(mu, sigma^2 * B * B^T) over R^n where:
    mu  : mean vector (n,)
    sigma: scalar step size
    B   : transformation matrix (n x n), initialised to identity

  Each generation:
    1. Sample z_k ~ N(0, I) for k = 1,...,lambda
    2. Compute x_k = mu + sigma * B @ z_k, clip to bounds
    3. Evaluate f(x_k), rank descending
    4. Compute utility values u_k (fitness shaping)
    5. Natural gradient:
         grad_delta = sum_k u_k * z_k
         grad_M     = sum_k u_k * (outer(z_k, z_k) - I)
    6. Update:
         mu    <- mu + eta_mu * sigma * B @ grad_delta
         sigma <- sigma * exp(eta_sigma / 2 * trace(grad_M))
         B     <- B @ expm(eta_B / 2 * (grad_M - trace(grad_M)/n * I))

  Learning rates (default from Wierstra et al. 2014):
    eta_mu    = 1
    eta_sigma = eta_B = (3 + ln(n)) / (5 * sqrt(n))

  Fitness shaping (rank-based utility, Wierstra et al. 2014 eq. 10):
    u_k = max(0, ln(lambda/2 + 1) - ln(rank_k)) for rank_k = 1,...,lambda
    Normalise: u_k = u_k / sum(u) - 1/lambda
"""

import json
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import expm
from tqdm import tqdm

from src.evolution.evaluate import evaluate_candidate
from src.rewards.components import INTRINSIC_SCALE
from src.rewards.weight_schedule import WeightSchedule


def _utility(lam: int) -> np.ndarray:
    """Compute rank-based utility values for fitness shaping.

    Returns utility array u of shape (lam,) where u[0] corresponds to the
    best individual (highest fitness). Utilities sum to zero.

    Args:
        lam: Population size.

    Returns:
        Utility array of shape (lam,), normalised to sum to zero.
    """
    # Raw utilities: u_k = max(0, ln(lam/2 + 1) - ln(k)) for k = 1,...,lam
    raw = np.array([max(0.0, np.log(lam / 2.0 + 1) - np.log(k))
                    for k in range(1, lam + 1)])
    # Normalise so they sum to zero
    u = raw / raw.sum() - 1.0 / lam
    return u


def run_xnes(
    env_id: str,
    K: int = 5,
    population_size: int = 12,
    max_generations: int = 15,
    total_timesteps: int = 200_000,
    n_seeds_per_eval: int = 1,
    eval_seed: int = 123,
    n_eval_episodes: int = 50,
    fully_observable: bool = True,
    sigma0: float = 2.0,
    ppo_kwargs: dict = None,
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = 'results/evolution',
) -> dict:
    """Run xNES to evolve optimal intrinsic reward weight schedules.

    Args:
        env_id: MiniGrid environment ID.
        K: Number of control points for weight functions.
        population_size: Population size (lambda) per generation.
        max_generations: Number of generations to run.
        total_timesteps: PPO training timesteps per candidate evaluation.
        n_seeds_per_eval: Number of seeds to average fitness over per candidate.
        eval_seed: Fixed seed for evaluation episodes.
        n_eval_episodes: Number of evaluation episodes per seed.
        fully_observable: Whether to use FullyObsWrapper.
        sigma0: Initial isotropic step size.
        ppo_kwargs: Additional PPO hyperparameters.
        seed: Random seed for sampling.
        n_jobs: Number of parallel workers (-1 = all cores).
        output_dir: Directory to save per-generation and final results.

    Returns:
        Dictionary with best_params, best_fitness, history, and config.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    dim = WeightSchedule.N_COMPONENTS * K            # 15
    lam = population_size
    lower, upper = -10.0, 10.0

    # Initialise distribution parameters
    mu = np.full(dim, -3.0)                          # mean: softplus(-3) ≈ 0.049
    sigma = float(sigma0)
    B = np.eye(dim)                                  # transformation matrix

    # Learning rates (Wierstra et al. 2014)
    eta_mu = 1.0
    eta_sigma_B = (3.0 + np.log(dim)) / (5.0 * np.sqrt(dim))

    # Precompute utility vector (fixed across generations)
    u = _utility(lam)                                # shape (lam,)

    eval_seeds = [seed + i for i in range(n_seeds_per_eval)]
    history = []
    best_params_overall = mu.copy()
    best_fitness_overall = -np.inf

    for gen in range(max_generations):
        # ── 1. Sample noise vectors z_k ~ N(0, I) ──────────────────────────
        Z = rng.randn(lam, dim)                      # shape (lam, dim)

        # ── 2. Compute candidate solutions and clip to bounds ───────────────
        X = mu + sigma * (Z @ B.T)                   # shape (lam, dim)
        X = np.clip(X, lower, upper)

        # ── 3. Evaluate candidates in parallel ──────────────────────────────
        fitnesses = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_candidate)(
                X[k], env_id, total_timesteps, K, eval_seeds,
                eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
            )
            for k in tqdm(range(lam), desc=f'Gen {gen}', leave=False)
        )
        fitnesses = np.array(fitnesses, dtype=float)

        # ── 4. Rank and assign utilities ────────────────────────────────────
        # Rank indices descending by fitness (best = rank 0)
        rank_order = np.argsort(fitnesses)[::-1]     # indices sorted best->worst
        # u[0] for best, u[1] for second-best, etc.
        u_assigned = np.zeros(lam)
        for rank, idx in enumerate(rank_order):
            u_assigned[idx] = u[rank]

        # ── 5. Natural gradients ────────────────────────────────────────────
        # grad_delta: weighted sum of noise vectors
        grad_delta = np.einsum('k,ki->i', u_assigned, Z)  # shape (dim,)

        # grad_M: weighted sum of (z_k z_k^T - I)
        grad_M = np.zeros((dim, dim))
        for k in range(lam):
            grad_M += u_assigned[k] * (np.outer(Z[k], Z[k]) - np.eye(dim))

        # ── 6. Update distribution parameters ──────────────────────────────
        mu = mu + eta_mu * sigma * B @ grad_delta

        # Scalar step size update: exp(eta_sigma/2 * tr(grad_M))
        sigma = sigma * np.exp(eta_sigma_B / 2.0 * np.trace(grad_M))
        sigma = float(np.clip(sigma, 1e-8, 1e3))    # numerical safety

        # Matrix update: B <- B @ expm(eta_B/2 * (grad_M - tr(grad_M)/n * I))
        rank_one_correction = (np.trace(grad_M) / dim) * np.eye(dim)
        B = B @ expm(eta_sigma_B / 2.0 * (grad_M - rank_one_correction))

        # ── 7. Track best overall ────────────────────────────────────────────
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness_overall:
            best_fitness_overall = float(fitnesses[gen_best_idx])
            best_params_overall = X[gen_best_idx].copy()

        # ── 8. Log generation ────────────────────────────────────────────────
        gen_stats = {
            'generation': gen,
            'best_fitness': float(np.max(fitnesses)),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'all_fitnesses': fitnesses.tolist(),
            'best_params': X[gen_best_idx].tolist(),
            'all_params': X.tolist(),
            'sigma': sigma,
        }
        history.append(gen_stats)

        print(f"Gen {gen}: best={gen_stats['best_fitness']:.4f}, "
              f"mean={gen_stats['mean_fitness']:.4f}, "
              f"std={gen_stats['std_fitness']:.4f}, "
              f"sigma={sigma:.4f}")

        with open(os.path.join(output_dir, f'gen_{gen:03d}.json'), 'w') as f:
            json.dump(gen_stats, f, indent=2)

    results = {
        'algo': 'xnes',
        'best_params': best_params_overall.tolist(),
        'best_fitness': best_fitness_overall,
        'history': history,
        'config': {
            'algo': 'xnes',
            'env_id': env_id,
            'K': K,
            'population_size': lam,
            'max_generations': max_generations,
            'total_timesteps': total_timesteps,
            'n_seeds_per_eval': n_seeds_per_eval,
            'seed': seed,
            'sigma0': sigma0,
            'x0': np.full(dim, -3.0).tolist(),
            'eval_seed': eval_seed,
            'n_eval_episodes': n_eval_episodes,
            'fully_observable': fully_observable,
            'n_jobs': n_jobs,
            'intrinsic_scale': INTRINSIC_SCALE,
            'n_components': WeightSchedule.N_COMPONENTS,
            'bounds': [-10, 10],
            'total_budget': lam * max_generations,
            'xnes_options': {
                'eta_mu': eta_mu,
                'eta_sigma_B': float(eta_sigma_B),
                'fitness_shaping': 'rank-based utility (Wierstra et al. 2014)',
            },
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(output_dir, 'best_params.npy'), best_params_overall)

    return results
