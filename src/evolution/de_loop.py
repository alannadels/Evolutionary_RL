"""Differential Evolution search for optimal reward weight schedules.

Adapted from DEAlignment (point cloud registration) to the black-box PPO
fitness evaluation setting. Core algorithm is identical: DE/best/1/bin mutation
with binomial crossover and greedy selection.

Reference:
  Storn, R. & Price, K. (1997). Differential Evolution — A Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces.
  Journal of Global Optimization, 11(4), 341-359.
"""

import json
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.evolution.evaluate import evaluate_candidate
from src.rewards.components import INTRINSIC_SCALE
from src.rewards.weight_schedule import WeightSchedule


def run_de(
    env_id: str,
    K: int = 5,
    population_size: int = 12,
    max_generations: int = 15,
    total_timesteps: int = 200_000,
    n_seeds_per_eval: int = 1,
    eval_seed: int = 123,
    n_eval_episodes: int = 50,
    fully_observable: bool = True,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = 'best1bin',
    ppo_kwargs: dict = None,
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = 'results/evolution',
) -> dict:
    """Run DE/best/1/bin to evolve optimal intrinsic reward weight schedules.

    Each generation:
      1. Evaluate all P candidates in parallel.
      2. For each individual i, select the current best (x_best) plus two
         random distinct individuals (x_r1, x_r2) from the population.
      3. Mutant: v_i = x_best + F * (x_r1 - x_r2), clipped to bounds.
      4. Trial via binomial crossover: u_i[d] = v_i[d] if rand < CR else x_i[d],
         with at least one dimension always taken from v_i (j_rand).
      5. Greedy selection: keep trial if fitness(trial) >= fitness(x_i).

    Args:
        env_id: MiniGrid environment ID.
        K: Number of control points for weight functions.
        population_size: Number of individuals (P).
        max_generations: Number of generations.
        total_timesteps: PPO training timesteps per candidate evaluation.
        n_seeds_per_eval: Number of seeds to average fitness over per candidate.
        eval_seed: Fixed seed for evaluation episodes.
        n_eval_episodes: Number of evaluation episodes per seed.
        fully_observable: Whether to use FullyObsWrapper.
        F: Mutation scale factor in (0, 2]. Controls step size of mutation.
        CR: Crossover probability in [0, 1]. Controls fraction of mutant dims used.
        strategy: Mutation strategy. Only 'best1bin' is currently supported.
        ppo_kwargs: Additional PPO hyperparameters.
        seed: Random seed for population initialisation and crossover.
        n_jobs: Number of parallel workers (-1 = all cores).
        output_dir: Directory to save per-generation and final results.

    Returns:
        Dictionary with best_params, best_fitness, history, and config.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    dim = WeightSchedule.N_COMPONENTS * K   # 15
    P = population_size
    lower, upper = -10.0, 10.0

    eval_seeds = [seed + i for i in range(n_seeds_per_eval)]

    # Initialise population: uniform in bounds, with x0 = np.full(dim, -3.0) as first individual
    population = rng.uniform(lower, upper, (P, dim))
    population[0] = np.full(dim, -3.0)     # anchor at CMA-ES initialisation point

    history = []
    best_params_overall = population[0].copy()
    best_fitness_overall = -np.inf

    # Evaluate initial generation (generation 0)
    fitnesses = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_candidate)(
            population[i], env_id, total_timesteps, K, eval_seeds,
            eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
        )
        for i in tqdm(range(P), desc='Gen 0 (init)', leave=False)
    )
    fitnesses = np.array(fitnesses, dtype=float)

    best_idx = int(np.argmax(fitnesses))
    best_fitness_overall = float(fitnesses[best_idx])
    best_params_overall = population[best_idx].copy()

    gen_stats = {
        'generation': 0,
        'best_fitness': float(fitnesses[best_idx]),
        'mean_fitness': float(np.mean(fitnesses)),
        'std_fitness': float(np.std(fitnesses)),
        'all_fitnesses': fitnesses.tolist(),
        'best_params': population[best_idx].tolist(),
        'all_params': population.tolist(),
        'phase': 'init',
    }
    history.append(gen_stats)
    print(f"Gen 0 (init): best={gen_stats['best_fitness']:.4f}, "
          f"mean={gen_stats['mean_fitness']:.4f}, "
          f"std={gen_stats['std_fitness']:.4f}")
    with open(os.path.join(output_dir, 'gen_000.json'), 'w') as f:
        json.dump(gen_stats, f, indent=2)

    for gen in range(1, max_generations):
        # ── Mutation: DE/best/1 ────────────────────────────────────────────
        best_idx = int(np.argmax(fitnesses))
        x_best = population[best_idx]

        # For each individual, draw two distinct random indices ≠ i
        mutants = np.empty_like(population)
        for i in range(P):
            candidates = [j for j in range(P) if j != i]
            r1, r2 = rng.choice(candidates, size=2, replace=False)
            v = x_best + F * (population[r1] - population[r2])
            mutants[i] = np.clip(v, lower, upper)

        # ── Crossover: binomial ────────────────────────────────────────────
        trials = np.empty_like(population)
        for i in range(P):
            j_rand = rng.randint(0, dim)
            mask = rng.random(dim) < CR
            mask[j_rand] = True                 # guarantee at least one mutant dim
            trials[i] = np.where(mask, mutants[i], population[i])

        # ── Evaluate trials in parallel ────────────────────────────────────
        trial_fitnesses = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_candidate)(
                trials[i], env_id, total_timesteps, K, eval_seeds,
                eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
            )
            for i in tqdm(range(P), desc=f'Gen {gen}', leave=False)
        )
        trial_fitnesses = np.array(trial_fitnesses, dtype=float)

        # ── Greedy selection ───────────────────────────────────────────────
        improved = trial_fitnesses >= fitnesses
        population[improved] = trials[improved]
        fitnesses[improved] = trial_fitnesses[improved]

        # ── Track best overall ─────────────────────────────────────────────
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness_overall:
            best_fitness_overall = float(fitnesses[gen_best_idx])
            best_params_overall = population[gen_best_idx].copy()

        # ── Log generation ─────────────────────────────────────────────────
        gen_stats = {
            'generation': gen,
            'best_fitness': float(fitnesses[gen_best_idx]),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'all_fitnesses': fitnesses.tolist(),
            'best_params': population[gen_best_idx].tolist(),
            'all_params': population.tolist(),
        }
        history.append(gen_stats)

        print(f"Gen {gen}: best={gen_stats['best_fitness']:.4f}, "
              f"mean={gen_stats['mean_fitness']:.4f}, "
              f"std={gen_stats['std_fitness']:.4f}")

        with open(os.path.join(output_dir, f'gen_{gen:03d}.json'), 'w') as f:
            json.dump(gen_stats, f, indent=2)

    results = {
        'algo': 'de',
        'best_params': best_params_overall.tolist(),
        'best_fitness': best_fitness_overall,
        'history': history,
        'config': {
            'algo': 'de',
            'env_id': env_id,
            'K': K,
            'population_size': P,
            'max_generations': max_generations,
            'total_timesteps': total_timesteps,
            'n_seeds_per_eval': n_seeds_per_eval,
            'seed': seed,
            'x0': np.full(dim, -3.0).tolist(),
            'eval_seed': eval_seed,
            'n_eval_episodes': n_eval_episodes,
            'fully_observable': fully_observable,
            'n_jobs': n_jobs,
            'intrinsic_scale': INTRINSIC_SCALE,
            'n_components': WeightSchedule.N_COMPONENTS,
            'bounds': [-10, 10],
            'total_budget': P * max_generations,
            'de_options': {
                'strategy': strategy,
                'F': F,
                'CR': CR,
                'init': 'uniform with x0=np.full(15, -3.0) as first individual',
            },
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(output_dir, 'best_params.npy'), best_params_overall)

    return results
