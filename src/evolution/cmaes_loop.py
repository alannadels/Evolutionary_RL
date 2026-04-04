"""CMA-ES evolutionary search for optimal reward weight schedules.

Reference: Hansen, N. (2016). The CMA evolution strategy: A tutorial.
"""

import json
import os

import cma
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.evolution.evaluate import evaluate_candidate
from src.rewards.components import INTRINSIC_SCALE
from src.rewards.weight_schedule import WeightSchedule


def run_cmaes(
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
    """Run CMA-ES to evolve optimal intrinsic reward weight schedules.

    Args:
        env_id: MiniGrid environment ID.
        K: Number of control points for weight functions.
        population_size: CMA-ES population size (lambda).
        max_generations: Maximum number of CMA-ES generations.
        total_timesteps: PPO training timesteps per candidate evaluation.
        n_seeds_per_eval: Number of seeds to average fitness over per candidate.
        eval_seed: Fixed seed for evaluation episodes.
        n_eval_episodes: Number of evaluation episodes per seed.
        fully_observable: Whether to use FullyObsWrapper.
        sigma0: Initial step size for CMA-ES.
        ppo_kwargs: Additional PPO hyperparameters.
        seed: Random seed for CMA-ES.
        n_jobs: Number of parallel workers (-1 = all cores).
        output_dir: Directory to save per-generation and final results.

    Returns:
        Dictionary with best_params, best_fitness, history, and config.
    """
    os.makedirs(output_dir, exist_ok=True)

    dim = WeightSchedule.N_COMPONENTS * K
    x0 = np.full(dim, -3.0)  # softplus(-3) ≈ 0.049, small initial intrinsic weights
    eval_seeds = [seed + i for i in range(n_seeds_per_eval)]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': population_size,
        'seed': seed,
        'maxiter': max_generations,
        'bounds': [-10, 10],
        'verbose': -1,
        'tolx': 0,
        'tolfun': 0,
        'tolstagnation': max_generations,
    })

    history = []
    gen = 0

    while not es.stop():
        solutions = es.ask()

        fitnesses = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_candidate)(
                sol, env_id, total_timesteps, K, eval_seeds,
                eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
            )
            for sol in tqdm(solutions, desc=f'Gen {gen}', leave=False)
        )

        # CMA-ES minimizes — negate fitness (we maximise return)
        neg_fitnesses = [-f for f in fitnesses]
        es.tell(solutions, neg_fitnesses)

        best_idx = int(np.argmax(fitnesses))
        gen_stats = {
            'generation': gen,
            'best_fitness': float(max(fitnesses)),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'all_fitnesses': [float(f) for f in fitnesses],
            'best_params': solutions[best_idx].tolist(),
            'all_params': [sol.tolist() for sol in solutions],
        }
        history.append(gen_stats)

        print(f"Gen {gen}: best={gen_stats['best_fitness']:.4f}, "
              f"mean={gen_stats['mean_fitness']:.4f}, "
              f"std={gen_stats['std_fitness']:.4f}")

        with open(os.path.join(output_dir, f'gen_{gen:03d}.json'), 'w') as f:
            json.dump(gen_stats, f, indent=2)

        gen += 1

    best_params = es.result.xbest
    best_fitness = float(-es.result.fbest)

    results = {
        'algo': 'cmaes',
        'best_params': best_params.tolist(),
        'best_fitness': best_fitness,
        'history': history,
        'config': {
            'algo': 'cmaes',
            'env_id': env_id,
            'K': K,
            'population_size': population_size,
            'max_generations': max_generations,
            'total_timesteps': total_timesteps,
            'n_seeds_per_eval': n_seeds_per_eval,
            'seed': seed,
            'sigma0': sigma0,
            'x0': x0.tolist(),
            'eval_seed': eval_seed,
            'n_eval_episodes': n_eval_episodes,
            'fully_observable': fully_observable,
            'n_jobs': n_jobs,
            'intrinsic_scale': INTRINSIC_SCALE,
            'n_components': WeightSchedule.N_COMPONENTS,
            'bounds': [-10, 10],
            'total_budget': population_size * max_generations,
            'cmaes_options': {
                'tolx': 0,
                'tolfun': 0,
                'tolstagnation': max_generations,
            },
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(output_dir, 'best_params.npy'), best_params)

    return results
