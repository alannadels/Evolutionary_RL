"""L-SHADE evolutionary search for optimal reward weight schedules.

L-SHADE: Success-History based Adaptive Differential Evolution with
Linear population size reduction.

Reference:
  Tanabe, R. & Fukunaga, A. (2014). Improving the Search Performance of
  SHADE Using Linear Population Size Reduction. IEEE CEC 2014.

Algorithm summary:
  Maintains a population of N individuals and a history memory of size H.

  Each generation:
    1. For each i: sample F_i ~ Cauchy(MF[r], 0.1) and
                              CR_i ~ Normal(MCR[r], 0.1) from history
    2. Mutation (current-to-pbest/1):
         v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)
       where x_pbest is random from top p_best*N, x_r2 from pop ∪ archive
    3. Crossover: binomial with CR_i, guarantee j_rand
    4. Greedy selection: keep trial if f(trial) >= f(x_i)
    5. Update history: weighted Lehmer mean for S_F, weighted mean for S_CR
       using improvement magnitudes as weights
    6. Linear population size reduction: N decreases from N_init to N_min=4
       Worst individuals removed each generation.
"""

import json
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.evolution.evaluate import evaluate_candidate
from src.rewards.components import INTRINSIC_SCALE
from src.rewards.weight_schedule import WeightSchedule


def run_lshade(
    env_id: str,
    K: int = 5,
    population_size: int = 12,
    max_generations: int = 15,
    total_timesteps: int = 200_000,
    n_seeds_per_eval: int = 1,
    eval_seed: int = 123,
    n_eval_episodes: int = 50,
    fully_observable: bool = True,
    H: int = 6,
    ppo_kwargs: dict = None,
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = 'results/evolution',
) -> dict:
    """Run L-SHADE to evolve optimal intrinsic reward weight schedules.

    Args:
        env_id: MiniGrid environment ID.
        K: Number of control points for weight functions.
        population_size: Initial population size (N_init).
        max_generations: Number of generations to run.
        total_timesteps: PPO training timesteps per candidate evaluation.
        n_seeds_per_eval: Number of seeds to average fitness over per candidate.
        eval_seed: Fixed seed for evaluation episodes.
        n_eval_episodes: Number of evaluation episodes per seed.
        fully_observable: Whether to use FullyObsWrapper.
        H: History memory size for adaptive F and CR.
        ppo_kwargs: Additional PPO hyperparameters.
        seed: Random seed for sampling and population initialisation.
        n_jobs: Number of parallel workers (-1 = all cores).
        output_dir: Directory to save per-generation and final results.

    Returns:
        Dictionary with best_params, best_fitness, history, and config.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    dim = WeightSchedule.N_COMPONENTS * K   # 15
    N_init = population_size
    N_min = 4
    lower, upper = -10.0, 10.0
    p_best = 0.1    # fraction of top individuals used in current-to-pbest mutation

    eval_seeds = [seed + i for i in range(n_seeds_per_eval)]

    # Initialise population: uniform in bounds, anchor [0] = np.full(dim, -3.0)
    population = rng.uniform(lower, upper, (N_init, dim))
    population[0] = np.full(dim, -3.0)

    # History memory for adaptive parameters
    MF = np.full(H, 0.5)    # F memory, initialised to 0.5
    MCR = np.full(H, 0.5)   # CR memory, initialised to 0.5
    k = 0                    # circular pointer into history arrays

    # External archive stores replaced individuals for diversity in x_r2 selection
    archive = []

    history = []
    best_params_overall = population[0].copy()
    best_fitness_overall = -np.inf

    # ── Evaluate initial population (generation 0) ─────────────────────────
    N = N_init
    fitnesses = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_candidate)(
            population[i], env_id, total_timesteps, K, eval_seeds,
            eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
        )
        for i in tqdm(range(N), desc='Gen 0 (init)', leave=False)
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
        'N': N,
        'phase': 'init',
    }
    history.append(gen_stats)
    print(f"Gen 0 (init): best={gen_stats['best_fitness']:.4f}, "
          f"mean={gen_stats['mean_fitness']:.4f}, "
          f"std={gen_stats['std_fitness']:.4f}, N={N}")
    with open(os.path.join(output_dir, 'gen_000.json'), 'w') as f:
        json.dump(gen_stats, f, indent=2)

    for gen in range(1, max_generations):
        # ── Linear population size reduction ───────────────────────────────
        # N decreases linearly from N_init (gen=1) to N_min (gen=max_generations-1)
        N_new = max(N_min, round(
            N_min + (N_init - N_min) * (1.0 - gen / (max_generations - 1))
        ))
        if N_new < N:
            n_remove = N - N_new
            worst_indices = np.argsort(fitnesses)[:n_remove]
            keep = np.array([i for i in range(N) if i not in set(worst_indices)])
            population = population[keep]
            fitnesses = fitnesses[keep]
            N = len(population)

        # ── Sample adaptive F and CR from history ──────────────────────────
        r_idx = rng.randint(0, H, size=N)

        # F ~ truncated Cauchy(MF[r], 0.1), clamped to (0, 1]
        F_vals = MF[r_idx] + 0.1 * np.tan(np.pi * (rng.uniform(size=N) - 0.5))
        F_vals = np.clip(F_vals, 1e-8, 1.0)

        # CR ~ Normal(MCR[r], 0.1), clamped to [0, 1]
        CR_vals = MCR[r_idx] + 0.1 * rng.randn(N)
        CR_vals = np.clip(CR_vals, 0.0, 1.0)

        # ── Mutation: current-to-pbest/1 ───────────────────────────────────
        n_pbest = max(2, round(p_best * N))
        pbest_indices = np.argsort(fitnesses)[-n_pbest:]   # top p_best fraction

        # Combined pool: population + archive for x_r2 selection
        if archive:
            pop_archive = np.vstack([population, np.array(archive)])
        else:
            pop_archive = population.copy()
        n_pool = len(pop_archive)

        mutants = np.empty((N, dim))
        for i in range(N):
            # x_pbest: random from top p_best*N, allow i if no other choice
            pbest_cands = [idx for idx in pbest_indices if idx != i]
            if not pbest_cands:
                pbest_cands = list(pbest_indices)
            pbest_pick = int(rng.choice(pbest_cands))

            # x_r1: random from current population, distinct from i
            r1_cands = [j for j in range(N) if j != i]
            r1 = int(rng.choice(r1_cands))

            # x_r2: random from pop ∪ archive, distinct from i and r1
            r2_cands = [j for j in range(n_pool) if j != i and (j >= N or j != r1)]
            if not r2_cands:
                r2_cands = [j for j in range(n_pool) if j != i]
            r2 = int(rng.choice(r2_cands))

            v = (population[i]
                 + F_vals[i] * (population[pbest_pick] - population[i])
                 + F_vals[i] * (population[r1] - pop_archive[r2]))
            mutants[i] = np.clip(v, lower, upper)

        # ── Crossover: binomial ────────────────────────────────────────────
        trials = np.empty((N, dim))
        for i in range(N):
            j_rand = rng.randint(0, dim)
            mask = rng.random(dim) < CR_vals[i]
            mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], population[i])

        # ── Evaluate trials in parallel ────────────────────────────────────
        trial_fitnesses = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_candidate)(
                trials[i], env_id, total_timesteps, K, eval_seeds,
                eval_seed, n_eval_episodes, fully_observable, ppo_kwargs,
            )
            for i in tqdm(range(N), desc=f'Gen {gen}', leave=False)
        )
        trial_fitnesses = np.array(trial_fitnesses, dtype=float)

        # ── Greedy selection and history update ───────────────────────────
        S_F, S_CR, delta_f = [], [], []
        for i in range(N):
            if trial_fitnesses[i] >= fitnesses[i]:
                if trial_fitnesses[i] > fitnesses[i]:
                    # Record successful parameters and improvement
                    S_F.append(float(F_vals[i]))
                    S_CR.append(float(CR_vals[i]))
                    delta_f.append(float(trial_fitnesses[i] - fitnesses[i]))
                    archive.append(population[i].copy())
                population[i] = trials[i]
                fitnesses[i] = trial_fitnesses[i]

        # Cap archive to N_init individuals (random eviction)
        if len(archive) > N_init:
            surplus = len(archive) - N_init
            evict = set(rng.choice(len(archive), size=surplus, replace=False).tolist())
            archive = [archive[j] for j in range(len(archive)) if j not in evict]

        # Update history memory if any strictly-improving successes occurred
        if S_F:
            w = np.array(delta_f, dtype=float)
            w /= w.sum()
            SF = np.array(S_F)
            SCR = np.array(S_CR)
            # Weighted Lehmer mean for F (more weight to larger improvements)
            MF[k] = float(np.sum(w * SF ** 2) / np.sum(w * SF))
            # Weighted arithmetic mean for CR
            MCR[k] = float(np.dot(w, SCR))
            k = (k + 1) % H

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
            'N': N,
            'MF_mean': float(np.mean(MF)),
            'MCR_mean': float(np.mean(MCR)),
        }
        history.append(gen_stats)

        print(f"Gen {gen}: best={gen_stats['best_fitness']:.4f}, "
              f"mean={gen_stats['mean_fitness']:.4f}, "
              f"std={gen_stats['std_fitness']:.4f}, N={N}")

        with open(os.path.join(output_dir, f'gen_{gen:03d}.json'), 'w') as f:
            json.dump(gen_stats, f, indent=2)

    results = {
        'algo': 'lshade',
        'best_params': best_params_overall.tolist(),
        'best_fitness': best_fitness_overall,
        'history': history,
        'config': {
            'algo': 'lshade',
            'env_id': env_id,
            'K': K,
            'population_size': N_init,
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
            'total_budget': N_init * max_generations,
            'lshade_options': {
                'H': H,
                'N_min': N_min,
                'p_best': p_best,
                'F_init': 'Cauchy(0.5, 0.1)',
                'CR_init': 'Normal(0.5, 0.1)',
                'mutation': 'current-to-pbest/1/bin',
                'pop_reduction': f'linear {N_init} -> {N_min}',
            },
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(output_dir, 'best_params.npy'), best_params_overall)

    return results
