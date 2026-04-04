"""Runtime logging utility.

Prints and saves all experiment parameters at the start of each run.
Output is saved to {task_dir}/methods/{algo}_{step}_log.txt.
"""

import os
import sys
import platform
import datetime


def log_runtime_config(config: dict, step: str, output_dir: str):
    """Print and save all runtime parameters.

    Derives the algo name from config['evolution']['algo'] and saves the log
    to {task_dir}/methods/{algo}_{step}_log.txt, where task_dir is two levels
    above output_dir (e.g. results/doorkey6x6/{algo}/{step} -> results/doorkey6x6).

    Args:
        config: Full config dict loaded from yaml.
        step: Which pipeline step ('evolution', 'evolved', 'baselines').
        output_dir: The step-level output directory (e.g. results/doorkey6x6/cmaes/evolution).
    """
    import stable_baselines3, gymnasium, minigrid, cma, numpy, torch
    from src.rewards.components import INTRINSIC_SCALE

    algo = config['evolution'].get('algo', 'cmaes')

    lines = []

    def p(s=''):
        lines.append(s)
        print(s)

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    p('=' * 72)
    p(f'RUNTIME LOG — {algo.upper()} — {step.upper()}')
    p(f'Timestamp: {now}')
    p('=' * 72)

    p()
    p('SOFTWARE VERSIONS')
    p(f'  Python:             {sys.version.split()[0]}')
    p(f'  stable-baselines3:  {stable_baselines3.__version__}')
    p(f'  gymnasium:          {gymnasium.__version__}')
    p(f'  minigrid:           {minigrid.__version__}')
    p(f'  cma:                {cma.__version__}')
    p(f'  numpy:              {numpy.__version__}')
    p(f'  torch:              {torch.__version__}')
    p(f'  Platform:           {platform.platform()}')

    p()
    p('ENVIRONMENT')
    p(f'  env_id:             {config["env"]["id"]}')
    p(f'  fully_observable:   {config["env"]["fully_observable"]}')

    p()
    p('PPO HYPERPARAMETERS')
    ppo = config['ppo']
    p(f'  policy:             {ppo["policy"]}')
    p(f'  network:            [128, 128]')
    p(f'  learning_rate:      {ppo["learning_rate"]}')
    p(f'  n_steps:            {ppo["n_steps"]}')
    p(f'  batch_size:         {ppo["batch_size"]}')
    p(f'  n_epochs:           {ppo["n_epochs"]}')
    p(f'  gamma:              {ppo["gamma"]}')
    p(f'  gae_lambda:         {ppo["gae_lambda"]}')
    p(f'  clip_range:         {ppo["clip_range"]}')
    p(f'  ent_coef:           {ppo["ent_coef"]}')
    p(f'  vf_coef:            {ppo["vf_coef"]}')
    p(f'  max_grad_norm:      {ppo["max_grad_norm"]}')
    p(f'  total_timesteps:    {ppo["total_timesteps"]:,}')

    p()
    p('SEEDS')
    base = config['seeds']['training_base']
    n = config['seeds']['n_experiment_seeds']
    seeds = list(range(base, base + n))
    p(f'  training_base:      {base}')
    p(f'  n_experiment_seeds: {n}')
    p(f'  training_seeds:     {seeds}')
    p(f'  eval_seed:          {config["evaluation"]["eval_seed"]}')

    p()
    p('EVALUATION')
    p(f'  n_eval_episodes:    {config["evaluation"]["n_eval_episodes"]}')
    p(f'  eval_freq_episodes: 100')
    p(f'  deterministic:      True')

    p()
    p('REWARD FORMULATION')
    p(f'  intrinsic_scale:    {INTRINSIC_SCALE}')
    p(f'  formula:            r_task + w_agency*r_agency + w_novelty*r_novelty + w_reactivity*r_reactivity')
    p(f'  parameterization:   softplus(p) — non-negative weights guaranteed')
    p(f'  normalization:      NOT applied during training; visualization only')

    if step == 'evolution':
        p()
        p(f'EVOLUTION PARAMETERS — {algo.upper()}')
        evo = config['evolution']
        p(f'  algo:               {algo}')
        p(f'  K:                  {evo["K"]}')
        p(f'  population_size:    {evo["population_size"]}')
        p(f'  max_generations:    {evo["max_generations"]}')
        p(f'  n_seeds_per_eval:   {evo["n_seeds_per_eval"]}')
        p(f'  x0:                 np.full(15, -3.0)')
        p(f'  bounds:             [-10, 10]')
        p(f'  n_jobs:             -1 (all CPU cores)')
        p(f'  total_budget:       {evo["population_size"] * evo["max_generations"]}')

        if algo == 'cmaes' and 'cmaes' in evo:
            p(f'  sigma0:             {evo["cmaes"]["sigma0"]}')
            p(f'  tolx:               0')
            p(f'  tolfun:             0')
            p(f'  tolstagnation:      {evo["max_generations"]}')
        elif algo == 'xnes' and 'xnes' in evo:
            p(f'  sigma0:             {evo["xnes"]["sigma0"]}')
            p(f'  eta_mu:             1.0')
            p(f'  eta_sigma_B:        (3 + ln(n)) / (5 * sqrt(n))')
            p(f'  fitness_shaping:    weighted rank-based utility')
        elif algo == 'de' and 'de' in evo:
            de_cfg = evo['de']
            p(f'  strategy:           {de_cfg["strategy"]}')
            p(f'  F (mutation):       {de_cfg["mutation_min"]} (fixed)')
            p(f'  CR (recombination): {de_cfg["recombination"]}')
            p(f'  init:               uniform with x0=np.full(15, -3.0) as first individual')
        elif algo == 'lshade' and 'lshade' in evo:
            p(f'  H (history size):   {evo["lshade"]["H"]}')
            p(f'  N_min:              4')
            p(f'  F_init:             Cauchy(0.5, 0.1)')
            p(f'  CR_init:            Normal(0.5, 0.1)')
            p(f'  mutation:           current-to-pbest/1/bin')
            p(f'  pop_reduction:      linear N_init -> N_min')

    p()
    p('OUTPUT')
    p(f'  output_dir:         {config["output_dir"]}')
    p('=' * 72)

    # Save to task-level methods dir: go up two levels from step output_dir
    # e.g. results/doorkey6x6/cmaes/evolution -> results/doorkey6x6
    algo_dir = os.path.dirname(output_dir)       # results/doorkey6x6/cmaes
    task_dir = os.path.dirname(algo_dir)         # results/doorkey6x6
    methods_dir = os.path.join(task_dir, 'methods')
    os.makedirs(methods_dir, exist_ok=True)
    log_path = os.path.join(methods_dir, f'{algo}_{step}_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[Methods log saved: {log_path}]')
