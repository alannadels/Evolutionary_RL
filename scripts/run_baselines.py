"""Run all baseline conditions with multiple training seeds.

Usage:
    python scripts/run_baselines.py configs/doorkey6x6.yaml
    python scripts/run_baselines.py configs/keycorridor.yaml
    python scripts/run_baselines.py configs/doorkey6x6_cmaes.yaml --condition reversed
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.schedules import BASELINE_SCHEDULES
from src.training.ppo_trainer import train_ppo
from src.utils.runtime_log import log_runtime_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to yaml config file')
    parser.add_argument('--condition', default=None,
                        help='Run only this baseline condition (e.g. reversed)')
    parser.add_argument('--output-name', default=None,
                        help='Override the output directory name for the condition '
                             '(e.g. reversed_fixed). Only used with --condition.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_id = config['env']['id']
    total_timesteps = config['ppo']['total_timesteps']
    K = config['evolution']['K']
    eval_seed = config['evaluation']['eval_seed']
    n_eval_episodes = config['evaluation']['n_eval_episodes']
    fully_observable = config['env']['fully_observable']
    base_seed = config['seeds']['training_base']
    n_seeds = config['seeds']['n_experiment_seeds']
    output_base = os.path.join(config['output_dir'], 'baselines')
    log_runtime_config(config, step='baselines', output_dir=output_base)

    schedules_to_run = (
        {args.condition: BASELINE_SCHEDULES[args.condition]}
        if args.condition else BASELINE_SCHEDULES
    )
    # Map from output directory name -> schedule function
    # If --output-name is given, the single condition saves under that name instead
    output_name_override = args.output_name if args.condition else None

    for name, schedule_fn in schedules_to_run.items():
        out_name = output_name_override if output_name_override else name
        print(f"\n{'='*60}")
        print(f"Baseline: {name} -> {out_name}  |  {env_id}  |  {total_timesteps/1e6:.1f}M steps")
        print(f"{'='*60}")

        output_dir = os.path.join(output_base, out_name)
        os.makedirs(output_dir, exist_ok=True)

        schedule = schedule_fn(total_timesteps, K)
        all_results = []

        for i in range(n_seeds):
            seed = base_seed + i
            print(f"  Seed {seed} ({i+1}/{n_seeds})...", end=' ', flush=True)

            model_path = os.path.join(output_dir, 'models', f'seed_{seed}')
            result = train_ppo(
                env_id=env_id,
                weight_schedule=schedule,
                total_timesteps=total_timesteps,
                seed=seed,
                eval_seed=eval_seed,
                n_eval_episodes=n_eval_episodes,
                fully_observable=fully_observable,
                model_save_path=model_path,
            )

            print(f"fitness={result['fitness']:.4f}")
            all_results.append({
                'seed': seed,
                'fitness': result['fitness'],
                'eval_returns': result['eval_returns'],
                'training_task_rewards': result['training_task_rewards'],
                'learning_curve': result['learning_curve'],
            })

        fitnesses = [r['fitness'] for r in all_results]
        summary = {
            'baseline': out_name,
            'condition': name,
            'env_id': env_id,
            'total_timesteps': total_timesteps,
            'eval_seed': eval_seed,
            'n_eval_episodes': n_eval_episodes,
            'eval_freq_episodes': 100,
            'K': K,
            'schedule_params': schedule.raw.flatten().tolist(),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'all_results': all_results,
        }

        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f"  {out_name}: mean={summary['mean_fitness']:.4f} ± {summary['std_fitness']:.4f}")


if __name__ == '__main__':
    main()
