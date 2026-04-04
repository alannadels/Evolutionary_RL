"""Run the evolved schedule with multiple training seeds.

Usage:
    python scripts/run_evolved.py configs/doorkey6x6.yaml
    python scripts/run_evolved.py configs/keycorridor.yaml
"""

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.weight_schedule import WeightSchedule
from src.training.ppo_trainer import train_ppo
from src.utils.runtime_log import log_runtime_config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_id = config['env']['id']
    total_timesteps = config['ppo']['total_timesteps']
    K = config['evolution']['K']
    eval_seed = config['evaluation']['eval_seed']
    n_eval_episodes = config['evaluation']['n_eval_episodes']
    fully_observable = config['env']['fully_observable']
    base_seed = config['seeds']['training_base']
    n_seeds = config['seeds']['n_experiment_seeds']

    evolution_dir = os.path.join(config['output_dir'], 'evolution')
    output_dir = os.path.join(config['output_dir'], 'evolved')
    os.makedirs(output_dir, exist_ok=True)
    log_runtime_config(config, step='evolved', output_dir=output_dir)

    best_params = np.load(os.path.join(evolution_dir, 'best_params.npy'))
    schedule = WeightSchedule(best_params, total_timesteps, K)

    print(f"\n{'='*60}")
    print(f"Evolved schedule  |  {env_id}  |  {total_timesteps/1e6:.1f}M steps")
    print(f"{'='*60}")
    print(f"Running {n_seeds} seeds...")

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
        'condition': 'evolved',
        'env_id': env_id,
        'total_timesteps': total_timesteps,
        'eval_seed': eval_seed,
        'n_eval_episodes': n_eval_episodes,
        'eval_freq_episodes': 100,
        'K': K,
        'best_params': best_params.tolist(),
        'mean_fitness': float(np.mean(fitnesses)),
        'std_fitness': float(np.std(fitnesses)),
        'all_results': all_results,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print(f"\nEvolved: mean={summary['mean_fitness']:.4f} ± {summary['std_fitness']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
