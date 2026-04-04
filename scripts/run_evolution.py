"""Run evolutionary search to discover optimal reward weight trajectories.

Dispatches to the appropriate algorithm based on config['evolution']['algo'].
Supported algorithms: cmaes, xnes, de, lshade.

Usage:
    python scripts/run_evolution.py configs/doorkey6x6_cmaes.yaml
    python scripts/run_evolution.py configs/doorkey6x6_xnes.yaml
    python scripts/run_evolution.py configs/doorkey6x6_de.yaml
    python scripts/run_evolution.py configs/doorkey6x6_lshade.yaml
    python scripts/run_evolution.py configs/keycorridor_cmaes.yaml
    # etc.
"""

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

    algo = config['evolution'].get('algo', 'cmaes')
    output_dir = os.path.join(config['output_dir'], 'evolution')
    log_runtime_config(config, step='evolution', output_dir=output_dir)

    # ── Dispatch to algorithm ──────────────────────────────────────────────
    if algo == 'cmaes':
        from src.evolution.cmaes_loop import run_cmaes
        evo_cfg = config['evolution'].get('cmaes', {})
        results = run_cmaes(
            env_id=config['env']['id'],
            K=config['evolution']['K'],
            population_size=config['evolution']['population_size'],
            max_generations=config['evolution']['max_generations'],
            total_timesteps=config['ppo']['total_timesteps'],
            n_seeds_per_eval=config['evolution']['n_seeds_per_eval'],
            eval_seed=config['evaluation']['eval_seed'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            fully_observable=config['env']['fully_observable'],
            sigma0=evo_cfg.get('sigma0', 2.0),
            seed=config['seeds']['training_base'],
            n_jobs=-1,
            output_dir=output_dir,
        )

    elif algo == 'xnes':
        from src.evolution.xnes_loop import run_xnes
        evo_cfg = config['evolution'].get('xnes', {})
        results = run_xnes(
            env_id=config['env']['id'],
            K=config['evolution']['K'],
            population_size=config['evolution']['population_size'],
            max_generations=config['evolution']['max_generations'],
            total_timesteps=config['ppo']['total_timesteps'],
            n_seeds_per_eval=config['evolution']['n_seeds_per_eval'],
            eval_seed=config['evaluation']['eval_seed'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            fully_observable=config['env']['fully_observable'],
            sigma0=evo_cfg.get('sigma0', 2.0),
            seed=config['seeds']['training_base'],
            n_jobs=-1,
            output_dir=output_dir,
        )

    elif algo == 'de':
        from src.evolution.de_loop import run_de
        evo_cfg = config['evolution'].get('de', {})
        results = run_de(
            env_id=config['env']['id'],
            K=config['evolution']['K'],
            population_size=config['evolution']['population_size'],
            max_generations=config['evolution']['max_generations'],
            total_timesteps=config['ppo']['total_timesteps'],
            n_seeds_per_eval=config['evolution']['n_seeds_per_eval'],
            eval_seed=config['evaluation']['eval_seed'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            fully_observable=config['env']['fully_observable'],
            F=evo_cfg.get('mutation_min', 0.8),
            CR=evo_cfg.get('recombination', 0.9),
            strategy=evo_cfg.get('strategy', 'best1bin'),
            seed=config['seeds']['training_base'],
            n_jobs=-1,
            output_dir=output_dir,
        )

    elif algo == 'lshade':
        from src.evolution.lshade_loop import run_lshade
        evo_cfg = config['evolution'].get('lshade', {})
        results = run_lshade(
            env_id=config['env']['id'],
            K=config['evolution']['K'],
            population_size=config['evolution']['population_size'],
            max_generations=config['evolution']['max_generations'],
            total_timesteps=config['ppo']['total_timesteps'],
            n_seeds_per_eval=config['evolution']['n_seeds_per_eval'],
            eval_seed=config['evaluation']['eval_seed'],
            n_eval_episodes=config['evaluation']['n_eval_episodes'],
            fully_observable=config['env']['fully_observable'],
            H=evo_cfg.get('H', 6),
            seed=config['seeds']['training_base'],
            n_jobs=-1,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: cmaes, xnes, de, lshade.")

    print(f"\nEvolution complete! ({algo.upper()})")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Results saved to: {output_dir}")

    # ── Train and save one model with the best params at the base seed ─────
    print(f"\nTraining best {algo.upper()} model (seed={config['seeds']['training_base']})...")
    best_params = np.array(results['best_params'])
    schedule = WeightSchedule(best_params, config['ppo']['total_timesteps'], config['evolution']['K'])
    model_path = os.path.join(output_dir, 'best_model')
    train_ppo(
        env_id=config['env']['id'],
        weight_schedule=schedule,
        total_timesteps=config['ppo']['total_timesteps'],
        seed=config['seeds']['training_base'],
        eval_seed=config['evaluation']['eval_seed'],
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        fully_observable=config['env']['fully_observable'],
        model_save_path=model_path,
    )
    print(f"Best {algo.upper()} model saved to: {model_path}.zip")


if __name__ == '__main__':
    main()
