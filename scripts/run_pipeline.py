"""Full experimental pipeline for a single task.

Runs in order:
  1. CMA-ES evolution  → results/{task}/evolution/
  2. Evolved schedule (10 seeds) → results/{task}/evolved/
  3. All baselines (10 seeds each) → results/{task}/baselines/

Usage:
    python scripts/run_pipeline.py configs/doorkey6x6.yaml
    python scripts/run_pipeline.py configs/keycorridor.yaml
"""

import subprocess
import sys
import os
import time

def run_step(script, config_path, label):
    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"{'#'*60}\n")
    start = time.time()
    result = subprocess.run(
        [sys.executable, '-u', script, config_path],
        check=True,
    )
    elapsed = (time.time() - start) / 60
    print(f"\n[{label}] done in {elapsed:.1f} min")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    steps = [
        (os.path.join(root, 'scripts', 'run_evolution.py'), 'Step 1: CMA-ES Evolution'),
        (os.path.join(root, 'scripts', 'run_evolved.py'),   'Step 2: Evolved Schedule (10 seeds)'),
        (os.path.join(root, 'scripts', 'run_baselines.py'), 'Step 3: Baselines (10 seeds each)'),
    ]

    total_start = time.time()
    for script, label in steps:
        run_step(script, config_path, label)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total_elapsed:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
