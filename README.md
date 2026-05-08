# Evolutionary Developmental Reward Schedules

This repository implements an evolutionary framework for discovering optimal developmental reward weight schedules in deep reinforcement learning. Rather than fixing reward weights throughout training, we parameterize three intrinsic motivation signals — agency, novelty, and reactivity — as time-varying piecewise-linear functions and use evolutionary optimization to discover schedules that maximize task performance.

---

## Overview

Biological organisms develop under progressively changing motivational drives. Inspired by this, we ask: *can evolutionary optimization rediscover biologically-aligned reward development?*

We evolve reward weight schedules over three intrinsic signals:
- **Agency**: reward for causing state changes (movement, picking up objects, opening doors)
- **Novelty**: count-based exploration bonus (1/√N(s))
- **Reactivity**: proximity-based attraction to the task goal

Each signal's weight follows a piecewise-linear trajectory parameterized over K=5 control points. Four evolutionary algorithms search this parameter space:
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy
- **xNES** — Exponential Natural Evolution Strategy
- **DE** — Differential Evolution
- **L-SHADE** — Success-History based Adaptive Differential Evolution with linear population size reduction

Experiments run on two MiniGrid environments: **DoorKey-6x6** and **KeyCorridorS3R1**.

---

## Repository Structure

```
├── src/
│   ├── rewards/
│   │   ├── components.py        # Agency, novelty, reactivity reward functions
│   │   ├── composite.py         # Gym wrapper combining task + intrinsic rewards
│   │   └── weight_schedule.py   # Piecewise-linear weight schedule
│   ├── baselines/
│   │   └── schedules.py         # Fixed baseline schedules (ablation conditions)
│   ├── envs/
│   │   └── env_factory.py       # Environment construction with wrappers
│   ├── training/
│   │   ├── ppo_trainer.py       # PPO training loop + evaluation callbacks
│   │   ├── evaluation.py        # Clean policy evaluation (sparse task reward)
│   │   └── feature_extractor.py # Custom CNN for MiniGrid observations
│   ├── evolution/
│   │   ├── evaluate.py          # Shared candidate evaluation function
│   │   ├── cmaes_loop.py        # CMA-ES search loop
│   │   ├── xnes_loop.py         # xNES search loop
│   │   ├── de_loop.py           # DE search loop
│   │   └── lshade_loop.py       # L-SHADE search loop
│   └── utils/
│       ├── plotting.py          # Trajectory and performance visualization utilities
│       └── runtime_log.py       # Experiment configuration logging
├── scripts/
│   ├── run_evolution.py         # Run evolutionary search (dispatches by algo)
│   ├── run_evolved.py           # Evaluate evolved schedule across 10 seeds
│   ├── run_baselines.py         # Run ablation baseline conditions
│   └── run_pipeline.py          # Run full pipeline (evolution → evolved → baselines)
├── configs/
│   ├── doorkey6x6_cmaes.yaml    # DoorKey-6x6 + CMA-ES config
│   ├── doorkey6x6_xnes.yaml     # DoorKey-6x6 + xNES config
│   ├── doorkey6x6_de.yaml       # DoorKey-6x6 + DE config
│   ├── doorkey6x6_lshade.yaml   # DoorKey-6x6 + L-SHADE config
│   ├── keycorridor_cmaes.yaml   # KeyCorridor + CMA-ES config
│   ├── keycorridor_xnes.yaml    # KeyCorridor + xNES config
│   ├── keycorridor_de.yaml      # KeyCorridor + DE config
│   └── keycorridor_lshade.yaml  # KeyCorridor + L-SHADE config
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

### Run full pipeline (evolution + evaluation + baselines)

```bash
python scripts/run_pipeline.py configs/doorkey6x6_cmaes.yaml
python scripts/run_pipeline.py configs/keycorridor_cmaes.yaml
```

### Run evolutionary search only

```bash
python scripts/run_evolution.py configs/doorkey6x6_cmaes.yaml
python scripts/run_evolution.py configs/doorkey6x6_xnes.yaml
python scripts/run_evolution.py configs/doorkey6x6_de.yaml
python scripts/run_evolution.py configs/doorkey6x6_lshade.yaml
```

### Evaluate the evolved schedule (10 seeds)

```bash
python scripts/run_evolved.py configs/doorkey6x6_cmaes.yaml
```

### Run ablation baselines

```bash
python scripts/run_baselines.py configs/doorkey6x6_cmaes.yaml
```

Results are saved to the `output_dir` specified in each config.

---

## Method

The reward at each timestep is:

```
r = r_task + α(t)·r_agency + β(t)·r_novelty + γ(t)·r_reactivity
```

where α(t), β(t), γ(t) are time-varying weights produced by the `WeightSchedule` class. The task reward r_task is always present at full strength; intrinsic signals are supplementary.

Each evolutionary algorithm searches over the 15-dimensional parameter space (3 components × K=5 control points) to maximize mean episodic return across evaluation seeds.

---

## Baselines

Four ablation conditions are compared against each evolved schedule:

| Condition | Description |
|-----------|-------------|
| Extrinsic Only | No intrinsic signals (task reward only) |
| Developmental | Fixed schedule matching biological order: agency → novelty → reactivity |
| Reversed | Reversed biological order: reactivity → novelty → agency |
| Fixed Equal | All three intrinsic signals at constant equal weights throughout training |

---

## Citation

If you use this code, please cite:

```
TBD
```
