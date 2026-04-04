"""Utility functions for plotting weight trajectories, performance comparisons, and learning curves."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.rewards.weight_schedule import WeightSchedule

# Style
sns.set_theme(style='whitegrid')
COMPONENT_NAMES = ['Agency', 'Novelty', 'Reactivity']
COMPONENT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a']


def plot_weight_trajectory(
    schedule: WeightSchedule,
    title: str = 'Weight Trajectory',
    save_path: str = None,
    ax: plt.Axes = None,
):
    """Plot the 3 weight functions over training time."""
    timesteps, weights = schedule.get_all_weights(n_points=200)
    normalized_t = timesteps / timesteps[-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for i, (name, color) in enumerate(zip(COMPONENT_NAMES, COMPONENT_COLORS)):
        ax.plot(normalized_t, weights[i], label=name, color=color, linewidth=2)

    ax.set_xlabel('Training Progress')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return ax


def plot_trajectory_comparison(
    schedules: dict,
    save_path: str = None,
):
    """Plot multiple weight trajectories side by side for comparison.

    Args:
        schedules: dict mapping name -> WeightSchedule
    """
    n = len(schedules)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, schedule) in zip(axes, schedules.items()):
        plot_weight_trajectory(schedule, title=name, ax=ax)
        if ax != axes[0]:
            ax.set_ylabel('')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


def plot_performance_comparison(
    results: dict,
    title: str = 'Performance Comparison',
    save_path: str = None,
):
    """Bar chart comparing mean fitness across conditions with error bars.

    Args:
        results: dict mapping condition_name -> list of fitness values
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]

    colors = sns.color_palette('Set2', len(names))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Mean Episodic Return (Task Reward)')
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=15, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


def plot_fitness_convergence(
    history: list,
    title: str = 'CMA-ES Fitness Convergence',
    save_path: str = None,
):
    """Plot CMA-ES best and mean fitness per generation.

    Args:
        history: list of dicts with 'generation', 'best_fitness', 'mean_fitness'
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    gens = [h['generation'] for h in history]
    best = [h['best_fitness'] for h in history]
    mean = [h['mean_fitness'] for h in history]

    ax.plot(gens, best, label='Best', color='#e41a1c', linewidth=2)
    ax.plot(gens, mean, label='Mean', color='#377eb8', linewidth=2)

    if 'std_fitness' in history[0]:
        std = [h['std_fitness'] for h in history]
        mean_arr = np.array(mean)
        std_arr = np.array(std)
        ax.fill_between(gens, mean_arr - std_arr, mean_arr + std_arr,
                         alpha=0.2, color='#377eb8')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Mean Eval Return)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


def plot_learning_curves(
    curves: dict,
    title: str = 'Learning Curves (Periodic Eval)',
    save_path: str = None,
):
    """Plot periodic eval learning curves for multiple conditions.

    Matches Arditi et al. 2025 protocol: every 100 training episodes,
    50 clean eval episodes are run and mean return is recorded.

    Args:
        curves: dict mapping condition_name -> list of seed curves, where
                each seed curve is a list of {'episode': int, 'mean_return': float}
                dicts (as stored in results.json learning_curve field).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = sns.color_palette('Set2', len(curves))

    for (name, seed_curves), color in zip(curves.items(), colors):
        # Extract aligned arrays across seeds
        min_len = min(len(c) for c in seed_curves)
        episodes = np.array([pt['episode'] for pt in seed_curves[0][:min_len]])
        returns = np.array([
            [pt['mean_return'] for pt in c[:min_len]]
            for c in seed_curves
        ])  # shape: (n_seeds, n_checkpoints)

        mean = returns.mean(axis=0)
        std = returns.std(axis=0)

        ax.plot(episodes, mean, label=name, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Mean Eval Return (Task Reward, 50 episodes)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig
