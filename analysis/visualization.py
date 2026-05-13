"""
Visualization utilities for training results and evaluation
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(ev_history_random, ev_history_heuristic, loss_history,
                        phase1_episodes, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DRQN Two-Phase Curriculum Training',
                 fontsize=14, fontweight='bold')

    # EV curves
    if ev_history_random:
        eps_r, evs_r = zip(*ev_history_random)
        axes[0].plot(eps_r, evs_r, 'b-o', markersize=4, label='vs Random')

    if ev_history_heuristic:
        eps_h, evs_h = zip(*ev_history_heuristic)
        axes[0].plot(eps_h, evs_h, 'g-s', markersize=4, label='vs Heuristic')

    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(phase1_episodes, color='orange', linestyle=':',
                   linewidth=1.5, label='Phase 2 starts')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Avg chips / hand')
    axes[0].set_title('Expected Value — Both Opponents')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curve
    if loss_history:
        eps_l, losses = zip(*loss_history)
        window = 200
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')

        axes[1].plot(eps_l[window-1:], smoothed, 'r-',
                    linewidth=1.5, label='Smoothed')
        axes[1].plot(eps_l, losses, 'r-', alpha=0.12,
                    linewidth=0.5, label='Raw')
        axes[1].axvline(phase1_episodes, color='orange', linestyle=':',
                       linewidth=1.5, label='Phase 2 starts')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Huber Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved training curves to {save_path}')

    plt.show()


def plot_comparison_curves(drqn_hist_random, drqn_hist_heuristic,
                           dqn_hist_random, dqn_hist_heuristic,
                           phase1_episodes, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DRQN vs DQN — Two-Phase Curriculum Training',
                 fontsize=14, fontweight='bold')

    for ax, (drqn_hist, dqn_hist), title in zip(
        axes,
        [(drqn_hist_random, dqn_hist_random),
         (drqn_hist_heuristic, dqn_hist_heuristic)],
        ['EV vs Random Agent', 'EV vs Heuristic Agent']
    ):
        if drqn_hist:
            eps, evs = zip(*drqn_hist)
            ax.plot(eps, evs, 'b-o', markersize=4, label='DRQN (LSTM)')

        if dqn_hist:
            eps, evs = zip(*dqn_hist)
            ax.plot(eps, evs, 'r-s', markersize=4, label='DQN (no memory)')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(phase1_episodes, color='orange', linestyle=':',
                  linewidth=1.5, label='Phase 2 starts')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg chips / hand')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved comparison curves to {save_path}')

    plt.show()


def plot_ev_comparison_bar(results, save_path=None):
    labels = list(results.keys())
    values = list(results.values())
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Average chips / hand')
    ax.set_title('Final EV Comparison — DRQN vs All Baselines',
                fontsize=13, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        offset = 0.03 if val >= 0 else -0.07
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f'{val:+.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved EV comparison to {save_path}')

    plt.show()


def plot_action_distribution(action_stats_dict, save_path=None):
    matchups = list(action_stats_dict.keys())
    num_matchups = len(matchups)

    fig, axes = plt.subplots(1, num_matchups, figsize=(5 * num_matchups, 5))
    if num_matchups == 1:
        axes = [axes]

    fig.suptitle('Action Distribution by Matchup', fontsize=14, fontweight='bold')

    for ax, (matchup, stats) in zip(axes, action_stats_dict.items()):
        action_counts = stats['action_counts']
        total = stats['total_actions']

        actions = list(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        percentages = [100 * c / total if total else 0 for c in counts]

        colors_map = {
            'Fold': 'tomato',
            'Call': 'skyblue',
            'Check': 'lightgreen',
            'Raise': 'gold'
        }
        colors = [colors_map.get(a, 'gray') for a in actions]

        ax.bar(actions, percentages, color=colors, edgecolor='white', linewidth=1.2)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(matchup)
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, (action, pct) in enumerate(zip(actions, percentages)):
            ax.text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved action distribution to {save_path}')

    plt.show()
