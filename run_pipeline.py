"""
Main entry point for DRQN training and evaluation pipeline
Stages: Setup → Training (Phase 1 + Phase 2) → Evaluation → Analysis
"""

import os
import rlcard
from rlcard.utils import set_seed
from rlcard.agents import RandomAgent, DQNAgent

from config import *
from agents import DRQNAgent, ConservativeHeuristicAgent
from training import CurriculumTrainer
from evaluation import (
    evaluate_agents,
    compute_action_distribution,
    print_evaluation_summary,
    print_advanced_analysis,
    compare_agents,
    statistical_significance
)
from analysis import plot_training_curves, plot_comparison_curves, plot_ev_comparison_bar


def setup_environment():
    print("=" * 60)
    print("SETUP: Initializing Environment and Agents")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Set random seed
    set_seed(SEED)

    # Initialize environment
    env = rlcard.make(ENV_NAME)
    print(f'Environment: {ENV_NAME}')
    print(f'Device: {DEVICE}')

    # Get state and action dimensions
    raw_shape = env.state_shape[0]
    state_shape = raw_shape[0] if isinstance(raw_shape, list) else raw_shape
    num_actions = env.num_actions
    print(f'State shape: {state_shape} | Num actions: {num_actions}')

    # Initialize agents
    print("\nInitializing agents...")

    agent_drqn = DRQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        device=DEVICE,
        hidden_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        min_replay=MIN_REPLAY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )
    seq_info = f" (max seq len: {MAX_SEQUENCE_LENGTH})" if MAX_SEQUENCE_LENGTH else ""
    print(f"DRQN Agent (LSTM-based){seq_info}")

    agent_dqn = DQNAgent(
        num_actions=num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[HIDDEN_SIZE, HIDDEN_SIZE],
        device=DEVICE
    )
    print(f"DQN Agent (feedforward)")

    agent_random = RandomAgent(num_actions)
    print(f"Random Agent")

    agent_heuristic = ConservativeHeuristicAgent(num_actions)
    print(f"Heuristic Agent (rule-based)")

    print()
    return env, agent_drqn, agent_dqn, agent_random, agent_heuristic


def train_drqn(env, agent_drqn, agent_random, agent_heuristic):
    print("=" * 60)
    print("TRAINING DRQN AGENT")
    print("  Mode: Two-Phase Curriculum")
    print(f"  Phase 1 LR: {LEARNING_RATE}")
    print("=" * 60)

    trainer = CurriculumTrainer(
        env=env,
        agent=agent_drqn,
        opponent_phase1=agent_random,
        opponent_phase2=agent_heuristic
    )

    history = trainer.train()

    # Save model
    agent_drqn.save_model(DRQN_CHECKPOINT)
    print(f"\nSaved DRQN model to {DRQN_CHECKPOINT}")

    return history


def train_dqn(env, agent_dqn, agent_random, agent_heuristic):
    print("\n" + "=" * 60)
    print("TRAINING DQN AGENT (Baseline)")
    print("=" * 60)

    trainer = CurriculumTrainer(
        env=env,
        agent=agent_dqn,
        opponent_phase1=agent_random,
        opponent_phase2=agent_heuristic
    )

    history = trainer.train()
    return history


def evaluate_all(env, agent_drqn, agent_dqn, agent_random, agent_heuristic):
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION ({NUM_EVAL_HANDS} hands per matchup)")
    print("=" * 60)

    results = {}

    # DRQN evaluations
    print('\nDRQN vs ...')
    ev_vs_random, traj_drqn_vs_random = evaluate_agents(
        env, agent_drqn, agent_random, NUM_EVAL_HANDS, 'Random'
    )
    ev_vs_heuristic, traj_drqn_vs_heuristic = evaluate_agents(
        env, agent_drqn, agent_heuristic, NUM_EVAL_HANDS, 'Heuristic'
    )
    ev_vs_dqn, traj_drqn_vs_dqn = evaluate_agents(
        env, agent_drqn, agent_dqn, NUM_EVAL_HANDS, 'Standard DQN'
    )

    # DQN evaluations
    print('\nDQN vs ...')
    dqn_vs_random, traj_dqn_vs_random = evaluate_agents(
        env, agent_dqn, agent_random, NUM_EVAL_HANDS, 'Random'
    )
    dqn_vs_heuristic, traj_dqn_vs_heuristic = evaluate_agents(
        env, agent_dqn, agent_heuristic, NUM_EVAL_HANDS, 'Heuristic'
    )

    # Store results
    results['DRQN vs Random'] = {
        'ev': ev_vs_random,
        'trajectories': traj_drqn_vs_random,
        'action_stats': compute_action_distribution(traj_drqn_vs_random, ACTION_NAMES)
    }
    results['DRQN vs Heuristic'] = {
        'ev': ev_vs_heuristic,
        'trajectories': traj_drqn_vs_heuristic,
        'action_stats': compute_action_distribution(traj_drqn_vs_heuristic, ACTION_NAMES)
    }
    results['DRQN vs DQN'] = {
        'ev': ev_vs_dqn,
        'trajectories': traj_drqn_vs_dqn,
        'action_stats': compute_action_distribution(traj_drqn_vs_dqn, ACTION_NAMES)
    }
    results['DQN vs Random'] = {
        'ev': dqn_vs_random,
        'trajectories': traj_dqn_vs_random,
        'action_stats': compute_action_distribution(traj_dqn_vs_random, ACTION_NAMES)
    }
    results['DQN vs Heuristic'] = {
        'ev': dqn_vs_heuristic,
        'trajectories': traj_dqn_vs_heuristic,
        'action_stats': compute_action_distribution(traj_dqn_vs_heuristic, ACTION_NAMES)
    }

    # Print summary
    print_evaluation_summary(results)

    # Advanced analysis
    print_advanced_analysis(results)

    # Head-to-head comparison
    compare_agents(results, 'DRQN', 'DQN')

    # Statistical significance
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 60)

    sig_test = statistical_significance(ev_vs_dqn, NUM_EVAL_HANDS)
    print(f"\nDRQN vs DQN (head-to-head):")
    print(f"  Expected Value: {sig_test['ev']:+.3f} chips/hand")
    print(f"  Standard Error: ±{sig_test['std_error']:.3f}")
    print(f"  95% CI: [{sig_test['ci_95_lower']:+.3f}, {sig_test['ci_95_upper']:+.3f}]")
    print(f"  Z-score: {sig_test['z_score']:.2f}")
    print(f"  P-value: {sig_test['p_value']:.6f}")
    print(f"  Statistically Significant: {'YES' if sig_test['significant'] else 'NO'}")

    # Final verdict
    print("\n" + "=" * 60)
    if ev_vs_dqn > 0:
        print("✓ SUCCESS: DRQN outperforms the memoryless DQN!")
        print(f"  DRQN advantage: {ev_vs_dqn:+.3f} chips/hand")
        if sig_test['significant']:
            print(f"  Result is statistically significant (p < 0.05)")
    else:
        print("⚠ NOTE: DQN held its ground — consider more training episodes.")
    print("=" * 60)

    return results


def visualize_results(drqn_history, dqn_history, eval_results):
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Training curves (DRQN only)
    plot_training_curves(
        ev_history_random=drqn_history['ev_random'],
        ev_history_heuristic=drqn_history['ev_heuristic'],
        loss_history=drqn_history['loss'],
        phase1_episodes=NUM_EPISODES_PHASE1,
        save_path=f'{PLOT_DIR}/drqn_training_curves.png'
    )

    # DRQN vs DQN comparison
    plot_comparison_curves(
        drqn_hist_random=drqn_history['ev_random'],
        drqn_hist_heuristic=drqn_history['ev_heuristic'],
        dqn_hist_random=dqn_history['ev_random'],
        dqn_hist_heuristic=dqn_history['ev_heuristic'],
        phase1_episodes=NUM_EPISODES_PHASE1,
        save_path=f'{PLOT_DIR}/drqn_vs_dqn_comparison.png'
    )

    print("\nAll plots saved to", PLOT_DIR)


def main():
    """Main pipeline execution."""
    print("\n" + "=" * 60)
    print("DRQN LEDUC HOLD'EM TRAINING PIPELINE")
    print("Mastering Imperfect Information Games")
    print("=" * 60 + "\n")

    # Stage 1: Setup
    env, agent_drqn, agent_dqn, agent_random, agent_heuristic = setup_environment()

    # Stage 2: Training
    drqn_history = train_drqn(env, agent_drqn, agent_random, agent_heuristic)
    dqn_history = train_dqn(env, agent_dqn, agent_random, agent_heuristic)

    # Stage 3: Evaluation
    eval_results = evaluate_all(env, agent_drqn, agent_dqn, agent_random, agent_heuristic)

    # Stage 4: Visualization
    visualize_results(drqn_history, dqn_history, eval_results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")
    print("\nNext steps:")
    print("  1. Review plots in outputs/plots/")
    print("  2. Check detailed results in evaluation summary above")
    print("  3. Open notebooks/DLPW.ipynb for interactive analysis")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
