"""
Evaluation metrics for poker agents
"""


def evaluate_agents(env, agent_a, agent_b, num_hands=3000, label=''):
    """
    Run num_hands games and return average EV for agent_a (player 0).

    Args:
        env: RLCard environment
        agent_a: First agent (evaluated)
        agent_b: Second agent (opponent)
        num_hands: Number of hands to play
        label: Description label for logging

    Returns:
        tuple: (avg_ev, trajectories)
            - avg_ev: Average chips/hand for agent_a
            - trajectories: List of game trajectories
    """
    env.set_agents([agent_a, agent_b])
    total = 0
    trajectories_collected = []

    for _ in range(num_hands):
        # Reset hand sequence for DRQN agents
        if hasattr(agent_a, 'current_hand_sequence'):
            agent_a.current_hand_sequence = []

        trajectories, payoffs = env.run(is_training=False)
        total += payoffs[0]
        trajectories_collected.append(trajectories[0])

    avg_ev = total / num_hands

    if label:
        print(f'  {label:30s}: {avg_ev:+.3f} chips/hand')

    return avg_ev, trajectories_collected


def compute_action_distribution(trajectories, action_names):
    """
    Compute action distribution and bluff rate from trajectories.

    Args:
        trajectories: List of game trajectories
        action_names: Dictionary mapping action strings to display names

    Returns:
        dict: Statistics including action counts and bluff rate
    """
    action_counts = {name: 0 for name in action_names.values()}
    bluff_attempts = 0
    total_jack_steps = 0

    for hand_trajectory in trajectories:
        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})
            action_record = state.get('action_record', [])

            # Get player 0's actions
            player0_actions = [a for pid, a in action_record if pid == 0]
            if not player0_actions:
                continue

            action_str = player0_actions[-1]
            name = action_names.get(action_str, 'Unknown')
            action_counts[name] = action_counts.get(name, 0) + 1

            # Detect bluffs (raising with Jack)
            raw_hand = raw_obs.get('hand', '')
            rank = raw_hand[-1] if isinstance(raw_hand, str) and raw_hand else ''

            if rank == 'J':
                total_jack_steps += 1
                if action_str == 'raise':
                    bluff_attempts += 1

    total_actions = sum(action_counts.values())
    bluff_rate = 100 * bluff_attempts / total_jack_steps if total_jack_steps else 0

    return {
        'action_counts': action_counts,
        'total_actions': total_actions,
        'bluff_attempts': bluff_attempts,
        'total_jack_steps': total_jack_steps,
        'bluff_rate': bluff_rate
    }


def print_evaluation_summary(results):
    """
    Print formatted evaluation summary.

    Args:
        results: Dictionary of evaluation results
    """
    print('\n' + '=' * 60)
    print('FINAL EVALUATION SUMMARY')
    print('=' * 60)

    for matchup, metrics in results.items():
        print(f'\n{matchup}')
        print('-' * 40)
        print(f"  Expected Value: {metrics['ev']:+.3f} chips/hand")

        if 'action_stats' in metrics:
            stats = metrics['action_stats']
            action_counts = stats['action_counts']
            total = stats['total_actions']

            print(f"  Action Distribution:")
            for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / total if total else 0
                print(f"    {action:8s}: {count:5d} ({pct:.1f}%)")

            print(f"  Bluff Rate (raise with Jack): {stats['bluff_rate']:.1f}%  "
                  f"({stats['bluff_attempts']}/{stats['total_jack_steps']} decisions)")
