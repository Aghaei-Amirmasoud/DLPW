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


def compute_advanced_metrics(trajectories):
    """
    Compute advanced poker metrics from trajectories.

    Args:
        trajectories: List of game trajectories

    Returns:
        dict: Advanced statistics including aggression, VPIP, etc.
    """
    # Voluntary Put-in-Pot (VPIP): % of hands where player voluntarily put money in
    # Pre-Flop Raise (PFR): % of hands where player raised pre-flop
    # Aggression Factor (AF): (raises + bets) / calls
    # WTSD: Went to Showdown %

    total_hands = len(trajectories)
    vpip_hands = 0  # Hands where player voluntarily invested chips
    pfr_hands = 0   # Hands where player raised in first round
    total_raises = 0
    total_calls = 0
    total_folds = 0
    total_checks = 0
    showdown_hands = 0

    # Round-specific metrics
    round1_actions = {'raise': 0, 'call': 0, 'fold': 0, 'check': 0}
    round2_actions = {'raise': 0, 'call': 0, 'fold': 0, 'check': 0}

    # Hand strength metrics
    jack_hands = 0
    queen_hands = 0
    king_hands = 0

    for hand_trajectory in trajectories:
        hand_actions = []
        hand_card = None
        current_round = 1
        went_to_showdown = False

        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})
            action_record = state.get('action_record', [])

            # Track hand card
            if hand_card is None:
                raw_hand = raw_obs.get('hand', '')
                if raw_hand:
                    rank = raw_hand[-1] if isinstance(raw_hand, str) else ''
                    hand_card = rank
                    if rank == 'J':
                        jack_hands += 1
                    elif rank == 'Q':
                        queen_hands += 1
                    elif rank == 'K':
                        king_hands += 1

            # Detect round change (when public card appears)
            if raw_obs.get('public_card'):
                current_round = 2

            # Get player 0's actions
            player0_actions = [a for pid, a in action_record if pid == 0]
            if player0_actions:
                action = player0_actions[-1]
                hand_actions.append(action)

                # Count action types
                if action == 'raise':
                    total_raises += 1
                elif action == 'call':
                    total_calls += 1
                elif action == 'fold':
                    total_folds += 1
                elif action == 'check':
                    total_checks += 1

                # Round-specific tracking
                if current_round == 1:
                    round1_actions[action] = round1_actions.get(action, 0) + 1
                else:
                    round2_actions[action] = round2_actions.get(action, 0) + 1

        # VPIP: Did player voluntarily put chips in? (call or raise, not check)
        if any(a in ['call', 'raise'] for a in hand_actions):
            vpip_hands += 1

        # PFR: Did player raise in round 1?
        if hand_actions and hand_actions[0] == 'raise':
            pfr_hands += 1

        # WTSD: Check if hand went to showdown (no fold in actions)
        if 'fold' not in hand_actions and len(hand_actions) > 0:
            went_to_showdown = True
            showdown_hands += 1

    # Calculate percentages
    vpip = 100 * vpip_hands / total_hands if total_hands else 0
    pfr = 100 * pfr_hands / total_hands if total_hands else 0
    wtsd = 100 * showdown_hands / total_hands if total_hands else 0
    aggression_factor = (total_raises) / max(total_calls, 1)  # Avoid div by zero
    fold_to_aggression = 100 * total_folds / max(total_hands, 1)

    return {
        'vpip': vpip,  # Voluntary Put in Pot %
        'pfr': pfr,    # Pre-Flop Raise %
        'af': aggression_factor,  # Aggression Factor
        'wtsd': wtsd,  # Went to Showdown %
        'fold_pct': 100 * total_folds / max(total_hands, 1),
        'round1_aggression': 100 * round1_actions['raise'] / max(sum(round1_actions.values()), 1),
        'round2_aggression': 100 * round2_actions['raise'] / max(sum(round2_actions.values()), 1),
        'jack_hands': jack_hands,
        'queen_hands': queen_hands,
        'king_hands': king_hands,
        'total_hands': total_hands
    }


def compute_win_rate_by_hand(trajectories):
    """
    Compute win rate broken down by starting hand (J, Q, K).

    Args:
        trajectories: List of game trajectories

    Returns:
        dict: Win rates for each hand type
    """
    hand_results = {
        'J': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0},
        'Q': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0},
        'K': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0}
    }

    for hand_trajectory in trajectories:
        hand_card = None
        payoff = 0

        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})

            # Get hand card
            if hand_card is None:
                raw_hand = raw_obs.get('hand', '')
                if raw_hand:
                    hand_card = raw_hand[-1] if isinstance(raw_hand, str) else ''

            # Get payoff (last state)
            if 'payoff' in state:
                payoff = state['payoff']

        if hand_card in hand_results:
            hand_results[hand_card]['total'] += 1
            if payoff > 0:
                hand_results[hand_card]['wins'] += 1
            elif payoff < 0:
                hand_results[hand_card]['losses'] += 1
            else:
                hand_results[hand_card]['ties'] += 1

    # Calculate win rates
    for card in hand_results:
        total = hand_results[card]['total']
        if total > 0:
            hand_results[card]['win_rate'] = 100 * hand_results[card]['wins'] / total
            hand_results[card]['avg_payoff'] = (
                (hand_results[card]['wins'] - hand_results[card]['losses']) / total
            )

    return hand_results


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


def print_advanced_analysis(results):
    """
    Print advanced analysis including poker-specific metrics.

    Args:
        results: Dictionary of evaluation results with trajectories
    """
    print('\n' + '=' * 60)
    print('ADVANCED ANALYSIS')
    print('=' * 60)

    for matchup, metrics in results.items():
        if 'trajectories' not in metrics:
            continue

        print(f'\n{matchup}')
        print('-' * 40)

        # Compute advanced metrics
        adv_metrics = compute_advanced_metrics(metrics['trajectories'])

        print(f"  Playing Style Metrics:")
        print(f"    VPIP (Voluntary Put in Pot):  {adv_metrics['vpip']:.1f}%")
        print(f"    PFR (Pre-Flop Raise):          {adv_metrics['pfr']:.1f}%")
        print(f"    Aggression Factor:             {adv_metrics['af']:.2f}")
        print(f"    Went to Showdown:              {adv_metrics['wtsd']:.1f}%")
        print(f"    Fold Percentage:               {adv_metrics['fold_pct']:.1f}%")

        print(f"  Round-Specific Aggression:")
        print(f"    Round 1 (Pre-Flop):            {adv_metrics['round1_aggression']:.1f}% raises")
        print(f"    Round 2 (Post-Flop):           {adv_metrics['round2_aggression']:.1f}% raises")

        # Win rate by hand
        hand_stats = compute_win_rate_by_hand(metrics['trajectories'])
        print(f"  Performance by Starting Hand:")
        for card in ['K', 'Q', 'J']:
            if hand_stats[card]['total'] > 0:
                wins = hand_stats[card]['wins']
                total = hand_stats[card]['total']
                win_rate = hand_stats[card]['win_rate']
                avg_payoff = hand_stats[card]['avg_payoff']
                print(f"    {card}: {wins}/{total} wins ({win_rate:.1f}%) | "
                      f"Avg Payoff: {avg_payoff:+.2f}")


def compare_agents(results, agent1_name, agent2_name):
    """
    Direct comparison between two agents across all matchups.

    Args:
        results: Dictionary of evaluation results
        agent1_name: Name of first agent (e.g., 'DRQN')
        agent2_name: Name of second agent (e.g., 'DQN')
    """
    print('\n' + '=' * 60)
    print(f'HEAD-TO-HEAD COMPARISON: {agent1_name} vs {agent2_name}')
    print('=' * 60)

    # Find head-to-head matchup
    h2h_key = f'{agent1_name} vs {agent2_name}'
    if h2h_key in results:
        ev = results[h2h_key]['ev']
        print(f'\nDirect Matchup:')
        print(f"  {agent1_name} EV: {ev:+.3f} chips/hand")

        if ev > 0.1:
            print(f"  Result: {agent1_name} WINS decisively")
        elif ev > 0:
            print(f"  Result: {agent1_name} has slight edge")
        elif ev > -0.1:
            print(f"  Result: Nearly even match")
        else:
            print(f"  Result: {agent2_name} WINS")

    # Compare performance vs baselines
    print(f'\nPerformance vs Baselines:')

    baselines = ['Random', 'Heuristic']
    for baseline in baselines:
        key1 = f'{agent1_name} vs {baseline}'
        key2 = f'{agent2_name} vs {baseline}'

        if key1 in results and key2 in results:
            ev1 = results[key1]['ev']
            ev2 = results[key2]['ev']
            diff = ev1 - ev2

            print(f'\n  vs {baseline}:')
            print(f"    {agent1_name}: {ev1:+.3f}")
            print(f"    {agent2_name}: {ev2:+.3f}")
            print(f"    Difference: {diff:+.3f} ({agent1_name if diff > 0 else agent2_name} better)")

    # Compare playing styles
    print(f'\nPlaying Style Comparison:')

    style_keys = [f'{agent1_name} vs Random', f'{agent2_name} vs Random']
    if all(k in results and 'trajectories' in results[k] for k in style_keys):
        styles = []
        for key in style_keys:
            adv = compute_advanced_metrics(results[key]['trajectories'])
            styles.append(adv)

        print(f'\n  Metric                    {agent1_name:>15s}  {agent2_name:>15s}')
        print(f'  {"-"*55}')
        print(f'  VPIP                      {styles[0]["vpip"]:>14.1f}%  {styles[1]["vpip"]:>14.1f}%')
        print(f'  Pre-Flop Raise            {styles[0]["pfr"]:>14.1f}%  {styles[1]["pfr"]:>14.1f}%')
        print(f'  Aggression Factor         {styles[0]["af"]:>14.2f}   {styles[1]["af"]:>14.2f}')
        print(f'  Went to Showdown          {styles[0]["wtsd"]:>14.1f}%  {styles[1]["wtsd"]:>14.1f}%')
        print(f'  Round 1 Aggression        {styles[0]["round1_aggression"]:>14.1f}%  {styles[1]["round1_aggression"]:>14.1f}%')
        print(f'  Round 2 Aggression        {styles[0]["round2_aggression"]:>14.1f}%  {styles[1]["round2_aggression"]:>14.1f}%')


def statistical_significance(ev, num_hands, std_dev=2.0):
    """
    Calculate statistical significance of EV result.

    Args:
        ev: Expected value (chips/hand)
        num_hands: Number of hands played
        std_dev: Estimated standard deviation (default 2.0 for poker)

    Returns:
        dict: Statistical metrics including confidence interval and p-value
    """
    import math

    # Standard error
    se = std_dev / math.sqrt(num_hands)

    # 95% confidence interval
    ci_95 = 1.96 * se

    # Z-score (for null hypothesis that true EV = 0)
    z_score = ev / se if se > 0 else 0

    # Approximate p-value (two-tailed)
    # Using approximation: p ≈ 2 * (1 - Φ(|z|))
    if abs(z_score) > 6:
        p_value = 0.0
    else:
        from math import erf
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(z_score) / math.sqrt(2))))

    return {
        'ev': ev,
        'std_error': se,
        'ci_95_lower': ev - ci_95,
        'ci_95_upper': ev + ci_95,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
