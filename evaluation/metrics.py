"""
Evaluation metrics for poker agents
"""

def evaluate_agents(env, agent_a, agent_b, num_hands=3000, label=''):
    env.set_agents([agent_a, agent_b])
    total = 0
    trajectories_collected = []

    for _ in range(num_hands):
        # Reset hand sequence for DRQN agents
        if hasattr(agent_a, 'current_hand_sequence'):
            agent_a.current_hand_sequence = []

        trajectories, payoffs = env.run(is_training=False)
        total += payoffs[0]
        # Store trajectory WITH payoff so we can track wins/losses by hand
        trajectories_collected.append((trajectories[0], payoffs[0]))

    avg_ev = total / num_hands

    if label:
        print(f'  {label:30s}: {avg_ev:+.3f} chips/hand')

    return avg_ev, trajectories_collected


def compute_action_distribution(trajectories, action_names):
    action_counts = {name: 0 for name in action_names.values()}
    bluff_attempts = 0
    total_jack_steps = 0

    for traj_data in trajectories:
        # Handle both old format (just trajectory) and new format (trajectory, payoff)
        if isinstance(traj_data, tuple):
            hand_trajectory, _ = traj_data
        else:
            hand_trajectory = traj_data

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

    for traj_data in trajectories:
        # Handle both old format (just trajectory) and new format (trajectory, payoff)
        if isinstance(traj_data, tuple):
            hand_trajectory, _ = traj_data
        else:
            hand_trajectory = traj_data
        hand_actions = []
        hand_card = None
        current_round = 1
        went_to_showdown = False

        # Track actions counted for this hand to avoid double-counting
        hand_action_counts = {'raise': 0, 'call': 0, 'fold': 0, 'check': 0}
        hand_round1_actions = {'raise': 0, 'call': 0, 'fold': 0, 'check': 0}
        hand_round2_actions = {'raise': 0, 'call': 0, 'fold': 0, 'check': 0}

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

                # Count each action type once per occurrence in this hand
                hand_action_counts[action] = hand_action_counts.get(action, 0) + 1

                # Round-specific tracking for this hand
                if current_round == 1:
                    hand_round1_actions[action] = hand_round1_actions.get(action, 0) + 1
                else:
                    hand_round2_actions[action] = hand_round2_actions.get(action, 0) + 1

        # Add this hand's action counts to totals
        total_raises += hand_action_counts['raise']
        total_calls += hand_action_counts['call']
        total_folds += min(hand_action_counts['fold'], 1)  # At most 1 fold per hand
        total_checks += hand_action_counts['check']

        # Add round-specific counts
        for action in ['raise', 'call', 'fold', 'check']:
            round1_actions[action] += hand_round1_actions[action]
            round2_actions[action] += hand_round2_actions[action]

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
    fold_pct = 100 * total_folds / max(total_hands, 1)  # Fixed: folds per hand, not per action

    return {
        'vpip': vpip,  # Voluntary Put in Pot %
        'pfr': pfr,    # Pre-Flop Raise %
        'af': aggression_factor,  # Aggression Factor
        'wtsd': wtsd,  # Went to Showdown %
        'fold_pct': fold_pct,
        'round1_aggression': 100 * round1_actions['raise'] / max(sum(round1_actions.values()), 1),
        'round2_aggression': 100 * round2_actions['raise'] / max(sum(round2_actions.values()), 1),
        'jack_hands': jack_hands,
        'queen_hands': queen_hands,
        'king_hands': king_hands,
        'total_hands': total_hands
    }


def compute_win_rate_by_hand(trajectories):
    hand_results = {
        'J': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'payoff_sum': 0},
        'Q': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'payoff_sum': 0},
        'K': {'wins': 0, 'losses': 0, 'ties': 0, 'total': 0, 'payoff_sum': 0}
    }

    for traj_data in trajectories:
        # Unpack trajectory and payoff
        if isinstance(traj_data, tuple):
            hand_trajectory, payoff = traj_data
        else:
            # Fallback for old format
            hand_trajectory = traj_data
            payoff = 0

        hand_card = None

        # Get hand card from first state with hand info
        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})

            # Get hand card (only need to do this once per hand)
            if hand_card is None:
                raw_hand = raw_obs.get('hand', '')
                if raw_hand:
                    # Handle both string and list formats
                    if isinstance(raw_hand, list):
                        hand_str = raw_hand[0] if raw_hand else ''
                    else:
                        hand_str = raw_hand

                    if hand_str:
                        hand_card = hand_str[-1]  # Get rank (J, Q, or K)
                        break  # Found it, no need to continue

        # Record results
        if hand_card in hand_results:
            hand_results[hand_card]['total'] += 1
            hand_results[hand_card]['payoff_sum'] += payoff

            if payoff > 0:
                hand_results[hand_card]['wins'] += 1
            elif payoff < 0:
                hand_results[hand_card]['losses'] += 1
            else:
                hand_results[hand_card]['ties'] += 1

    # Calculate win rates and average payoffs
    for card in hand_results:
        total = hand_results[card]['total']
        if total > 0:
            hand_results[card]['win_rate'] = 100 * hand_results[card]['wins'] / total
            hand_results[card]['avg_payoff'] = hand_results[card]['payoff_sum'] / total
        else:
            hand_results[card]['win_rate'] = 0
            hand_results[card]['avg_payoff'] = 0

    return hand_results


def print_evaluation_summary(results):
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


def analyze_queen_play(trajectories):
    queen_stats = {
        'total_hands': 0,
        'round1_raise': 0,
        'round1_call': 0,
        'round1_check': 0,
        'round1_fold': 0,
        'round2_raise': 0,
        'round2_call': 0,
        'round2_check': 0,
        'round2_fold': 0,
        'won': 0,
        'lost': 0,
        'tied': 0,
        'total_payoff': 0,
        'folded_queens': 0,  # Times folded with Queen
        'showdown_queens': 0  # Times went to showdown with Queen
    }

    for traj_data in trajectories:
        if isinstance(traj_data, tuple):
            hand_trajectory, payoff = traj_data
        else:
            hand_trajectory = traj_data
            payoff = 0

        # Check if this hand has a Queen
        hand_card = None
        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue
            raw_obs = state.get('raw_obs', {})
            if hand_card is None:
                raw_hand = raw_obs.get('hand', '')
                if raw_hand:
                    if isinstance(raw_hand, list):
                        hand_str = raw_hand[0] if raw_hand else ''
                    else:
                        hand_str = raw_hand
                    if hand_str:
                        hand_card = hand_str[-1]
                        break

        if hand_card != 'Q':
            continue  # Not a Queen hand

        queen_stats['total_hands'] += 1
        queen_stats['total_payoff'] += payoff

        if payoff > 0:
            queen_stats['won'] += 1
        elif payoff < 0:
            queen_stats['lost'] += 1
        else:
            queen_stats['tied'] += 1

        # Track actions by round
        current_round = 1
        folded = False

        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})
            action_record = state.get('action_record', [])

            # Detect round change
            if raw_obs.get('public_card'):
                current_round = 2

            # Get player 0's actions
            player0_actions = [a for pid, a in action_record if pid == 0]
            if player0_actions:
                action = player0_actions[-1]

                if current_round == 1:
                    if action == 'raise':
                        queen_stats['round1_raise'] += 1
                    elif action == 'call':
                        queen_stats['round1_call'] += 1
                    elif action == 'check':
                        queen_stats['round1_check'] += 1
                    elif action == 'fold':
                        queen_stats['round1_fold'] += 1
                        folded = True
                else:  # Round 2
                    if action == 'raise':
                        queen_stats['round2_raise'] += 1
                    elif action == 'call':
                        queen_stats['round2_call'] += 1
                    elif action == 'check':
                        queen_stats['round2_check'] += 1
                    elif action == 'fold':
                        queen_stats['round2_fold'] += 1
                        folded = True

        if folded:
            queen_stats['folded_queens'] += 1
        else:
            queen_stats['showdown_queens'] += 1

    # Calculate percentages
    total = queen_stats['total_hands']
    if total > 0:
        queen_stats['win_rate'] = 100 * queen_stats['won'] / total
        queen_stats['avg_payoff'] = queen_stats['total_payoff'] / total
        queen_stats['fold_rate'] = 100 * queen_stats['folded_queens'] / total
        queen_stats['showdown_rate'] = 100 * queen_stats['showdown_queens'] / total

        # Round 1 percentages
        r1_total = (queen_stats['round1_raise'] + queen_stats['round1_call'] +
                    queen_stats['round1_check'] + queen_stats['round1_fold'])
        if r1_total > 0:
            queen_stats['round1_raise_pct'] = 100 * queen_stats['round1_raise'] / r1_total
            queen_stats['round1_call_pct'] = 100 * queen_stats['round1_call'] / r1_total
            queen_stats['round1_check_pct'] = 100 * queen_stats['round1_check'] / r1_total
            queen_stats['round1_fold_pct'] = 100 * queen_stats['round1_fold'] / r1_total

        # Round 2 percentages
        r2_total = (queen_stats['round2_raise'] + queen_stats['round2_call'] +
                    queen_stats['round2_check'] + queen_stats['round2_fold'])
        if r2_total > 0:
            queen_stats['round2_raise_pct'] = 100 * queen_stats['round2_raise'] / r2_total
            queen_stats['round2_call_pct'] = 100 * queen_stats['round2_call'] / r2_total
            queen_stats['round2_check_pct'] = 100 * queen_stats['round2_check'] / r2_total
            queen_stats['round2_fold_pct'] = 100 * queen_stats['round2_fold'] / r2_total

    return queen_stats


def analyze_round_performance(trajectories):
    round_stats = {
        'hands_folded_round1': 0,
        'hands_folded_round2': 0,
        'hands_to_showdown': 0,
        'total_hands': len(trajectories),
        'round1_aggressive_wins': 0,  # Won after raising in Round 1
        'round1_aggressive_total': 0,
        'round2_aggressive_wins': 0,  # Won after raising in Round 2
        'round2_aggressive_total': 0
    }

    for traj_data in trajectories:
        if isinstance(traj_data, tuple):
            hand_trajectory, payoff = traj_data
        else:
            hand_trajectory = traj_data
            payoff = 0

        current_round = 1
        folded_round = None
        raised_round1 = False
        raised_round2 = False

        for state in hand_trajectory:
            if not isinstance(state, dict):
                continue

            raw_obs = state.get('raw_obs', {})
            action_record = state.get('action_record', [])

            # Detect round change
            if raw_obs.get('public_card'):
                current_round = 2

            # Get player 0's actions
            player0_actions = [a for pid, a in action_record if pid == 0]
            if player0_actions:
                action = player0_actions[-1]

                if action == 'fold':
                    if folded_round is None:
                        folded_round = current_round

                if action == 'raise':
                    if current_round == 1:
                        raised_round1 = True
                    else:
                        raised_round2 = True

        # Record results
        if folded_round == 1:
            round_stats['hands_folded_round1'] += 1
        elif folded_round == 2:
            round_stats['hands_folded_round2'] += 1
        else:
            round_stats['hands_to_showdown'] += 1

        # Track aggressive play success
        if raised_round1:
            round_stats['round1_aggressive_total'] += 1
            if payoff > 0:
                round_stats['round1_aggressive_wins'] += 1

        if raised_round2:
            round_stats['round2_aggressive_total'] += 1
            if payoff > 0:
                round_stats['round2_aggressive_wins'] += 1

    # Calculate percentages
    total = round_stats['total_hands']
    if total > 0:
        round_stats['fold_round1_pct'] = 100 * round_stats['hands_folded_round1'] / total
        round_stats['fold_round2_pct'] = 100 * round_stats['hands_folded_round2'] / total
        round_stats['showdown_pct'] = 100 * round_stats['hands_to_showdown'] / total

        if round_stats['round1_aggressive_total'] > 0:
            round_stats['round1_aggressive_win_rate'] = (
                100 * round_stats['round1_aggressive_wins'] / round_stats['round1_aggressive_total']
            )

        if round_stats['round2_aggressive_total'] > 0:
            round_stats['round2_aggressive_win_rate'] = (
                100 * round_stats['round2_aggressive_wins'] / round_stats['round2_aggressive_total']
            )

    return round_stats


def compare_queen_play(results, agent1_name, agent2_name):
    print('\n' + '=' * 60)
    print(f'QUEEN PLAY COMPARISON: {agent1_name} vs {agent2_name}')
    print('=' * 60)

    # Find matchup vs Heuristic for both agents
    key1 = f'{agent1_name} vs Heuristic'
    key2 = f'{agent2_name} vs Heuristic'

    if key1 not in results or key2 not in results:
        print("Error: Matchups vs Heuristic not found")
        return

    if 'trajectories' not in results[key1] or 'trajectories' not in results[key2]:
        print("Error: Trajectories not available")
        return

    # Analyze Queen play for both agents
    queen1 = analyze_queen_play(results[key1]['trajectories'])
    queen2 = analyze_queen_play(results[key2]['trajectories'])

    print(f'\n{agent1_name} Queens vs Heuristic:')
    print(f"  Total Queen Hands: {queen1['total_hands']}")
    print(f"  Win Rate: {queen1.get('win_rate', 0):.1f}%")
    print(f"  Average Payoff: {queen1.get('avg_payoff', 0):+.3f}")
    print(f"  Fold Rate: {queen1.get('fold_rate', 0):.1f}%")
    print(f"  Showdown Rate: {queen1.get('showdown_rate', 0):.1f}%")

    print(f'\n  Round 1 Actions:')
    print(f"    Raise:  {queen1.get('round1_raise_pct', 0):.1f}%")
    print(f"    Call:   {queen1.get('round1_call_pct', 0):.1f}%")
    print(f"    Check:  {queen1.get('round1_check_pct', 0):.1f}%")
    print(f"    Fold:   {queen1.get('round1_fold_pct', 0):.1f}%")

    print(f'\n  Round 2 Actions:')
    print(f"    Raise:  {queen1.get('round2_raise_pct', 0):.1f}%")
    print(f"    Call:   {queen1.get('round2_call_pct', 0):.1f}%")
    print(f"    Check:  {queen1.get('round2_check_pct', 0):.1f}%")
    print(f"    Fold:   {queen1.get('round2_fold_pct', 0):.1f}%")

    print(f'\n{agent2_name} Queens vs Heuristic:')
    print(f"  Total Queen Hands: {queen2['total_hands']}")
    print(f"  Win Rate: {queen2.get('win_rate', 0):.1f}%")
    print(f"  Average Payoff: {queen2.get('avg_payoff', 0):+.3f}")
    print(f"  Fold Rate: {queen2.get('fold_rate', 0):.1f}%")
    print(f"  Showdown Rate: {queen2.get('showdown_rate', 0):.1f}%")

    print(f'\n  Round 1 Actions:')
    print(f"    Raise:  {queen2.get('round1_raise_pct', 0):.1f}%")
    print(f"    Call:   {queen2.get('round1_call_pct', 0):.1f}%")
    print(f"    Check:  {queen2.get('round1_check_pct', 0):.1f}%")
    print(f"    Fold:   {queen2.get('round1_fold_pct', 0):.1f}%")

    print(f'\n  Round 2 Actions:')
    print(f"    Raise:  {queen2.get('round2_raise_pct', 0):.1f}%")
    print(f"    Call:   {queen2.get('round2_call_pct', 0):.1f}%")
    print(f"    Check:  {queen2.get('round2_check_pct', 0):.1f}%")
    print(f"    Fold:   {queen2.get('round2_fold_pct', 0):.1f}%")

    # Key differences
    print(f'\n' + '=' * 60)
    print('KEY DIFFERENCES:')
    print('=' * 60)

    win_diff = queen1.get('win_rate', 0) - queen2.get('win_rate', 0)
    print(f"\nWin Rate Gap: {win_diff:+.1f} percentage points ({agent1_name})")

    r1_raise_diff = queen1.get('round1_raise_pct', 0) - queen2.get('round1_raise_pct', 0)
    r2_raise_diff = queen1.get('round2_raise_pct', 0) - queen2.get('round2_raise_pct', 0)

    print(f"Round 1 Raise Rate: {r1_raise_diff:+.1f}% ({agent1_name})")
    print(f"Round 2 Raise Rate: {r2_raise_diff:+.1f}% ({agent1_name})")

    if abs(r2_raise_diff) > 10:
        print(f"\n⚠️  Large Round 2 aggression difference detected!")
        print(f"    {agent1_name} raises {queen1.get('round2_raise_pct', 0):.1f}% in Round 2")
        print(f"    {agent2_name} raises {queen2.get('round2_raise_pct', 0):.1f}% in Round 2")
