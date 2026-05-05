"""
Conservative Heuristic Agent for Leduc Hold'em
Rule-based agent that bets purely on card strength - no bluffing, no memory
"""


class ConservativeHeuristicAgent:
    """
    Bets strictly on card strength. No memory, no bluffing.

    Strategy:
    - If paired with community card: aggressive (raise/call)
    - King (K): aggressive raise/call
    - Queen (Q): cautious call/check, fold if facing bet
    - Jack (J): check/fold
    """

    def __init__(self, num_actions):
        """
        Args:
            num_actions: Number of possible actions
        """
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        """
        Training mode step (same as eval_step for heuristic).

        Args:
            state: Current game state

        Returns:
            action: Selected action
        """
        # Unpack the tuple: grab the action, throw away the empty dictionary
        action, _ = self.eval_step(state)
        return action

    def eval_step(self, state):
        """
        Evaluation mode step - select action based on heuristic rules.

        Args:
            state: Current game state

        Returns:
            (action, info_dict): Selected action and empty info dictionary
        """
        raw_obs = state['raw_obs']
        legal_actions = raw_obs['legal_actions']

        # Get private card rank
        raw_hand = raw_obs['hand']
        hand_str = raw_hand[0] if isinstance(raw_hand, list) else raw_hand
        rank = hand_str[-1]

        # Get public card rank (if revealed)
        raw_public = raw_obs['public_card']
        public_rank = None
        if raw_public:
            pub_str = raw_public[0] if isinstance(raw_public, list) else raw_public
            public_rank = pub_str[-1]

        # Decision logic with public card (Round 2)
        if public_rank:
            # Pair: always aggressive
            if public_rank == rank:
                if 'raise' in legal_actions:
                    return 'raise', {}
                if 'call' in legal_actions:
                    return 'call', {}

            # King: aggressive
            if rank == 'K':
                if 'raise' in legal_actions:
                    return 'raise', {}
                if 'call' in legal_actions:
                    return 'call', {}

            # Queen: cautious
            if rank == 'Q':
                if 'check' not in legal_actions:
                    if 'fold' in legal_actions:
                        return 'fold', {}
                if 'check' in legal_actions:
                    return 'check', {}
                if 'call' in legal_actions:
                    return 'call', {}

            # Jack: defensive
            if rank == 'J':
                if 'check' in legal_actions:
                    return 'check', {}
                if 'fold' in legal_actions:
                    return 'fold', {}

        # Decision logic without public card (Round 1)
        else:
            if rank == 'K':
                if 'raise' in legal_actions:
                    return 'raise', {}
                if 'call' in legal_actions:
                    return 'call', {}

            if rank == 'Q':
                if 'check' in legal_actions:
                    return 'check', {}
                if 'call' in legal_actions:
                    return 'call', {}

            if rank == 'J':
                if 'check' in legal_actions:
                    return 'check', {}
                if 'fold' in legal_actions:
                    return 'fold', {}

        # Fallback: return first legal action
        return legal_actions[0], {}
