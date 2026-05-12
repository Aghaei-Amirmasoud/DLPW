"""
Two-phase curriculum training for DRQN and DQN agents
"""

import rlcard
from rlcard.utils import reorganize
from rlcard.agents import RandomAgent

from config import *


class CurriculumTrainer:
    """
    Two-phase curriculum training:
    Phase 1: Train vs Random Agent (learn basic card strength)
    Phase 2: Train vs Heuristic Agent (learn to exploit predictable play)
    """

    def __init__(self, env, agent, opponent_phase1, opponent_phase2):
        """
        Args:
            env: RLCard environment
            agent: Agent to train (DRQN or DQN)
            opponent_phase1: Weak opponent for Phase 1 (typically Random)
            opponent_phase2: Stronger opponent for Phase 2 (typically Heuristic)
        """
        self.env = env
        self.agent = agent
        self.opponent_phase1 = opponent_phase1
        self.opponent_phase2 = opponent_phase2

        # Evaluation opponents
        self.eval_random = RandomAgent(env.num_actions)

        # Training history
        self.ev_history_random = []
        self.ev_history_heuristic = []
        self.loss_history = []

    def train_episode(self, opponent=None):
        """
        Run one training episode.

        Args:
            opponent: Opponent to train against (if None, uses current opponent)

        Returns:
            float or None: Loss value if training occurred
        """
        if opponent is not None:
            self.env.set_agents([self.agent, opponent])

        trajectories, payoffs = self.env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:
            self.agent.feed(ts)

        # Try to train - handle case where buffer not full yet
        try:
            loss = self.agent.train()
        except (ValueError, IndexError):
            # Buffer not full yet (RLCard DQN agent issue)
            loss = None

        return loss

    def evaluate(self, opponent, num_hands=1000):
        """
        Evaluate agent against an opponent.

        Args:
            opponent: Opponent agent
            num_hands: Number of hands to play

        Returns:
            float: Average EV (chips/hand)
        """
        self.env.set_agents([self.agent, opponent])
        total = 0

        for _ in range(num_hands):
            # Reset hand sequence for DRQN
            if hasattr(self.agent, 'current_hand_sequence'):
                self.agent.current_hand_sequence = []

            _, payoffs = self.env.run(is_training=False)
            total += payoffs[0]

        return total / num_hands

    def train_phase1(self, num_episodes, eval_every, eval_num):
        """
        Phase 1: Train against Random agent.

        Args:
            num_episodes: Number of training episodes
            eval_every: Evaluate every N episodes
            eval_num: Number of evaluation hands
        """
        print("=" * 60)
        print(f"PHASE 1: Training vs Random Agent (0-{num_episodes})")
        print("=" * 60)

        self.env.set_agents([self.agent, self.opponent_phase1])

        for episode in range(num_episodes):
            loss = self.train_episode()

            if loss is not None and LOG_LOSS:
                self.loss_history.append((episode, loss))

            if episode % eval_every == 0:
                ev_random = self.evaluate(self.eval_random, eval_num)
                ev_heuristic = self.evaluate(self.opponent_phase2, eval_num)

                self.ev_history_random.append((episode, ev_random))
                self.ev_history_heuristic.append((episode, ev_heuristic))

                epsilon = getattr(self.agent, 'epsilon', 0.0)
                print(f"[P1] Ep {episode:05d} | ε={epsilon:.3f} | "
                      f"EV vs Random: {ev_random:+.3f} | "
                      f"EV vs Heuristic: {ev_heuristic:+.3f}")

                # Restore training opponent
                self.env.set_agents([self.agent, self.opponent_phase1])

    def train_phase2(self, start_episode, num_episodes, eval_every, eval_num):
        """
        Phase 2: Train against Heuristic agent.

        Args:
            start_episode: Starting episode number (for continuity)
            num_episodes: Number of training episodes
            eval_every: Evaluate every N episodes
            eval_num: Number of evaluation hands
        """
        print()
        print("=" * 60)
        print(f"PHASE 2: Fine-tuning vs Heuristic ({start_episode}-{start_episode + num_episodes})")
        print("=" * 60)

        # Reset epsilon if configured (to encourage exploration of Heuristic-specific strategies)
        if EPSILON_RESET_PHASE2 is not None:
            if hasattr(self.agent, 'epsilon'):
                old_epsilon = self.agent.epsilon
                self.agent.epsilon = EPSILON_RESET_PHASE2
                print(f"[Phase 2] Epsilon reset: {old_epsilon:.3f} → {EPSILON_RESET_PHASE2:.3f}")
                print(f"[Phase 2] Rationale: Encourage exploration of new strategies vs Heuristic")
                print()

        self.env.set_agents([self.agent, self.opponent_phase2])

        for episode in range(start_episode, start_episode + num_episodes):
            loss = self.train_episode()

            if loss is not None and LOG_LOSS:
                self.loss_history.append((episode, loss))

            if episode % eval_every == 0:
                ev_random = self.evaluate(self.eval_random, eval_num)
                ev_heuristic = self.evaluate(self.opponent_phase2, eval_num)

                self.ev_history_random.append((episode, ev_random))
                self.ev_history_heuristic.append((episode, ev_heuristic))

                epsilon = getattr(self.agent, 'epsilon', 0.0)
                print(f"[P2] Ep {episode:05d} | ε={epsilon:.3f} | "
                      f"EV vs Random: {ev_random:+.3f} | "
                      f"EV vs Heuristic: {ev_heuristic:+.3f}")

                # Restore training opponent
                self.env.set_agents([self.agent, self.opponent_phase2])

    def train(self):
        """
        Run complete two-phase curriculum training.

        Returns:
            dict: Training history (EV and loss)
        """

        self.train_phase1(NUM_EPISODES_PHASE1, EVALUATE_EVERY, EVALUATE_NUM)
        self.train_phase2(NUM_EPISODES_PHASE1, NUM_EPISODES_PHASE2,
                         EVALUATE_EVERY, EVALUATE_NUM)

        print()
        print("--- TRAINING COMPLETE ---")

        return {
            'ev_random': self.ev_history_random,
            'ev_heuristic': self.ev_history_heuristic,
            'loss': self.loss_history
        }
