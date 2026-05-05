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

    Supports:
    - Self-play training
    - Multi-task learning (mixed opponents)
    - Learning rate scheduling
    """

    def __init__(self, env, agent, opponent_phase1, opponent_phase2,
                 use_self_play=False, multi_task=False, multi_task_ratio=0.5):
        """
        Args:
            env: RLCard environment
            agent: Agent to train (DRQN or DQN)
            opponent_phase1: Weak opponent for Phase 1 (typically Random)
            opponent_phase2: Stronger opponent for Phase 2 (typically Heuristic)
            use_self_play: If True, use self-play in Phase 2
            multi_task: If True, train vs both opponents simultaneously
            multi_task_ratio: Ratio of opponent_phase1 vs opponent_phase2 (0.5 = 50/50)
        """
        self.env = env
        self.agent = agent
        self.opponent_phase1 = opponent_phase1
        self.opponent_phase2 = opponent_phase2
        self.use_self_play = use_self_play
        self.multi_task = multi_task
        self.multi_task_ratio = multi_task_ratio

        # Self-play opponent (copy of agent)
        self.self_play_opponent = None
        self.self_play_update_freq = 500  # Update self-play opponent every N episodes

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
        Phase 1: Train against Random agent (or mixed if multi-task).

        Args:
            num_episodes: Number of training episodes
            eval_every: Evaluate every N episodes
            eval_num: Number of evaluation hands
        """
        print("=" * 60)
        if self.multi_task:
            print(f"PHASE 1: Multi-Task Training (Random {self.multi_task_ratio:.0%} + "
                  f"Heuristic {1-self.multi_task_ratio:.0%}) (0-{num_episodes})")
        else:
            print(f"PHASE 1: Training vs Random Agent (0-{num_episodes})")
        print("=" * 60)

        self.env.set_agents([self.agent, self.opponent_phase1])

        for episode in range(num_episodes):
            # Multi-task: randomly select opponent
            if self.multi_task:
                import random
                opponent = self.opponent_phase1 if random.random() < self.multi_task_ratio else self.opponent_phase2
                loss = self.train_episode(opponent=opponent)
            else:
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

    def train_phase2(self, start_episode, num_episodes, eval_every, eval_num, new_lr=None):
        """
        Phase 2: Train against Heuristic agent (or self-play, or mixed).

        Args:
            start_episode: Starting episode number (for continuity)
            num_episodes: Number of training episodes
            eval_every: Evaluate every N episodes
            eval_num: Number of evaluation hands
            new_lr: New learning rate for Phase 2 (optional)
        """
        print()
        print("=" * 60)

        # Update learning rate if provided
        if new_lr is not None and hasattr(self.agent, 'update_learning_rate'):
            print(f"PHASE 2: Fine-tuning with Lower LR ({start_episode}-{start_episode + num_episodes})")
            self.agent.update_learning_rate(new_lr)

        # Setup opponent
        if self.use_self_play:
            print(f"PHASE 2: Self-Play Training ({start_episode}-{start_episode + num_episodes})")
            import copy
            self.self_play_opponent = copy.deepcopy(self.agent)
            training_opponent = self.self_play_opponent
        elif self.multi_task:
            print(f"PHASE 2: Multi-Task Training (continued) ({start_episode}-{start_episode + num_episodes})")
            training_opponent = None  # Will be selected randomly
        else:
            print(f"PHASE 2: Fine-tuning vs Heuristic ({start_episode}-{start_episode + num_episodes})")
            training_opponent = self.opponent_phase2

        print("=" * 60)

        if not self.multi_task:
            self.env.set_agents([self.agent, training_opponent])

        for episode in range(start_episode, start_episode + num_episodes):
            # Multi-task: randomly select opponent
            if self.multi_task:
                import random
                opponent = self.opponent_phase1 if random.random() < self.multi_task_ratio else self.opponent_phase2
                loss = self.train_episode(opponent=opponent)
            # Self-play: update opponent periodically
            elif self.use_self_play:
                if episode % self.self_play_update_freq == 0 and episode > start_episode:
                    import copy
                    self.self_play_opponent = copy.deepcopy(self.agent)
                    self.env.set_agents([self.agent, self.self_play_opponent])
                    print(f"  [Self-play opponent updated at episode {episode}]")
                loss = self.train_episode()
            else:
                loss = self.train_episode()

            if loss is not None and LOG_LOSS:
                self.loss_history.append((episode, loss))

            if episode % eval_every == 0:
                ev_random = self.evaluate(self.eval_random, eval_num)
                ev_heuristic = self.evaluate(self.opponent_phase2, eval_num)

                self.ev_history_random.append((episode, ev_random))
                self.ev_history_heuristic.append((episode, ev_heuristic))

                epsilon = getattr(self.agent, 'epsilon', 0.0)
                phase_label = "SP" if self.use_self_play else "MT" if self.multi_task else "P2"
                print(f"[{phase_label}] Ep {episode:05d} | ε={epsilon:.3f} | "
                      f"EV vs Random: {ev_random:+.3f} | "
                      f"EV vs Heuristic: {ev_heuristic:+.3f}")

                # Restore training opponent
                if not self.multi_task:
                    self.env.set_agents([self.agent, training_opponent])

    def train(self):
        """
        Run complete two-phase curriculum training.

        Returns:
            dict: Training history (EV and loss)
        """
        from config import LEARNING_RATE_PHASE2

        self.train_phase1(NUM_EPISODES_PHASE1, EVALUATE_EVERY, EVALUATE_NUM)
        self.train_phase2(NUM_EPISODES_PHASE1, NUM_EPISODES_PHASE2,
                         EVALUATE_EVERY, EVALUATE_NUM, new_lr=LEARNING_RATE_PHASE2)

        print()
        print("--- TRAINING COMPLETE ---")

        return {
            'ev_random': self.ev_history_random,
            'ev_heuristic': self.ev_history_heuristic,
            'loss': self.loss_history
        }
