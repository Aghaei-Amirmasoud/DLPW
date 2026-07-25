"""
Sequence Replay Buffer for storing complete episode sequences
"""

import random
from collections import deque


class SequenceReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_sequence, action_sequence, reward_sequence):
        """
        Store one full episode (hand). All three sequences share the same
        length T = number of decisions the agent made in that hand:
          obs_sequence[t]    -> observation the agent saw before deciding at t
          action_sequence[t] -> action taken at t
          reward_sequence[t] -> reward received after acting at t (0 except
                                 at the terminal step, which holds the payoff)
        """
        self.buffer.append((list(obs_sequence), list(action_sequence), list(reward_sequence)))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)