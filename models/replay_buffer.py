"""
Sequence Replay Buffer for storing complete episode sequences
"""

import random
from collections import deque


class SequenceReplayBuffer:
    """
    Stores complete hand sequences: (obs_sequence, action, reward).
    Uses deque(maxlen=N) for O(1) eviction.

    v1 bug fix: Instead of storing isolated (s, a, r) transitions, we store
    the full episode sequence per hand. During training we pad sequences to
    the same length within a batch.
    """

    def __init__(self, capacity=5000):
        """
        Args:
            capacity: Maximum number of sequences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_sequence, action, reward):
        """
        Add a complete episode sequence to the buffer.

        Args:
            obs_sequence: List of observations from the episode
            action: Final action taken
            reward: Final reward received
        """
        self.buffer.append((list(obs_sequence), action, reward))

    def sample(self, batch_size):
        """
        Sample a random batch of sequences.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            List of (obs_sequence, action, reward) tuples
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
