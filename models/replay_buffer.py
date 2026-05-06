"""
Sequence Replay Buffer for DRQN with proper TD learning
"""

import random
from collections import deque


class SequenceReplayBuffer:
    """
    Stores episode trajectories as sequences of transitions.
    Each episode is a list of (obs_history, action, reward, next_obs_history, done).

    For DRQN, we need to maintain observation history for the LSTM while
    using proper TD targets with bootstrapping.
    """

    def __init__(self, capacity=5000):
        """
        Args:
            capacity: Maximum number of episodes to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, episode_transitions):
        """
        Add a complete episode (list of transitions) to the buffer.

        Args:
            episode_transitions: List of (obs_history, action, reward, next_obs_history, done) tuples
        """
        if len(episode_transitions) > 0:
            self.buffer.append(episode_transitions)

    def sample(self, batch_size):
        """
        Sample a random batch of episodes.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            List of episodes (each episode is a list of transitions)
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        """Return current buffer size (number of episodes)."""
        return len(self.buffer)
