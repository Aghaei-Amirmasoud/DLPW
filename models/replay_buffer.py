"""
Sequence Replay Buffer for storing complete episode sequences
"""

import random
from collections import deque


class SequenceReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_sequence, action, reward):
        self.buffer.append((list(obs_sequence), action, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
