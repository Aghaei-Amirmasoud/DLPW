"""
Deep Recurrent Q-Network (DRQN) architecture for Leduc Hold'em
LSTM-based network for handling sequential decision making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeducDRQN(nn.Module):

    def __init__(self, state_shape, num_actions, hidden_size=64):
        """
        Args:
            state_shape: Dimension of state observation
            num_actions: Number of possible actions
            hidden_size: Size of hidden layers and LSTM
        """
        super().__init__()
        self.fc1 = nn.Linear(state_shape, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden_state=None):
        # x: (batch, seq_len, state_shape)
        batch, seq_len, feat = x.shape

        # Apply the linear layer across all timesteps efficiently
        x_flat = x.reshape(batch * seq_len, feat)
        x_flat = F.relu(self.fc1(x_flat))
        x = x_flat.reshape(batch, seq_len, -1)

        # LSTM processes the sequence
        lstm_out, hidden_state = self.lstm(x, hidden_state)

        # Only take the final timestep output
        last_step = lstm_out[:, -1, :]
        q_values = self.fc2(last_step)

        return q_values, hidden_state
