"""
DRQN Agent with sequence replay buffer and target network
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from models.drqn import LeducDRQN
from models.replay_buffer import SequenceReplayBuffer


class DRQNAgent:
    def __init__(self, state_shape, num_actions, device,
                 hidden_size=64, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 buffer_capacity=5000, batch_size=64, min_replay=256,
                 target_update_freq=50, l2_reg=0.0, max_sequence_length=None):

        self.use_raw = False
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay = min_replay
        self.target_update_freq = target_update_freq
        self.max_sequence_length = max_sequence_length
        self._train_steps = 0

        # Online network (trained every step)
        self.model = LeducDRQN(state_shape, num_actions, hidden_size).to(device)

        # Target network (frozen, synced periodically for stable Q-targets)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.lr = lr

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = SequenceReplayBuffer(buffer_capacity)
        self.current_hand_sequence = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seq_to_tensor(self, sequence):
        arr = np.array(sequence, dtype=np.float32)  # avoids slow-list warning
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, T, feat)

    def _greedy_action(self, legal_actions):
        seq_t = self._seq_to_tensor(self.current_hand_sequence)
        with torch.no_grad():
            q_values, _ = self.model(seq_t)
            q_values = q_values.cpu().numpy()[0]

        # Mask illegal actions
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_values[a]

        return int(np.argmax(masked))

    # ------------------------------------------------------------------
    # RLCard interface
    # ------------------------------------------------------------------

    def step(self, state):
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        self.current_hand_sequence.append(obs)

        # Truncate sequence if max length specified
        if self.max_sequence_length is not None and len(self.current_hand_sequence) > self.max_sequence_length:
            self.current_hand_sequence = self.current_hand_sequence[-self.max_sequence_length:]

        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)

        return self._greedy_action(legal_actions)

    def eval_step(self, state):
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        self.current_hand_sequence.append(obs)

        # Truncate sequence if max length specified
        if self.max_sequence_length is not None and len(self.current_hand_sequence) > self.max_sequence_length:
            self.current_hand_sequence = self.current_hand_sequence[-self.max_sequence_length:]

        return self._greedy_action(legal_actions), {}

    def feed(self, transition):
        state, action, reward, next_state, done = transition

        if done and len(self.current_hand_sequence) > 0:
            self.replay_buffer.push(self.current_hand_sequence, action, reward)
            self.current_hand_sequence = []  # reset for next hand

    def train(self):
        if len(self.replay_buffer) < self.min_replay:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Pad sequences to the same length within this batch
        max_len = max(len(seq) for seq, _, _ in batch)
        feat = batch[0][0][0].shape[0]

        padded = np.zeros((self.batch_size, max_len, feat), dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.int64)
        rewards = np.zeros(self.batch_size, dtype=np.float32)

        for i, (seq, action, reward) in enumerate(batch):
            seq_arr = np.array(seq, dtype=np.float32)
            padded[i, :len(seq)] = seq_arr
            actions[i] = action
            rewards[i] = reward

        states_t = torch.from_numpy(padded).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)

        # Current Q-values from online network
        self.model.train()
        current_q, _ = self.model(states_t)  # (B, num_actions)
        current_q = current_q.gather(1, actions_t).squeeze(1)  # (B,)

        # Target: terminal transitions -> target = reward only (no next-state bootstrap)
        target_q = rewards_t

        # Huber loss (more robust than MSE to outlier rewards)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Sync target network periodically
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self._train_steps
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self._train_steps = checkpoint['train_steps']
