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
        self.current_action_sequence = []
        self.current_reward_sequence = []

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

        self.current_action_sequence.append(action)
        self.current_reward_sequence.append(reward)

        # Mirror the sliding-window truncation applied to current_hand_sequence
        # in step()/eval_step(), so obs[t]/action[t]/reward[t] stay aligned.
        if self.max_sequence_length is not None:
            self.current_action_sequence = self.current_action_sequence[-self.max_sequence_length:]
            self.current_reward_sequence = self.current_reward_sequence[-self.max_sequence_length:]

        if done:
            if len(self.current_hand_sequence) > 0:
                self.replay_buffer.push(
                    self.current_hand_sequence,
                    self.current_action_sequence,
                    self.current_reward_sequence
                )
            # Reset for next hand
            self.current_hand_sequence = []
            self.current_action_sequence = []
            self.current_reward_sequence = []

    def train(self):
        if len(self.replay_buffer) < self.min_replay:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Pad sequences (episodes) to the same length within this batch
        lengths = [len(obs_seq) for obs_seq, _, _ in batch]
        max_len = max(lengths)
        feat = batch[0][0][0].shape[0]

        obs_padded = np.zeros((self.batch_size, max_len, feat), dtype=np.float32)
        actions_padded = np.zeros((self.batch_size, max_len), dtype=np.int64)
        rewards_padded = np.zeros((self.batch_size, max_len), dtype=np.float32)
        mask = np.zeros((self.batch_size, max_len), dtype=np.float32)      # 1 at real (non-padded) steps
        done_mask = np.zeros((self.batch_size, max_len), dtype=np.float32)  # 1 at each hand's final step

        for i, (obs_seq, action_seq, reward_seq) in enumerate(batch):
            T = len(obs_seq)
            obs_padded[i, :T] = np.array(obs_seq, dtype=np.float32)
            actions_padded[i, :T] = np.array(action_seq, dtype=np.int64)
            rewards_padded[i, :T] = np.array(reward_seq, dtype=np.float32)
            mask[i, :T] = 1.0
            done_mask[i, T - 1] = 1.0

        obs_t = torch.from_numpy(obs_padded).to(self.device)
        actions_t = torch.from_numpy(actions_padded).long().to(self.device)
        rewards_t = torch.from_numpy(rewards_padded).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)
        done_t = torch.from_numpy(done_mask).to(self.device)

        # Current Q-values at every timestep from the online network
        self.model.train()
        q_all, _ = self.model(obs_t, return_all=True)  # (B, L, A)
        current_q = q_all.gather(2, actions_t.unsqueeze(2)).squeeze(2)  # (B, L)

        # Bootstrapped TD target from the frozen target network:
        #   target(t) = r(t) + gamma * max_a Q_target(t+1)   (t is not the hand's last step)
        #             = r(t)                                  (t IS the hand's last step -> no bootstrap)
        with torch.no_grad():
            q_target_all, _ = self.target_model(obs_t, return_all=True)  # (B, L, A)
            max_next_q = q_target_all.max(dim=2).values  # (B, L)

            next_q = torch.zeros_like(max_next_q)
            next_q[:, :-1] = max_next_q[:, 1:]  # shift left: index t now holds Q(t+1)

            target_q = rewards_t + self.gamma * next_q * (1 - done_t)

        # Huber loss (more robust than MSE to outlier rewards), averaged only over real (non-padded) steps
        loss_per_step = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (loss_per_step * mask_t).sum() / mask_t.sum()

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