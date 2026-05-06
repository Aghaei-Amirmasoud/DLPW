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
    """
    Deep Recurrent Q-Network Agent for Leduc Hold'em.

    Key features:
    - Proper sequence training: batch is padded sequences, not fake seq-len-1 tensors
    - Target network: frozen copy synced every target_update_freq gradient steps
    - Correct hand reset: feed() clears the sequence on done=True
    - Huber loss + gradient clipping: more robust than raw MSE
    - Fast tensors: np.array() before torch.from_numpy() — no slow-list warning
    """

    def __init__(self, state_shape, num_actions, device,
                 hidden_size=64, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 buffer_capacity=5000, batch_size=64, min_replay=256,
                 target_update_freq=50, l2_reg=0.0):
        """
        Initialize DRQN agent.

        Args:
            state_shape: Dimension of state observation
            num_actions: Number of possible actions
            device: torch device (cpu/cuda)
            hidden_size: Size of hidden layers
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            min_replay: Minimum buffer size before training
            target_update_freq: Steps between target network updates
        """
        self.use_raw = False
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay = min_replay
        self.target_update_freq = target_update_freq
        self._train_steps = 0

        # Online network (trained every step)
        self.model = LeducDRQN(state_shape, num_actions, hidden_size).to(device)

        # Target network (frozen, synced periodically for stable Q-targets)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()

        # Optimizer with L2 regularization (weight decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.lr = lr  # Store for learning rate scheduling

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = SequenceReplayBuffer(buffer_capacity)
        self.current_hand_sequence = []  # running obs list for the ongoing hand

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seq_to_tensor(self, sequence):
        """
        Convert list of numpy obs to (1, seq_len, feat) FloatTensor.
        Fast path via np.array to avoid slow-list warning.

        Args:
            sequence: List of numpy observations

        Returns:
            torch.Tensor of shape (1, seq_len, feat)
        """
        arr = np.array(sequence, dtype=np.float32)  # avoids slow-list warning
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, T, feat)

    def _greedy_action(self, legal_actions):
        """
        Run the online network on the current hand sequence, mask illegal actions.

        Args:
            legal_actions: List of legal action indices

        Returns:
            int: Selected action index
        """
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
        """
        Training mode: epsilon-greedy exploration.

        Args:
            state: Current game state

        Returns:
            int: Selected action
        """
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        self.current_hand_sequence.append(obs)

        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)

        return self._greedy_action(legal_actions)

    def eval_step(self, state):
        """
        Evaluation mode: greedy, no exploration.
        FIX: current_hand_sequence must be reset before each game.

        Args:
            state: Current game state

        Returns:
            (action, info_dict): Selected action and empty info dictionary
        """
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        self.current_hand_sequence.append(obs)

        return self._greedy_action(legal_actions), {}

    def feed(self, transition):
        """
        Called by RLCard after each step during training.
        On done=True: save the full hand sequence to the replay buffer and reset.

        Args:
            transition: (state, action, reward, next_state, done) tuple
        """
        state, action, reward, next_state, done = transition

        if done and len(self.current_hand_sequence) > 0:
            self.replay_buffer.push(self.current_hand_sequence, action, reward)
            self.current_hand_sequence = []  # reset for next hand

    def train(self):
        """
        One gradient step using a batch of full-episode sequences.

        Returns:
            float or None: Scalar loss for logging, or None if buffer not warm yet
        """
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
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self._train_steps
        }, path)

    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self._train_steps = checkpoint['train_steps']

    def update_learning_rate(self, new_lr):
        """Update learning rate for Phase 2 fine-tuning."""
        self.lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"  Learning rate updated: {new_lr}")
