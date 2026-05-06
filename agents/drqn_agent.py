"""
DRQN Agent with sequence replay buffer and target network
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random

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
        self.current_episode_transitions = []  # (obs_history, action, reward, next_obs_history, done)

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

        # Select action
        if np.random.rand() < self.epsilon:
            action = np.random.choice(legal_actions)
        else:
            action = self._greedy_action(legal_actions)

        return action

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
        Stores (obs_history, action, reward, next_obs_history, done).

        Args:
            transition: (state, action, reward, next_state, done) tuple
        """
        state, action, reward, next_state, done = transition

        # Current observation history (up to this point)
        obs_history = list(self.current_hand_sequence)

        # Next observation history (add next_state obs if not terminal)
        if not done and next_state is not None:
            next_obs_history = obs_history + [next_state['obs']]
        else:
            next_obs_history = obs_history  # Terminal state

        # Store transition
        self.current_episode_transitions.append((
            obs_history,
            action,
            reward,
            next_obs_history,
            done
        ))

        # On episode end: save episode to buffer and reset
        if done:
            self.replay_buffer.push(self.current_episode_transitions)
            self.current_hand_sequence = []
            self.current_episode_transitions = []

    def train(self):
        """
        One gradient step using proper TD learning with bootstrapping.

        Returns:
            float or None: Scalar loss for logging, or None if buffer not warm yet
        """
        if len(self.replay_buffer) < self.min_replay:
            return None

        # Sample episodes from buffer
        episodes = self.replay_buffer.sample(self.batch_size)

        # Collect all transitions from sampled episodes
        all_transitions = []
        for episode in episodes:
            all_transitions.extend(episode)

        # If we don't have enough transitions, return
        if len(all_transitions) == 0:
            return None

        # Randomly sample batch_size transitions
        if len(all_transitions) > self.batch_size:
            transitions = random.sample(all_transitions, self.batch_size)
        else:
            transitions = all_transitions

        # Find max sequence length for padding
        max_len = max(len(obs_hist) for obs_hist, _, _, _, _ in transitions)
        if max_len == 0:
            return None

        feat = transitions[0][0][0].shape[0] if len(transitions[0][0]) > 0 else 0
        if feat == 0:
            return None

        # Prepare batches
        states = np.zeros((len(transitions), max_len, feat), dtype=np.float32)
        next_states = np.zeros((len(transitions), max_len, feat), dtype=np.float32)
        actions = np.zeros(len(transitions), dtype=np.int64)
        rewards = np.zeros(len(transitions), dtype=np.float32)
        dones = np.zeros(len(transitions), dtype=np.float32)

        for i, (obs_hist, action, reward, next_obs_hist, done) in enumerate(transitions):
            # Pad observation histories
            if len(obs_hist) > 0:
                obs_arr = np.array(obs_hist, dtype=np.float32)
                states[i, :len(obs_hist)] = obs_arr

            if len(next_obs_hist) > 0:
                next_obs_arr = np.array(next_obs_hist, dtype=np.float32)
                next_states[i, :len(next_obs_hist)] = next_obs_arr

            actions[i] = action
            rewards[i] = reward
            dones[i] = float(done)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Compute current Q-values
        self.model.train()
        current_q, _ = self.model(states_t)
        current_q = current_q.gather(1, actions_t).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q, _ = self.target_model(next_states_t)
            next_q_max = next_q.max(1)[0]
            # TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            target_q = rewards_t + self.gamma * next_q_max * (1 - dones_t)

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
