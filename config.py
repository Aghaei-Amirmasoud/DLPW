"""
Configuration file for DLPW project
All hyperparameters, paths, and training settings
"""

import torch

# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENV_NAME = 'leduc-holdem'

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
HIDDEN_SIZE = 64
LSTM_LAYERS = 1

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
# Learning
LEARNING_RATE = 1e-3
LEARNING_RATE_PHASE2 = 1e-3  # Lower LR for Phase 2 (fine-tuning)
GAMMA = 0.99  # Discount factor
L2_REGULARIZATION = 0.0  # Weight decay to prevent overfitting

# Exploration
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

# Experience Replay
BUFFER_CAPACITY = 5000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 256

# Training Schedule
TARGET_UPDATE_FREQ = 50  # Update target network every N gradient steps

# Two-Phase Curriculum
NUM_EPISODES_PHASE1 = 15000  # Training vs Random
NUM_EPISODES_PHASE2 = 15000  # Training vs Heuristic
EVALUATE_EVERY = 1000
EVALUATE_NUM = 1000

# Final Evaluation
NUM_EVAL_HANDS = 3000

# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = 'outputs'
MODEL_DIR = f'{OUTPUT_DIR}/models'
PLOT_DIR = f'{OUTPUT_DIR}/plots'
NOTEBOOK_DIR = 'notebooks'

# Model checkpoints
DRQN_CHECKPOINT = f'{MODEL_DIR}/drqn_final.pt'
DQN_CHECKPOINT = f'{MODEL_DIR}/dqn_final.pt'

# Plots
TRAINING_CURVES_PLOT = f'{PLOT_DIR}/training_curves.png'
EV_COMPARISON_PLOT = f'{PLOT_DIR}/ev_comparison.png'
ACTION_DIST_PLOT = f'{PLOT_DIR}/action_distribution.png'

# ============================================================
# ACTION MAPPING
# ============================================================
ACTION_NAMES = {
    'call': 'Call',
    'raise': 'Raise',
    'fold': 'Fold',
    'check': 'Check'
}

# ============================================================
# LOGGING
# ============================================================
VERBOSE = True
LOG_LOSS = True
