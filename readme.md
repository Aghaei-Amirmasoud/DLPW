# Mastering Imperfect Information with Deep Recurrent Q-Networks in Leduc Hold'em

## Overview
This repository implements a **Deep Recurrent Q-Network (DRQN)** agent for mastering **Partially Observable Markov Decision Processes (POMDPs)** in the poker domain. The project focuses on **Leduc Hold'em**, a simplified poker variant where the agent must infer hidden information (opponent's cards) through betting patterns and action sequences.

The core innovation is a **recurrent architecture (LSTM)** that maintains memory of game state and betting sequences, enabling sophisticated strategies including **bluffing** and **opponent modeling**.

- **Course:** Deep Learning (Project Work - 3 CFU)
- **Area of Interest:** Reinforcement Learning & Sequential Decision Making

---

## Key Features

**DRQN Architecture**: LSTM-based Q-Network for sequential decision making  
**Two-Phase Curriculum Learning**: Progressive training from random to strategic opponents  
**Multiple Baselines**: Comparison with DQN, Heuristic, and Random agents  
**Bluff Detection**: Quantitative analysis of strategic deception  
**Modular Codebase**: Clean separation of agents, training, evaluation, and visualization  
**Target Network**: Stable Q-learning with frozen target network  
**Sequence Replay Buffer**: Full episode sequence storage for LSTM training  

---

## Quick Start

### Installation
```bash
git clone <repository-url>
cd DLPW
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python run_pipeline.py
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/DLPW.ipynb
```

```bash
# In Colab, just run all cells!
# All code is embedded - no setup required
```

---

## Project Structure

```
DLPW/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # All hyperparameters and paths
├── run_pipeline.py                    # Main entry point (training + evaluation)
│
├── agents/
│   ├── drqn_agent.py                 # DRQN agent with sequence replay
│   └── heuristic_agent.py            # Rule-based conservative agent
│
├── models/
│   ├── drqn.py                       # LSTM-based Q-Network architecture
│   └── replay_buffer.py              # Sequence replay buffer
│
├── training/
│   └── trainer.py                    # Two-phase curriculum trainer
│
├── evaluation/
│   └── metrics.py                    # EV, action distribution, bluff analysis
│
├── analysis/
│   └── visualization.py              # Training curves, comparison plots
│
├── outputs/
│   ├── models/                       # Saved model checkpoints
│   └── plots/                        # Training curves, evaluation charts
│
└── notebooks/
    ├── DLPW.ipynb                    # Notebook to run in Colab
```

---

## Environment: Leduc Hold'em

Leduc Hold'em is a benchmark environment provided by the **RLCard framework**.

### Game Mechanics
- **The Deck:** 6 cards (two suits, three ranks: Jack, Queen, King)
- **Winning Rules:**
  - **Pairs Win:** Private card matches the community card
  - **High Card Wins:** If no pairs, the highest rank wins (King > Queen > Jack)
  - **Ties:** Split pot
- **Betting Structure:**
  - **Ante:** 1 chip
  - **Round 1 (Pre-Flop):** 1 private card dealt. Bet/raise = 2 chips (max 2 raises)
  - **Round 2 (The Flop):** 1 community card revealed. Bet/raise = 4 chips (max 2 raises)

---

## Methodology

### DRQN Architecture

```
Input (batch, seq_len, 36) 
  → FC(36→64) + ReLU 
  → LSTM(64→64) 
  → last timestep 
  → FC(64→4) Q-values
```

**Key Components:**
- **Input State:** Observable variables (Agent's card ID, Community card ID, current pot size)
- **Memory Layer:** LSTM to process action sequences and infer hidden states
- **Output Layer:** Q-values for discrete actions: **Fold**, **Call/Check**, **Raise**

### Two-Phase Curriculum Training

**Phase 1 (Episodes 0–7,999):**  
Train vs **Random Agent** → Learn basic card strength and poker fundamentals

**Phase 2 (Episodes 7,999–29,999):**  
Train vs **Heuristic Agent** → Learn to exploit predictable patterns

### Key Technical Improvements
- **Sequence Replay Buffer**: Stores full episode sequences instead of single transitions
- **Target Network**: Frozen Q-network updated every N steps for stable learning
- **Fixed Eval Bug**: Hand sequences now reset properly between games
- **Fast Tensor Creation**: Optimized with `np.array()` to eliminate warnings
- **Gradient Clipping**: Robust training with Huber loss
- **Two-Phase Curriculum**: Learn fundamentals vs Random, then exploit patterns vs Heuristic

---

## Usage Examples

### 1. Train DRQN from Scratch
```python
from run_pipeline import setup_environment, train_drqn

env, agent_drqn, agent_dqn, agent_random, agent_heuristic = setup_environment()
drqn_history = train_drqn(env, agent_drqn, agent_random, agent_heuristic)
```

### 2. Evaluate Against Specific Opponent
```python
from evaluation import evaluate_agents

ev, trajectories = evaluate_agents(
    env, agent_drqn, agent_heuristic, 
    num_hands=3000, label='Heuristic'
)
print(f"EV: {ev:+.3f} chips/hand")
```

### 3. Analyze Action Distribution
```python
from evaluation import compute_action_distribution
from config import ACTION_NAMES

stats = compute_action_distribution(trajectories, ACTION_NAMES)
print(f"Bluff Rate: {stats['bluff_rate']:.1f}%")
```

### 4. Customize Hyperparameters
Edit `config.py`:
```python
NUM_EPISODES_PHASE1 = 20000     # Increase Phase 1 training
NUM_EPISODES_PHASE2 = 20000     # Increase Phase 2 training
LEARNING_RATE = 5e-4            # Lower learning rate
EPSILON_DECAY = 0.999           # Slower exploration decay
TARGET_UPDATE_FREQ = 100        # Update target network frequency
```

---

## Evaluation Metrics

### Primary Metric: Expected Value (EV)
Average chips won per hand. Positive EV indicates profitable strategy.

### Action Distribution
Percentage breakdown of **Fold** / **Call** / **Check** / **Raise** actions.

### Bluff Detection
Frequency of raising with weak hands (Jack) to force opponent folds.  
**Interpretation:** Non-zero bluff rate proves the LSTM learned betting context, not just card strength.


## Comparative Analysis

The DRQN agent is evaluated against:

1. **Random Agent**: Baseline for learning (implemented)
2. **Rule-Based Heuristic**: Conservative bot betting on card strength (implemented)
3. **Standard DQN Agent**: Feed-forward network to demonstrate the need for memory (implemented)
---

## Visualization Outputs

All plots are automatically saved to `outputs/plots/`:

1. **`drqn_training_curves.png`**: EV and loss curves during two-phase training
2. **`drqn_vs_dqn_comparison.png`**: Side-by-side performance comparison
---

## File Descriptions

### Core Modules
- **`config.py`**: Central configuration for all hyperparameters, paths, and settings
- **`run_pipeline.py`**: End-to-end training and evaluation pipeline

### Agents
- **`agents/drqn_agent.py`**: DRQN with epsilon-greedy exploration, target network, sequence replay
- **`agents/heuristic_agent.py`**: Deterministic rule-based agent (no learning)

### Models
- **`models/drqn.py`**: PyTorch LSTM-based Q-Network
- **`models/replay_buffer.py`**: Sequence-based experience replay (fixed from v1)

### Training
- **`training/trainer.py`**: Curriculum trainer with two-phase structure

### Evaluation
- **`evaluation/metrics.py`**: EV calculation, action distribution, bluff rate analysis

### Analysis
- **`analysis/visualization.py`**: Matplotlib-based plotting utilities

---

## Configuration Options

Edit `config.py` to customize:

### Training Schedule
```python
NUM_EPISODES_PHASE1 = 15000  # Training vs Random
NUM_EPISODES_PHASE2 = 15000  # Training vs Heuristic
EVALUATE_EVERY = 1000        # Evaluation frequency
```

### Model Architecture
```python
HIDDEN_SIZE = 64            # LSTM hidden dimension
LEARNING_RATE = 1e-3        # Adam optimizer LR
```

### Exploration
```python
EPSILON_START = 1.0         # Initial exploration
EPSILON_MIN = 0.05          # Minimum exploration
EPSILON_DECAY = 0.9995      # Decay rate per step
```

### Replay Buffer
```python
BUFFER_CAPACITY = 5000      # Max sequences stored
BATCH_SIZE = 64             # Training batch size
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- RLCard 1.2+
- NumPy 1.24+
- Matplotlib 3.7+
- Jupyter (optional, for notebooks)

See `requirements.txt` for exact versions.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dlpw2024,
  title={Mastering Imperfect Information with Deep Recurrent Q-Networks in Leduc Hold'em},
  author={[Amir Masoud Aghaei]},
  year={2026},
  note={Course Project: Deep Learning}
}
```

---

## References

1. Hausknecht, M., & Stone, P. (2015). Deep Recurrent Q-Learning for Partially Observable MDPs. *AAAI*.
2. Zha, D., et al. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *arXiv*.
3. Southey, F., et al. (2005). Bayes' Bluff: Opponent Modelling in Poker. *UAI*.

---

## Contact

For questions or collaboration:
- **Email:** [mailto:amirmasoud.aghaei@studio.unibo.it]
- **GitHub:** [https://github.com/Aghaei-Amirmasoud]

---

