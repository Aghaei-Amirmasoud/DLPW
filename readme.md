# Mastering Imperfect Information with Deep Recurrent Q-Networks in Leduc Hold'em

## Overview
This repository contains the implementation of a Reinforcement Learning (RL) agent designed to tackle **Partially Observable Markov Decision Processes (POMDPs)**. The project focuses on **Leduc Hold'em**, a simplified poker variant, where the agent must infer hidden information (the opponent's cards) through betting patterns.

The core solution implements a **Deep Recurrent Q-Network (DRQN)** using recurrent layers (LSTM/GRU) to maintain memory of the game state and betting sequences, enabling strategies such as bluffing.

- **Course:** Deep Learning (Project Work - 3 CFU)
- **Area of Interest:** Reinforcement Learning & Sequential Data Processing

## Requirements
- Python 3.x
- [PyTorch](https://pytorch.org/)
- [RLCard Framework](https://rlcard.org/)
- Jupyter Notebook (for running `.ipynb` files)

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DLPW
   ```
2. Install the required dependencies:
   ```bash
   pip install rlcard torch
   ```

## Environment: Leduc Hold'em
Leduc Hold'em is a benchmark environment provided by the **RLCard framework**.

### Game Mechanics
- **The Deck:** 6 cards (two suits, three ranks: Jack, Queen, King).
- **Winning Rules:**
  - **Pairs Win:** Private card matches the community card.
  - **High Card Wins:** If no pairs, the highest rank wins (King > Queen > Jack).
  - **Ties:** Split pot.
- **Betting Structure:**
  - **Ante:** 1 chip.
  - **Round 1 (Pre-Flop):** 1 private card dealt. Bet/raise = 2 chips (max 2 raises).
  - **Round 2 (The Flop):** 1 community card revealed. Bet/raise = 4 chips (max 2 raises).

## Methodology
The agent is built using **PyTorch** with a **DRQN** architecture:
- **Input State:** Observable variables (Agent's card ID, Community card ID, current pot size).
- **Memory Layer:** LSTM/GRU to process action sequences and infer hidden states.
- **Output Layer:** Linear layer for Q-values of discrete actions: **Fold**, **Call/Check**, **Raise**.

Training is conducted via self-play using a sequence-based experience replay buffer.

## Project Structure
- `DLPW.ipynb`: Main notebook containing the implementation, training loop, and evaluation. Currently contains the baseline evaluation using Random Agents.
- `README.md`: Project documentation.

## Scripts and Entry Points
- **Baseline Evaluation:** Run the cells in `DLPW.ipynb` to see the performance of Random vs. Random agents in Leduc Hold'em.
- **Training (TODO):** DRQN training implementation is under development.

## Evaluation Metrics
- **Primary Metric (Expected Value):** Average chips won per hand.
- **Action Distribution:** Percentage of Fold/Call/Raise actions.
- **Bluffing Frequency:** Analysis of high-risk raises (e.g., raising with a Jack) to force opponent folds.

## Comparative Analysis
The DRQN agent is evaluated against:
1. **Random Agent:** (Implemented) Baseline for learning.
2. **Rule-Based Agent:** Conservative bot betting on card strength.
3. **Standard DQN Agent (Optional):** Feed-forward network to demonstrate the need for memory.


