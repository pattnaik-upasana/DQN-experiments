# Deep Q-Network (DQN) with Transfer Learning

This repository contains the implementation of a Deep Q-Network (DQN) agent with transfer learning capabilities, developed as part of a thesis project on reinforcement learning in Atari environments.

Original Practice and Iterative Practice DQN algorithm and code by Raviteja Kancharla.

## Overview

The project implements a DQN agent that learns to play Atari games using transfer learning. The network uses pre-trained convolutional layers from one game (e.g., Pong) and fine-tunes them for a target game (e.g., Breakout or Tennis).

## Key Features

- **Transfer Learning**: Loads pre-trained CNN weights from a source task
- **Frame Stacking**: Processes sequences of 4 frames to capture temporal information
- **Parametric ReLU (PReLU)**: Custom activation function for improved learning
- **Huber Loss**: Robust loss function for Q-value estimation
- **Experience Replay**: Stores and samples past experiences for stable training
- **Target Network**: Separate network for stable Q-value targets
- **Gradient Clipping**: Prevents exploding gradients during training

## Requirements

- TensorFlow 1.10
- NumPy
- Matplotlib
- Python 3.x
- CUDA 9.2 (for GPU support)

## Configuration

Key hyperparameters in `config.py`:
- `frame_stack`: Number of frames to stack (default: 4)
- `batch_size`: Training batch size
- `discount_factor`: Reward discount factor (gamma)
- `grad_clip`: Gradient clipping threshold
- `eval_freq`: Evaluation frequency
- `iterations`: Total training iterations

### Key Parameters in `train.py`

- `weightsPath`: Path to pre-trained weights file
- `resultPath`: Directory for saving results
- `envName`: Name for the experiment
- `C['env_id']`: Atari environment name

## Transfer Learning Workflow

1. **Pre-training**: Train DQN on source task (e.g., Pong)
2. **Save Weights**: Extract and save CNN layer weights
3. **Transfer**: Load pre-trained CNN weights using `wt_cnn` parameter
4. **Fine-tune**: Train on target task (e.g., Breakout) with frozen/unfrozen CNN layers

## Output Files

- `reward_atari_base.pk1`: Episode rewards during training
- `trainMean_atari_base.pk1`: Mean training rewards (every 100 episodes)
- `evalMean_atari_base.pk1`: Mean evaluation rewards
- `video_atari_base.pk1`: Best evaluation episode frames
- `model.ckpt`: Saved TensorFlow model
