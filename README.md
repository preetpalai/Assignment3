Assignment 3 – Deep Q-Learning on Pong

Course: CSCN8020 – Reinforcement Learning Programming
Student: Preetpal Singh

This project implements a Deep Q-Network (DQN) to play the Atari game PongDeterministic-v4, including frame preprocessing, replay buffer, target network, epsilon-greedy exploration, and hyperparameter experiments.

Project Overview

This assignment demonstrates:

Implementing a DQN with convolutional layers

Preprocessing Atari frames (crop → downsample → grayscale → normalize)

Building and sampling from a replay buffer

Using a target network for stable training

Experimenting with hyperparameters:

Batch size (8 vs 16)

Target update frequency (10 vs 3 episodes)

Deep Q-Network Architecture

The DQN processes 4 stacked grayscale frames:

Input shape: (4, 80, 80)


Convolutional layers:

Layer	Channels	Kernel	Stride	Output Purpose
Conv1	4 → 32	8×8	4	Coarse spatial features
Conv2	32 → 64	4×4	2	Mid-level features
Conv3	64 → 64	3×3	1	Fine motion patterns

Fully connected layers:

Flatten → 2688 units

FC1 → 512

FC2 → 6 (legal Pong actions)

Other components:

Optimizer: Adam (1e-4)

Loss: MSE

Discount factor: γ = 0.95

Replay buffer: 50,000 transitions

Target network sync: every N episodes

Experiments Conducted
Experiment A – Baseline

Batch size: 8

Target update: 10

Experiment B – Batch Size Effect

Batch size: 16

Target update: 10

Experiment C – Target Update Frequency

Batch size: 8

Target update: 3

Each experiment recorded:

Episode rewards

Moving average rewards (window=5)

Training stability

Convergence behavior

Plots were generated for each experiment.

Key Results
Batch Size = 16

✔ More stable gradients
✔ Faster reward improvement
✔ Best moving-average performance

Target Update = 3

Frequent target updates → unstable learning
Oscillations and inconsistent reward trends

Baseline (8, 10)

✔ Moderate stability
✔ Slow but steady improvements

Best Performing Configuration

Batch size: 16
Target update: 10

This produced the smoothest learning curve and highest moving-average reward.
