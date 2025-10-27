# CartPole Reinforcement Learning

A comprehensive implementation of reinforcement learning algorithms for the CartPole-v1 environment from Gymnasium. This repository includes three different approaches: a simple heuristic controller, REINFORCE (Policy Gradient), and Proximal Policy Optimization (PPO).

## Overview

The CartPole problem involves balancing a pole on a cart by moving the cart left or right. The goal is to keep the pole balanced for as long as possible.

**State Space (4 dimensions):**
- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

**Action Space (2 discrete actions):**
- 0: Push cart left
- 1: Push cart right

**Success Criteria:**
- Maximum episode length: 500 steps
- Episode terminates if pole angle > 12° or cart position > 2.4

## Implementations

### 1. Simple Heuristic Controller (`cartpole.py`)
A baseline reference controller that uses a simple rule: push the cart in the direction of the pole's angular velocity.

```bash
python cartpole.py
```

### 2. REINFORCE Algorithm (`cartpole_reinforce.py`)
Monte Carlo policy gradient method that learns a policy network by sampling complete episodes.

**Features:**
- Policy Network: 4 → 128 → 128 → 2
- Episode-based learning
- Discounted return computation with normalization
- Learning rate: 0.001, γ: 0.99

**Usage:**
```bash
python cartpole_reinforce.py
```

**Expected Performance:**
- Learns to balance within 500-1000 episodes
- Achieves 400-500 reward consistently

### 3. Proximal Policy Optimization (`cartpole_ppo.py`)
State-of-the-art policy gradient method with clipped surrogate objective and value function.

**Features:**
- Actor-Critic architecture (shared features + separate heads)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective (ε = 0.2)
- Value function loss + entropy bonus
- Multiple optimization epochs per batch
- Learning rate: 3e-4, γ: 0.99, λ: 0.95

**Usage:**
```bash
python cartpole_ppo.py
```

**Expected Performance:**
- Faster learning than REINFORCE (200-400 episodes)
- More stable training
- Achieves maximum reward (500) consistently

### 4. PPO with Centered Cart (`cartpole_ppo_centered.py`)
Enhanced PPO implementation with modified reward function to keep the cart centered at x=0.

**Features:**
- All PPO features plus position-based reward shaping
- Modified reward: `r = base_reward - penalty_weight * (cart_position)²`
- Tracks both reward and cart position during training
- Numerical stability improvements (logits, reward clipping, NaN detection)
- Advantage clipping to prevent extreme values

**Usage:**
```bash
python cartpole_ppo_centered.py
```

**Adjustable Parameters:**
- `position_penalty_weight = 0.0`: No position penalty (standard CartPole)
- `position_penalty_weight = 0.1`: Moderate penalty (good balance)
- `position_penalty_weight = 0.5`: Strong penalty (prioritizes centering)

**Expected Performance:**
- Learns to balance AND stay centered
- Average |cart position| < 0.5 when trained

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- gymnasium[classic-control]
- torch
- numpy
- matplotlib

## Project Structure

```
.
├── cartpole.py                          # Simple heuristic controller
├── cartpole_reinforce.py                # REINFORCE implementation
├── cartpole_ppo.py                      # PPO implementation
├── cartpole_ppo_centered.py             # PPO with centered cart reward
├── show_params.py                       # Utility to display trained network parameters
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── cartpole_reinforce_policy.pth        # Trained REINFORCE model
├── cartpole_ppo_policy.pth              # Trained PPO model
├── cartpole_ppo_centered_policy.pth     # Trained centered PPO model
│
├── training_progress.png                # REINFORCE training plot
├── training_progress_ppo.png            # PPO training plot
└── training_progress_ppo_centered.png   # Centered PPO training plot
```

## Algorithm Comparison

| Algorithm | Episodes to Solve | Sample Efficiency | Stability | Complexity |
|-----------|------------------|-------------------|-----------|------------|
| Heuristic | N/A | N/A | High | Very Low |
| REINFORCE | 500-1000 | Low | Moderate | Low |
| PPO | 200-400 | High | High | Medium |
| PPO Centered | 300-500 | High | High | Medium |

## Training Visualizations

All implementations generate training progress plots showing:
- Episode rewards over time
- Moving average (50-episode window)
- For centered PPO: additional plot of cart position over time

## Loading Trained Models

### REINFORCE
```python
import torch
from cartpole_reinforce import REINFORCEAgent

agent = REINFORCEAgent(state_dim=4, action_dim=2)
agent.policy.load_state_dict(torch.load('cartpole_reinforce_policy.pth'))
```

### PPO
```python
import torch
from cartpole_ppo import PPOAgent

agent = PPOAgent(state_dim=4, action_dim=2)
agent.policy.load_state_dict(torch.load('cartpole_ppo_policy.pth'))
```

### PPO Centered
```python
import torch
from cartpole_ppo_centered import PPOAgent

agent = PPOAgent(state_dim=4, action_dim=2)
agent.policy.load_state_dict(torch.load('cartpole_ppo_centered_policy.pth'))
```

## Custom Training

All implementations support custom parameters:

```python
# REINFORCE
from cartpole_reinforce import train, evaluate, plot_training_progress
agent, rewards = train(episodes=2000)

# PPO
from cartpole_ppo import train, evaluate, plot_training_progress
agent, rewards = train(episodes=1000, update_every=2048)

# PPO Centered
from cartpole_ppo_centered import train, evaluate, plot_training_progress
agent, rewards, positions = train(
    episodes=1000,
    update_every=2048,
    position_penalty_weight=0.1
)
```

## Key Implementation Details

### REINFORCE
- Pure policy gradient with Monte Carlo returns
- No value function baseline
- Simple but high variance

### PPO
- Actor-Critic with shared feature extraction
- GAE for better advantage estimation
- Clipped surrogate prevents destructive policy updates
- Multiple epochs of mini-batch updates
- Entropy bonus encourages exploration

### Numerical Stability (PPO Centered)
- Network outputs logits (not probabilities) for numerical stability
- Reward clipping to [-10, 10]
- Advantage clipping to [-10, 10]
- NaN detection and recovery
- Gradient clipping (norm = 0.5)

## Algorithm References

**REINFORCE:**
Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.

**PPO:**
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

**GAE:**
Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.

## Contributing

Feel free to open issues or submit pull requests with improvements!

## License

MIT License - feel free to use this code for learning and research purposes.
