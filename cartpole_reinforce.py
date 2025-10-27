"""
REINFORCE (Policy Gradient) implementation for CartPole-v1
This implements the classic Monte Carlo policy gradient algorithm.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """Neural network that outputs action probabilities given a state."""

    def __init__(self, state_dim, action_dim, hidden_dim=4):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """Returns probability distribution over actions."""
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE agent that learns to balance the CartPole."""

    def __init__(self, state_dim, action_dim, learning_rate=0.005, gamma=0.99):
        """
        Args:
            state_dim: Dimension of state space (4 for CartPole)
            action_dim: Number of actions (2 for CartPole: left/right)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
        """
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Storage for episode data
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Select an action by sampling from the policy distribution.

        Args:
            state: Current environment state

        Returns:
            action: Selected action (0 or 1)
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()

        # Save log probability for training
        self.saved_log_probs.append(m.log_prob(action))

        return action.item()

    def compute_returns(self):
        """
        Compute discounted returns for each timestep in the episode.

        Returns:
            returns: List of discounted returns
        """
        returns = []
        G = 0

        # Compute returns backwards from end of episode
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Normalize returns for more stable training
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """
        Update policy using the REINFORCE algorithm.
        This implements the policy gradient theorem.
        """
        returns = self.compute_returns()

        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            # Policy gradient: ∇J(θ) ≈ ∑ ∇log π(a|s) * G
            policy_loss.append(-log_prob * G)

        # Compute total loss and update
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []

        return policy_loss.item()


def train(episodes=1000, render_every=None):
    """
    Train the REINFORCE agent on CartPole.

    Args:
        episodes: Number of episodes to train
        render_every: Render every N episodes (None for no rendering)

    Returns:
        agent: Trained agent
        episode_rewards: List of total rewards per episode
    """
    env = gym.make("CartPole-v1")

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole

    agent = REINFORCEAgent(state_dim, action_dim)
    episode_rewards = []

    print(f"Training REINFORCE agent for {episodes} episodes...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}\n")

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        # Collect one episode
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            agent.rewards.append(reward)
            episode_reward += reward

            if terminated or truncated:
                break

        # Update policy after episode
        loss = agent.update_policy()
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")

    env.close()
    print("\nTraining completed!")

    return agent, episode_rewards


def evaluate(agent, episodes=10, render=True):
    """
    Evaluate a trained agent.

    Args:
        agent: Trained REINFORCE agent
        episodes: Number of episodes to evaluate
        render: Whether to render the environment

    Returns:
        avg_reward: Average reward over evaluation episodes
    """
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    total_rewards = []

    print(f"\nEvaluating agent for {episodes} episodes...")

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # Use policy without exploration (deterministic)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy(state_tensor)
                action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")

    return avg_reward


def plot_training_progress(episode_rewards, window=50):
    """Plot training progress with moving average."""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')

    # Compute moving average
    moving_avg = np.convolve(episode_rewards,
                             np.ones(window)/window,
                             mode='valid')
    plt.plot(range(window-1, len(episode_rewards)),
             moving_avg,
             label=f'{window}-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training Progress on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_progress.png')
    print("Training progress plot saved as 'training_progress.png'")
    plt.show()


if __name__ == "__main__":
    # Train the agent
    agent, rewards = train(episodes=1000)

    # Plot training progress
    plot_training_progress(rewards)

    # Evaluate the trained agent
    evaluate(agent, episodes=5, render=True)

    # Save the trained model
    torch.save(agent.policy.state_dict(), 'cartpole_reinforce_policy.pth')
    print("\nTrained policy saved as 'cartpole_reinforce_policy.pth'")
