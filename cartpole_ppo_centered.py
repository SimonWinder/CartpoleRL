"""
Proximal Policy Optimization (PPO) implementation for CartPole-v1
with modified reward to keep cart centered at x=0.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """
    Actor-Critic network that outputs both policy (actor) and value estimate (critic).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy) - outputs logits for numerical stability
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """Returns both action logits and state value."""
        features = self.shared(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def act(self, state):
        """Select action and return log probability and value."""
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)  # Use logits for numerical stability
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value


class PPOAgent:
    """
    PPO agent that learns to balance the CartPole while keeping it centered.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            epsilon_clip: Clipping parameter for PPO objective
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            epochs: Number of optimization epochs per update
        """
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs

        # Storage for trajectory data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)

        # Store trajectory data
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def store_transition(self, reward, done):
        """Store reward and done flag for the last transition."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            next_value: Value estimate for the next state (0 if terminal)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        # Convert lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.cat(self.values).view(-1)  # Ensure 1-d to avoid 0-d tensor bug
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Compute GAE backwards through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Returns = Advantages + Values
        returns = advantages + values.detach()

        # Normalize advantages with improved stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:  # Only normalize if std is meaningful
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Clip advantages to prevent extreme values
        advantages = torch.clamp(advantages, -10, 10)

        return advantages, returns

    def update_policy(self):
        """Update policy using PPO objective."""
        # Get next state value for GAE (0 if episode ended)
        if self.dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                next_state = self.states[-1]
                _, next_value = self.policy(next_state)
                next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Convert trajectory data to tensors
        old_states = torch.cat(self.states).detach()
        old_actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.cat(self.log_probs).detach()

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            # Get current policy outputs
            action_logits, state_values = self.policy(old_states)
            state_values = state_values.squeeze()

            # Check for NaN in network outputs
            if torch.isnan(action_logits).any() or torch.isnan(state_values).any():
                print("WARNING: NaN detected in network outputs. Skipping update.")
                return 0.0, 0.0, 0.0

            # Compute action log probabilities using logits for numerical stability
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # Ratio for PPO objective: π_new / π_old
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (MSE)
            value_loss = nn.MSELoss()(state_values, returns)

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Check for NaN in loss
            if torch.isnan(loss):
                print("WARNING: NaN detected in loss. Skipping update.")
                return 0.0, 0.0, 0.0

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Clear trajectory storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        return (
            total_policy_loss / self.epochs,
            total_value_loss / self.epochs,
            total_entropy / self.epochs
        )


def compute_modified_reward(state, base_reward, position_penalty_weight=0.1):
    """
    Compute modified reward that penalizes cart position away from center.

    Args:
        state: Environment state [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        base_reward: Original reward from environment (+1 per step)
        position_penalty_weight: How much to penalize distance from center

    Returns:
        modified_reward: Reward with position penalty applied
    """
    cart_position = state[0]

    # Penalty based on squared distance from center
    # This creates a smooth penalty that increases with distance
    position_penalty = position_penalty_weight * (cart_position ** 2)

    # Modified reward = base reward - position penalty
    modified_reward = base_reward - position_penalty

    # Clip reward to prevent extreme values that can cause numerical instability
    modified_reward = np.clip(modified_reward, -10, 10)

    return modified_reward


def train(episodes=1000, max_timesteps=500, update_every=2048, position_penalty_weight=0.1):
    """
    Train the PPO agent on CartPole with modified reward to keep cart centered.

    Args:
        episodes: Maximum number of episodes to train
        max_timesteps: Maximum timesteps per episode
        update_every: Update policy after this many timesteps
        position_penalty_weight: Weight for penalizing distance from center (0 = no penalty)

    Returns:
        agent: Trained agent
        episode_rewards: List of total rewards per episode
        cart_positions: List of average cart positions per episode
    """
    env = gym.make("CartPole-v1")

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    episode_rewards = []
    cart_positions = []
    timestep_counter = 0

    print(f"Training PPO agent with centered cart reward...")
    print(f"Position penalty weight: {position_penalty_weight}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Updating policy every {update_every} timesteps\n")

    episode = 0
    while episode < episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_positions = []

        for t in range(max_timesteps):
            # Select action
            action = agent.select_action(state)

            # Take step in environment
            next_state, base_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Compute modified reward with position penalty
            modified_reward = compute_modified_reward(state, base_reward, position_penalty_weight)

            # Store transition
            agent.store_transition(modified_reward, done)

            episode_reward += modified_reward
            episode_positions.append(abs(state[0]))  # Track absolute cart position
            timestep_counter += 1
            state = next_state

            # Update policy if enough timesteps collected
            if timestep_counter % update_every == 0:
                policy_loss, value_loss, entropy = agent.update_policy()

            if done:
                break

        # If episode ended before update, trigger update
        if len(agent.rewards) > 0:
            policy_loss, value_loss, entropy = agent.update_policy()

        episode_rewards.append(episode_reward)
        cart_positions.append(np.mean(episode_positions))
        episode += 1

        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_position = np.mean(cart_positions[-50:])
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Avg |Position| (last 50): {avg_position:.3f} | "
                  f"Policy Loss: {policy_loss:.4f} | "
                  f"Value Loss: {value_loss:.4f}")

            # Early stopping if solved (considering position penalty)
            if avg_reward >= 450 - 50 * position_penalty_weight and avg_position < 0.5:
                print(f"\nEnvironment solved with centered cart in {episode} episodes!")
                break

    env.close()
    print("\nTraining completed!")

    return agent, episode_rewards, cart_positions


def evaluate(agent, episodes=10, render=True, position_penalty_weight=0.1):
    """
    Evaluate a trained agent.

    Args:
        agent: Trained PPO agent
        episodes: Number of episodes to evaluate
        render: Whether to render the environment
        position_penalty_weight: Weight for position penalty (should match training)

    Returns:
        avg_reward: Average reward over evaluation episodes
        avg_position: Average absolute cart position
    """
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    total_rewards = []
    total_positions = []

    print(f"\nEvaluating agent for {episodes} episodes...")

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_positions = []

        while True:
            # Use policy deterministically
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_logits, _ = agent.policy(state_tensor)
                action = torch.argmax(action_logits).item()

            next_state, base_reward, terminated, truncated, _ = env.step(action)

            # Compute modified reward
            modified_reward = compute_modified_reward(state, base_reward, position_penalty_weight)

            episode_reward += modified_reward
            episode_positions.append(abs(state[0]))
            state = next_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_positions.append(np.mean(episode_positions))
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Avg |Position| = {np.mean(episode_positions):.3f}")

    env.close()

    avg_reward = np.mean(total_rewards)
    avg_position = np.mean(total_positions)
    print(f"\nAverage reward: {avg_reward:.2f}")
    print(f"Average |position|: {avg_position:.3f}")

    return avg_reward, avg_position


def plot_training_progress(episode_rewards, cart_positions, window=50):
    """Plot training progress with moving average."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window)/window,
                                 mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)),
                 moving_avg,
                 label=f'{window}-Episode Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Progress (with Position Penalty)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot cart positions
    ax2.plot(cart_positions, alpha=0.3, label='Avg |Cart Position|', color='orange')
    if len(cart_positions) >= window:
        moving_avg_pos = np.convolve(cart_positions,
                                      np.ones(window)/window,
                                      mode='valid')
        ax2.plot(range(window-1, len(cart_positions)),
                 moving_avg_pos,
                 label=f'{window}-Episode Moving Average',
                 color='red')
    ax2.axhline(y=0.5, color='g', linestyle='--', label='Target (< 0.5)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average |Cart Position|')
    ax2.set_title('Cart Centering Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_progress_ppo_centered.png')
    print("Training progress plot saved as 'training_progress_ppo_centered.png'")
    plt.show()


if __name__ == "__main__":
    # Train the agent with position penalty
    # Adjust position_penalty_weight to control how much centering matters
    # Higher values = more emphasis on staying centered
    # 0.0 = no position penalty (standard CartPole)
    # 0.1 = moderate penalty (good balance)
    # 0.5 = strong penalty (prioritizes centering)

    agent, rewards, positions = train(
        episodes=1000,
        update_every=2048,
        position_penalty_weight=0.1
    )

    # Plot training progress
    plot_training_progress(rewards, positions)

    # Evaluate the trained agent
    evaluate(agent, episodes=5, render=True, position_penalty_weight=0.1)

    # Save the trained model
    torch.save(agent.policy.state_dict(), 'cartpole_ppo_centered_policy.pth')
    print("\nTrained policy saved as 'cartpole_ppo_centered_policy.pth'")
