"""
Simple heuristic controller for CartPole-v1
Uses the pole's angular velocity to decide which direction to push the cart.
"""
import gymnasium as gym
import numpy as np


def heuristic_policy(observation):
    """
    Simple heuristic: push cart in the direction of the pole's angular velocity.

    Args:
        observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Returns:
        action: 0 (left) or 1 (right)
    """
    # Index 3 is pole angular velocity
    pole_angular_velocity = observation[3]

    # Push right if pole is falling right, left if falling left
    action = 1 if pole_angular_velocity > 0 else 0

    return action


def evaluate_heuristic(episodes=10, render=True, verbose=True):
    """
    Evaluate the heuristic controller over multiple episodes.

    Args:
        episodes: Number of episodes to run
        render: Whether to render the environment
        verbose: Whether to print detailed output per episode

    Returns:
        avg_reward: Average reward over all episodes
    """
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    total_rewards = []

    print(f"Evaluating heuristic controller for {episodes} episodes...")
    print(f"Policy: Push cart in direction of pole's angular velocity\n")

    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0

        if verbose and episode == 0:
            print(f"Starting observation: {observation}")
            print("[cart_position, cart_velocity, pole_angle, pole_angular_velocity]\n")

        while True:
            # Use heuristic policy to select action
            action = heuristic_policy(observation)

            # Take action in environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    min_reward = np.min(total_rewards)
    max_reward = np.max(total_rewards)

    print(f"\n{'='*50}")
    print(f"Results over {episodes} episodes:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Std deviation:  {std_reward:.2f}")
    print(f"  Min reward:     {min_reward:.2f}")
    print(f"  Max reward:     {max_reward:.2f}")
    print(f"{'='*50}")

    return avg_reward


if __name__ == "__main__":
    # Run evaluation with multiple episodes
    evaluate_heuristic(episodes=10, render=True, verbose=True)
