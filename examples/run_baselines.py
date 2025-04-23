"""
Example script for running baseline exploration strategies.
"""

import gymnasium as gym
import numpy as np
from pyexplore.envs import MiniGridEnv
from pyexplore.exploration import EpsilonGreedy, BoltzmannExploration
from pyexplore.models import PolicyNetwork
from pyexplore.utils import Logger, plot_learning_curve

def run_experiment(env_name, strategy, num_episodes=1000):
    # Initialize environment and models
    env = MiniGridEnv(env_name)
    policy_net = PolicyNetwork(env.state_dim, env.action_dim)
    logger = Logger()
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get Q-values from policy network
            q_values = policy_net.predict(state)
            
            # Select action using exploration strategy
            action = strategy.select_action(q_values, state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update policy network (simple update for demonstration)
            policy_net.update(state, action, reward, next_state, done)
            
            # Update state and episode statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
        # Log episode statistics
        logger.log_episode(episode_reward, episode_length, 0)
        
        if (episode + 1) % 100 == 0:
            stats = logger.get_stats()
            print(f"Episode {episode + 1}:")
            print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Mean Length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
    
    return logger.episode_rewards

if __name__ == "__main__":
    # Run experiments with different strategies
    env_name = "MiniGrid-Empty-5x5-v0"
    
    # Epsilon-greedy
    print("\nRunning Epsilon-Greedy...")
    epsilon_greedy = EpsilonGreedy(epsilon=0.1)
    epsilon_rewards = run_experiment(env_name, epsilon_greedy)
    
    # Boltzmann
    print("\nRunning Boltzmann...")
    boltzmann = BoltzmannExploration(temperature=1.0)
    boltzmann_rewards = run_experiment(env_name, boltzmann)
    
    # Plot results
    plot_learning_curve(epsilon_rewards, title="Epsilon-Greedy Learning Curve")
    plot_learning_curve(boltzmann_rewards, title="Boltzmann Learning Curve") 