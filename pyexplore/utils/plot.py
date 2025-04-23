"""
Plotting utilities for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(rewards, window=100, title="Learning Curve"):
    """
    Plot the learning curve with moving average.
    
    Args:
        rewards: List of episode rewards
        window: Size of the moving average window
        title: Plot title
    """
    # Compute moving average
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(moving_avg, label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show() 