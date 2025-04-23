"""
Logging utility for tracking training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        
    def log_episode(self, reward, length, time):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_times.append(time)
        
    def get_stats(self):
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "mean_time": np.mean(self.episode_times),
            "std_time": np.std(self.episode_times)
        }
        
    def save(self, filename=None):
        if filename is None:
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            
        np.savez(
            f"{self.log_dir}/{filename}",
            rewards=self.episode_rewards,
            lengths=self.episode_lengths,
            times=self.episode_times
        ) 