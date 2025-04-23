"""
Wrapper for MiniGrid environments.
"""

import gymnasium as gym
import numpy as np

class MiniGridEnv:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        
    def reset(self):
        state, info = self.env.reset()
        return state, info
        
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close() 