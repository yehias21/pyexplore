import gymnasium as gym
import minigrid
from minigrid.wrappers import *
import torch
import numpy as np
from .base_env import BaseEnvironment

class MiniGridEnvironment(BaseEnvironment):
    def __init__(self, grid_type="MiniGrid-Empty-16x16-v0", max_steps=1000):
        self.grid_type = grid_type
        self.max_steps = max_steps
        self.env = None
    
    def create(self, **kwargs):
        env = gym.make(self.grid_type, render_mode=None, max_steps=self.max_steps)
        self.env = ImgObsWrapper(env)
        return self.env
    
    def preprocess(self, observation, device):
        # Extract object information
        (rows, cols, x) = observation.shape
        tmp = np.reshape(observation, [rows * cols * x, 1], "F")[0 : rows * cols]
        obj_info = np.reshape(tmp, [rows, cols], "C")
        
        # Normalize
        normalized = np.array(obj_info) / 10.0
        
        # Flatten and convert to tensor, then move to device
        return torch.from_numpy(normalized.flatten()).float().unsqueeze(0).to(device)
    
    def get_action_space(self):
        return 3  # left, right, move forward
    
    def get_observation_space(self):
        return 49  # size of the flattened input state (16x16 matrix of tile IDs) 