"""
Random Network Distillation (RND) exploration strategy.
"""

import numpy as np
import torch
import torch.nn as nn
from pyexplore.models.rnd_model import RNDNetwork

class RNDExplorer:
    def __init__(self, state_dim, learning_rate=0.001):
        self.rnd = RNDNetwork(state_dim)
        self.optimizer = torch.optim.Adam(self.rnd.parameters(), lr=learning_rate)
        
    def select_action(self, q_values, state):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get intrinsic reward
        intrinsic_reward = self.rnd.get_intrinsic_reward(state_tensor)
        
        # Combine extrinsic and intrinsic rewards
        total_reward = q_values + intrinsic_reward.detach().numpy()
        
        return np.argmax(total_reward)
        
    def update(self, state):
        # Update RND network
        state_tensor = torch.FloatTensor(state)
        
        self.optimizer.zero_grad()
        loss = self.rnd.compute_loss(state_tensor)
        loss.backward()
        self.optimizer.step() 