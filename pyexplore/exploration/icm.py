"""
Intrinsic Curiosity Module (ICM) exploration strategy.
"""

import numpy as np
import torch
import torch.nn as nn
from pyexplore.models.icm_model import ICMNetwork

class ICMExplorer:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.icm = ICMNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=learning_rate)
        
    def select_action(self, q_values, state):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get intrinsic reward
        intrinsic_reward = self.icm.get_intrinsic_reward(state_tensor)
        
        # Combine extrinsic and intrinsic rewards
        total_reward = q_values + intrinsic_reward.detach().numpy()
        
        return np.argmax(total_reward)
        
    def update(self, state, action, next_state):
        # Update ICM network
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        action_tensor = torch.FloatTensor([action])
        
        self.optimizer.zero_grad()
        loss = self.icm.compute_loss(state_tensor, action_tensor, next_state_tensor)
        loss.backward()
        self.optimizer.step() 