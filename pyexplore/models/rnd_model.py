"""
Random Network Distillation (RND) network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(RNDNetwork, self).__init__()
        
        # Target network (fixed)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictor network
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize target network and freeze its parameters
        self._initialize_target_net()
        
    def _initialize_target_net(self):
        # Initialize target network with random weights
        for param in self.target_net.parameters():
            param.requires_grad = False
            
    def get_intrinsic_reward(self, state):
        target_features = self.target_net(state)
        predicted_features = self.predictor_net(state)
        return F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
        
    def compute_loss(self, state):
        target_features = self.target_net(state)
        predicted_features = self.predictor_net(state)
        return F.mse_loss(predicted_features, target_features) 