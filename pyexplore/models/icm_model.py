"""
Intrinsic Curiosity Module (ICM) network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ICMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ICMNetwork, self).__init__()
        
        # Feature embedding
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Forward model
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Inverse model
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def get_intrinsic_reward(self, state):
        features = self.feature_net(state)
        return torch.norm(features, dim=1)
        
    def compute_loss(self, state, action, next_state):
        # Feature embeddings
        phi = self.feature_net(state)
        phi_next = self.feature_net(next_state)
        
        # Forward model loss
        forward_input = torch.cat([phi, action], dim=1)
        predicted_phi_next = self.forward_net(forward_input)
        forward_loss = F.mse_loss(predicted_phi_next, phi_next)
        
        # Inverse model loss
        inverse_input = torch.cat([phi, phi_next], dim=1)
        predicted_action = self.inverse_net(inverse_input)
        inverse_loss = F.mse_loss(predicted_action, action)
        
        return forward_loss + inverse_loss 