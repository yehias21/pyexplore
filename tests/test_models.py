"""
Tests for neural network models.
"""

import pytest
import torch
import numpy as np
from pyexplore.models import PolicyNetwork, ICMNetwork, RNDNetwork

def test_policy_network():
    net = PolicyNetwork(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    output = net(state)
    assert output.shape == (1, 2)
    
def test_icm_network():
    net = ICMNetwork(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    action = torch.randn(1, 2)
    next_state = torch.randn(1, 4)
    
    # Test forward pass
    features = net.feature_net(state)
    assert features.shape == (1, 128)
    
    # Test loss computation
    loss = net.compute_loss(state, action, next_state)
    assert isinstance(loss, torch.Tensor)
    
def test_rnd_network():
    net = RNDNetwork(state_dim=4)
    state = torch.randn(1, 4)
    
    # Test forward pass
    target_features = net.target_net(state)
    predictor_features = net.predictor_net(state)
    assert target_features.shape == (1, 128)
    assert predictor_features.shape == (1, 128)
    
    # Test loss computation
    loss = net.compute_loss(state)
    assert isinstance(loss, torch.Tensor) 