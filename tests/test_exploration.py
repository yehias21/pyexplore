"""
Tests for exploration strategies.
"""

import pytest
import numpy as np
from pyexplore.exploration import (
    EpsilonGreedy,
    BoltzmannExploration,
    ICMExplorer,
    RNDExplorer,
    CountBasedExplorer
)

def test_epsilon_greedy():
    strategy = EpsilonGreedy(epsilon=0.1)
    q_values = np.array([1.0, 2.0, 3.0])
    action = strategy.select_action(q_values)
    assert action in [0, 1, 2]
    
def test_boltzmann():
    strategy = BoltzmannExploration(temperature=1.0)
    q_values = np.array([1.0, 2.0, 3.0])
    action = strategy.select_action(q_values)
    assert action in [0, 1, 2]
    
def test_icm():
    strategy = ICMExplorer(state_dim=4, action_dim=2)
    q_values = np.array([1.0, 2.0])
    state = np.random.rand(4)
    action = strategy.select_action(q_values, state)
    assert action in [0, 1]
    
def test_rnd():
    strategy = RNDExplorer(state_dim=4)
    q_values = np.array([1.0, 2.0])
    state = np.random.rand(4)
    action = strategy.select_action(q_values, state)
    assert action in [0, 1]
    
def test_count_based():
    strategy = CountBasedExplorer(state_dim=4)
    q_values = np.array([1.0, 2.0])
    state = np.random.rand(4)
    action = strategy.select_action(q_values, state)
    assert action in [0, 1] 