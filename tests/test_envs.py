"""
Tests for environment wrappers.
"""

import pytest
import gymnasium as gym
from pyexplore.envs import MiniGridEnv

def test_minigrid_env_initialization():
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
    assert env.state_dim is not None
    assert env.action_dim is not None
    
def test_minigrid_env_reset():
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
    state, info = env.reset()
    assert state is not None
    assert info is not None
    
def test_minigrid_env_step():
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
    env.reset()
    next_state, reward, terminated, truncated, info = env.step(0)
    assert next_state is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info is not None 