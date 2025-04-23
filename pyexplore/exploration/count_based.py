"""
Count-based exploration strategy.
"""

import numpy as np
from collections import defaultdict

class CountBasedExplorer:
    def __init__(self, state_dim, beta=0.1):
        self.beta = beta
        self.counts = defaultdict(int)
        
    def _hash_state(self, state):
        # Convert state to a hashable tuple
        return tuple(state.flatten())
        
    def select_action(self, q_values, state):
        # Get state count
        state_hash = self._hash_state(state)
        count = self.counts[state_hash]
        
        # Compute bonus
        bonus = self.beta / np.sqrt(count + 1)
        
        # Combine with Q-values
        total_reward = q_values + bonus
        
        return np.argmax(total_reward)
        
    def update(self, state):
        # Update state count
        state_hash = self._hash_state(state)
        self.counts[state_hash] += 1 