"""
Epsilon-greedy exploration strategy.
"""

import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def select_action(self, q_values, state=None):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values) 