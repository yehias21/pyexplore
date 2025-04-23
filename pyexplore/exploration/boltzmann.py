"""
Boltzmann (softmax) exploration strategy.
"""

import numpy as np

class BoltzmannExploration:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        
    def select_action(self, q_values, state=None):
        exp_values = np.exp(q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(len(q_values), p=probs) 