from abc import ABC, abstractmethod
import torch

class BaseExplorationStrategy(ABC):
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device
    
    @abstractmethod
    def select_action(self, state, policy_net, steps_done):
        """Select an action based on the current state and exploration strategy."""
        pass
    
    @classmethod
    def from_config(cls, config):
        """Create an exploration strategy instance from a configuration dictionary."""
        return cls(
            num_actions=config['num_actions'],
            device=config['device']
        )
    
    def get_additional_reward(self, state):
        """Get any additional reward for the current state (e.g., for count-based exploration)."""
        return 0 