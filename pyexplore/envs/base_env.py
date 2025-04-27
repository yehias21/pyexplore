from abc import ABC, abstractmethod
import torch

class BaseEnvironment(ABC):
    @abstractmethod
    def create(self, **kwargs):
        """Create and return the environment instance."""
        pass
    
    @abstractmethod
    def preprocess(self, observation, device):
        """Preprocess the observation into a tensor."""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Return the number of possible actions."""
        pass
    
    @abstractmethod
    def get_observation_space(self):
        """Return the size of the observation space."""
        pass 