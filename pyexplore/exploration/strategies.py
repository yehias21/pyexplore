import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from .base_strategy import BaseExplorationStrategy

class ExplorationStrategy:
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device

class EpsilonGreedy(BaseExplorationStrategy):
    def __init__(self, num_actions, device, start_epsilon=1.0, stop_epsilon=0.01, decay_rate=3000):
        super().__init__(num_actions, device)
        self.start_epsilon = start_epsilon
        self.stop_epsilon = stop_epsilon
        self.decay_rate = decay_rate

    @classmethod
    def from_config(cls, config):
        return cls(
            num_actions=config['num_actions'],
            device=config['device'],
            start_epsilon=config.get('start_epsilon', 1.0),
            stop_epsilon=config.get('stop_epsilon', 0.01),
            decay_rate=config.get('decay_rate', 3000)
        )

    def select_action(self, state, policy_net, steps_done):
        sample = random.random()
        eps_threshold = self.stop_epsilon + (self.start_epsilon - self.stop_epsilon) * \
                       math.exp(-1.0 * steps_done / self.decay_rate)
        
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].unsqueeze(0)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], 
                              device=self.device, dtype=torch.long)

class Boltzmann(BaseExplorationStrategy):
    def __init__(self, num_actions, device, start_temp=1.5, min_temp=0.1, temp_decay=3000):
        super().__init__(num_actions, device)
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay

    @classmethod
    def from_config(cls, config):
        return cls(
            num_actions=config['num_actions'],
            device=config['device'],
            start_temp=config.get('start_temp', 1.5),
            min_temp=config.get('min_temp', 0.1),
            temp_decay=config.get('temp_decay', 3000)
        )

    def select_action(self, state, policy_net, steps_done):
        temperature = self.min_temp + (self.start_temp - self.min_temp) * \
                     math.exp(-1.0 * steps_done / self.temp_decay)
        with torch.no_grad():
            q_values = policy_net(state)
        probabilities = F.softmax(q_values / temperature, dim=1)
        return torch.multinomial(probabilities, 1)

class CountBased(BaseExplorationStrategy):
    def __init__(self, num_actions, device, beta=0.1):
        super().__init__(num_actions, device)
        self.beta = beta
        self.counts = {}

    @classmethod
    def from_config(cls, config):
        return cls(
            num_actions=config['num_actions'],
            device=config['device'],
            beta=config.get('beta', 0.1)
        )

    def get_count(self, state):
        state_key = tuple(state.cpu().numpy().flatten())
        return self.counts.get(state_key, 0)

    def update_count(self, state):
        state_key = tuple(state.cpu().numpy().flatten())
        self.counts[state_key] = self.counts.get(state_key, 0) + 1

    def get_bonus(self, state):
        count = max(1, self.get_count(state))
        return self.beta / np.sqrt(count)

    def get_additional_reward(self, state):
        return self.get_bonus(state)

    def select_action(self, state, policy_net, steps_done):
        bonus = self.get_bonus(state)
        self.update_count(state)
        with torch.no_grad():
            q_values = policy_net(state)
        q_values_with_bonus = q_values + bonus
        return q_values_with_bonus.max(1)[1].unsqueeze(0)

class RND(BaseExplorationStrategy):
    def __init__(self, num_actions, device, input_size, hidden_size=128, beta=0.1, learning_rate=1e-4):
        super().__init__(num_actions, device)
        self.beta = beta
        self.learning_rate = learning_rate
        
        # Target network (random and fixed)
        self.target_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        ).to(device)
        
        # Predictor network (trained)
        self.predictor_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=learning_rate)
        
        # Initialize target network with random weights and freeze it
        for param in self.target_network.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, config):
        return cls(
            num_actions=config['num_actions'],
            device=config['device'],
            input_size=config['input_size'],
            hidden_size=config.get('hidden_size', 128),
            beta=config.get('beta', 0.1),
            learning_rate=config.get('learning_rate', 1e-4)
        )

    def get_intrinsic_reward(self, state):
        """Calculate intrinsic reward based on prediction error."""
        with torch.no_grad():
            target_features = self.target_network(state)
        predicted_features = self.predictor_network(state)
        prediction_error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=1)
        return prediction_error

    def update(self, state):
        """Update the predictor network."""
        target_features = self.target_network(state)
        predicted_features = self.predictor_network(state)
        loss = F.mse_loss(predicted_features, target_features)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def get_additional_reward(self, state):
        """Get the intrinsic reward for the current state."""
        return self.beta * self.get_intrinsic_reward(state)

    def select_action(self, state, policy_net, steps_done):
        """Select action using the policy network."""
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.max(1)[1].unsqueeze(0) 