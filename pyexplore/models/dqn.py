import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class DQN(BaseModel):
    def __init__(self, input_size, num_actions, hidden_layer_size=(128, 128)):
        super(DQN, self).__init__(input_size, num_actions)
        self.hidden_layer_size = hidden_layer_size
        
        self.fc1 = nn.Linear(input_size, hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.fc3 = nn.Linear(hidden_layer_size[1], num_actions)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            input_size=config['input_size'],
            num_actions=config['output_size'],
            hidden_layer_size=config.get('hidden_layer_size', (128, 128))
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc3(x) 