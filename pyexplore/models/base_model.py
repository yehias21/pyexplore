import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    @classmethod
    def from_config(cls, config):
        """Create a model instance from a configuration dictionary."""
        return cls(
            input_size=config['input_size'],
            output_size=config['output_size']
        )
    
    def save(self, path):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path)) 