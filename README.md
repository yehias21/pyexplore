# PyExplore

A modular reinforcement learning framework for exploring different environments, models, and exploration strategies.

## Features

- Modular architecture for easy extension
- Support for different environments (currently MiniGrid)
- Multiple exploration strategies (Epsilon-Greedy, Boltzmann, Count-Based)
- Deep Q-Network (DQN) implementation
- Experiment tracking with Comet.ml
- Configuration-based setup
- Easy training and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pyexplore.git
cd pyexplore
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run training and evaluation with the default configuration:

```bash
python -m pyexplore.main --mode both
```

### Using Custom Configuration

1. Create a configuration file (see `examples/config.json` for reference)
2. Run with your configuration:

```bash
python -m pyexplore.main --mode both --config path/to/your/config.json
```

### Example

Run the MiniGrid example:

```bash
python examples/run_minigrid.py
```

## Configuration

The framework uses a JSON-based configuration system. Here's an example configuration:

```json
{
    "environment": {
        "name": "minigrid",
        "type": "MiniGridEnvironment",
        "params": {
            "grid_type": "MiniGrid-Empty-16x16-v0",
            "max_steps": 1000
        }
    },
    "model": {
        "name": "dqn",
        "type": "DQN",
        "params": {
            "hidden_layer_size": [128, 128]
        }
    },
    "exploration": {
        "name": "count_based",
        "type": "CountBased",
        "params": {
            "beta": 0.1
        }
    },
    "training": {
        "episodes": 2000,
        "batch_size": 128,
        "target_update": 1000,
        "gamma": 0.99,
        "memory_size": 200000,
        "learning_rate": 0.0005
    },
    "evaluation": {
        "episodes": 1000,
        "max_steps": 500
    }
}
```

## Extending the Framework

### Adding a New Environment

1. Create a new class inheriting from `BaseEnvironment`:
```python
from pyexplore.envs.base_env import BaseEnvironment

class NewEnvironment(BaseEnvironment):
    def __init__(self, **params):
        super().__init__()
        # Initialize with params
        
    def create(self, **kwargs):
        # Create and return environment
        
    def preprocess(self, observation, device):
        # Preprocess observation
        
    def get_action_space(self):
        # Return action space size
        
    def get_observation_space(self):
        # Return observation space size
```

2. Add it to your configuration:
```json
{
    "environment": {
        "name": "new_env",
        "type": "NewEnvironment",
        "params": {
            "param1": "value1",
            "param2": "value2"
        }
    }
}
```

### Adding a New Model

1. Create a new class inheriting from `BaseModel`:
```python
from pyexplore.models.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, input_size, output_size, **params):
        super().__init__(input_size, output_size)
        # Initialize model architecture
        
    def forward(self, x):
        # Define forward pass
```

2. Add it to your configuration:
```json
{
    "model": {
        "name": "new_model",
        "type": "NewModel",
        "params": {
            "param1": "value1"
        }
    }
}
```

### Adding a New Exploration Strategy

1. Create a new class inheriting from `BaseExplorationStrategy`:
```python
from pyexplore.exploration.base_strategy import BaseExplorationStrategy

class NewStrategy(BaseExplorationStrategy):
    def __init__(self, num_actions, device, **params):
        super().__init__(num_actions, device)
        # Initialize strategy parameters
        
    def select_action(self, state, policy_net, steps_done):
        # Implement action selection logic
```

2. Add it to your configuration:
```json
{
    "exploration": {
        "name": "new_strategy",
        "type": "NewStrategy",
        "params": {
            "param1": "value1"
        }
    }
}
```

## Results

[PLACEHOLDER: GIF of solved environment]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


