from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class EnvironmentConfig:
    name: str
    type: str
    params: Dict[str, Any]

@dataclass
class ModelConfig:
    name: str
    type: str
    params: Dict[str, Any]

@dataclass
class ExplorationConfig:
    name: str
    type: str
    params: Dict[str, Any]

@dataclass
class TrainingConfig:
    episodes: int
    batch_size: int
    target_update: int
    gamma: float
    memory_size: int
    learning_rate: float

@dataclass
class EvaluationConfig:
    episodes: int
    max_steps: int

@dataclass
class Config:
    environment: EnvironmentConfig
    model: ModelConfig
    exploration: ExplorationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return cls(
            environment=EnvironmentConfig(**config_dict['environment']),
            model=ModelConfig(**config_dict['model']),
            exploration=ExplorationConfig(**config_dict['exploration']),
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation'])
        )

# Default configuration
DEFAULT_CONFIG = {
    'environment': {
        'name': 'minigrid',
        'type': 'MiniGridEnvironment',
        'params': {
            'grid_type': 'MiniGrid-Empty-16x16-v0',
            'max_steps': 1000
        }
    },
    'model': {
        'name': 'dqn',
        'type': 'DQN',
        'params': {
            'hidden_layer_size': (128, 128)
        }
    },
    'exploration': {
        'name': 'count_based',
        'type': 'CountBased',
        'params': {
            'beta': 0.1
        }
    },
    'training': {
        'episodes': 2000,
        'batch_size': 128,
        'target_update': 1000,
        'gamma': 0.99,
        'memory_size': 200000,
        'learning_rate': 0.0005
    },
    'evaluation': {
        'episodes': 1000,
        'max_steps': 500
    }
} 