import argparse
import json
import importlib
import torch
from pyexplore.config import Config, DEFAULT_CONFIG
from pyexplore.train import train_model
from pyexplore.evaluate import evaluate_model

def load_config(config_path: str = None) -> Config:
    """Load configuration from file or use default."""
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = DEFAULT_CONFIG
    return Config.from_dict(config_dict)

def get_component(module_name: str, class_name: str):
    """Dynamically import and return a class from a module."""
    module = importlib.import_module(f'pyexplore.{module_name}')
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate RL agents')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], 
                       default='both', help='Mode to run: train, eval, or both')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for evaluation (required for eval mode)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode in ['eval', 'both'] and args.model_path is None and args.mode != 'both':
        raise ValueError("Model path is required for evaluation mode")
    
    if args.mode in ['train', 'both']:
        print(f"Starting training with {config.exploration.name} exploration...")
        model_path = train_model(config)
        print(f"Training completed. Model saved at: {model_path}")
    
    if args.mode in ['eval', 'both']:
        model_path = args.model_path if args.mode == 'eval' else model_path
        print(f"Starting evaluation of model: {model_path}")
        evaluate_model(model_path, config)
        print("Evaluation completed.")

if __name__ == "__main__":
    main() 