import torch
import numpy as np
from pyexplore.envs.minigrid_env import create_minigrid_environment, preprocess
from pyexplore.models.dqn import DQN

class Evaluator:
    def __init__(self, model_path, input_size, num_actions, device, eval_episodes=1000, max_steps=500):
        self.model_path = model_path
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        
        # Load the trained model
        self.policy_net = DQN(input_size, num_actions).to(device)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
        
        # Create environment
        self.env = create_minigrid_environment()
        
    def select_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].unsqueeze(0)
    
    def evaluate(self):
        print("Starting evaluation...")
        
        # Initialize counters for evaluation metrics
        finish_counter = 0.0
        total_steps = 0.0
        total_reward = 0.0
        
        # Run the evaluation loop
        for episode in range(self.eval_episodes):
            # Initialize the environment and state
            current_obs, _ = self.env.reset()
            current_state = preprocess(current_obs, self.device)
            
            # The main RL loop
            for step in range(self.max_steps):
                # Select and perform an action
                action = self.select_action(current_state)
                
                # Take action and observe result
                obs, reward, done, truncated, _ = self.env.step(action.item())
                
                if done or truncated:
                    total_reward += reward
                    total_steps += self.env.step_count
                    if done:
                        print(f"Finished evaluation episode {episode} with reward {reward:.2f}, "
                              f"{self.env.step_count} steps, reaching the goal")
                        finish_counter += 1
                    if truncated:
                        print(f"Failed evaluation episode {episode} with reward {reward:.2f}, "
                              f"{self.env.step_count} steps")
                    break
                
                # Move to the next state
                current_state = preprocess(obs, self.device)
        
        # Print a summary of the evaluation results
        completion_rate = finish_counter / self.eval_episodes
        avg_reward = total_reward / self.eval_episodes
        avg_steps = total_steps / self.eval_episodes
        
        print(f"Completion rate {completion_rate:.2f} with an average reward {avg_reward:.4f} "
              f"and average steps {avg_steps:.2f}")
        print("Finished evaluation!")
        
        return {
            "completion_rate": completion_rate,
            "average_reward": avg_reward,
            "average_steps": avg_steps
        } 