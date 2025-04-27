import torch
import torch.optim as optim
import torch.nn as nn
import comet_ml
from comet_ml.integration.pytorch import log_model
import random
import math
import os

from pyexplore.models.dqn import DQN
from pyexplore.exploration.strategies import EpsilonGreedy, Boltzmann, CountBased
from pyexplore.envs.minigrid_env import create_minigrid_environment, preprocess
from pyexplore.utils.memory import ReplayMemory

# Hyperparameters
NUM_ACTIONS = 3
INPUT_SIZE = 49
ALPHA = 0.0005
EPISODES = 2000
BATCH_SIZE = 128
TARGET_UPDATE = 1000
GAMMA = 0.99
MEMORY_SIZE = 200000

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def optimize_model(memory, policy_net, target_net, optimizer, device, batch_size, gamma, additional_reward=0):
    if len(memory) < batch_size:
        return None
    
    transitions = memory.sample(batch_size)
    batch = memory.Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.currentState).to(device)
    action_batch = torch.cat(batch.action).to(device)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    reward_batch = torch.tensor(batch.reward, device=device)
    if additional_reward != 0:
        reward_batch = reward_batch + additional_reward
    
    non_final_next_states = torch.cat([s for s in batch.nextState if s is not None]).to(device)
    next_state_values = torch.zeros(batch_size, device=device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)),
                                device=device, dtype=torch.bool)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    
    optimizer.step()
    return loss.item()

def train_model(exploration_method="epsilon_greedy"):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    random.seed(5)
    
    # Initialize Comet experiment
    experiment = comet_ml.start(
        api_key="IZUqacouHXlsJu3hXQ5LuIy5Z",
        project_name="dqn_16x16",
        workspace="yehias21",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "learning_rate": ALPHA,
        "batch_size": BATCH_SIZE,
        "memory_size": MEMORY_SIZE,
        "target_update": TARGET_UPDATE,
        "gamma": GAMMA,
        "episodes": EPISODES,
        "exploration_method": exploration_method,
        "device": str(device),
        "input_size": INPUT_SIZE,
        "hidden_layers": "128, 128",
        "dropout": 0.2
    })
    
    # Initialize environment and networks
    env = create_minigrid_environment()
    policy_net = DQN(INPUT_SIZE, NUM_ACTIONS).to(device)
    target_net = DQN(INPUT_SIZE, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Log model architecture
    log_model(experiment, policy_net, model_name=f"dqn_16x16_{exploration_method}")
    
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA, weight_decay=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Initialize exploration strategy
    if exploration_method == "epsilon_greedy":
        exploration = EpsilonGreedy(NUM_ACTIONS, device)
    elif exploration_method == "boltzmann":
        exploration = Boltzmann(NUM_ACTIONS, device)
    else:  # count_based
        exploration = CountBased(NUM_ACTIONS, device)
    
    # Training loop
    print(f"Starting training with {exploration_method} exploration on {device}...")
    steps_done = 0
    
    for episode in range(EPISODES):
        obs, info = env.reset()
        state = preprocess(obs, device)
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(1000):  # max steps per episode
            # Select action based on exploration method
            action = exploration.select_action(state, policy_net, steps_done)
            
            # Take action and observe result
            next_obs, reward, done, truncated, _ = env.step(action.item())
            next_state = preprocess(next_obs, device) if not (done or truncated) else None
            
            # Store transition in memory
            memory.push(state, action, next_state, torch.tensor([reward], device=device))
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Optimize model
            additional_reward = exploration.get_bonus(state) if isinstance(exploration, CountBased) and state is not None else 0
            loss = optimize_model(memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA, additional_reward)
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done or truncated:
                print(f"Episode {episode + 1}: steps = {step + 1}, reward = {total_reward}")
                break
            
            steps_done += 1
        
        # Log metrics to Comet
        experiment.log_metrics({
            "reward": total_reward,
            "steps": step + 1,
            "average_loss": episode_loss / loss_count if loss_count > 0 else 0,
            "epsilon": exploration.stop_epsilon + (exploration.start_epsilon - exploration.stop_epsilon) * \
                      math.exp(-1.0 * steps_done / exploration.decay_rate) if isinstance(exploration, EpsilonGreedy) else 0,
            "temperature": exploration.min_temp + (exploration.start_temp - exploration.min_temp) * \
                          math.exp(-1.0 * steps_done / exploration.temp_decay) if isinstance(exploration, Boltzmann) else 0
        }, step=episode)
    
    # Save model and close
    model_path = f"models/dqn_16x16_model_{exploration_method}.pth"
    torch.save(policy_net.state_dict(), model_path)
    experiment.end()
    env.close()
    print(f"Training completed with {exploration_method} exploration!")
    
    return model_path

if __name__ == "__main__":
    exploration_methods = ["count_based"]
    for method in exploration_methods:
        model_path = train_model(method) 