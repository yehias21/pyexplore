import torch
import comet_ml
from pyexplore.evaluation.evaluator import Evaluator

# Hyperparameters
NUM_ACTIONS = 3
INPUT_SIZE = 49
EVAL_EPISODES = 1000
MAX_STEPS = 500

def evaluate_model(model_path, exploration_method):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Comet experiment for evaluation
    experiment = comet_ml.start(
        api_key="IZUqacouHXlsJu3hXQ5LuIy5Z",
        project_name="dqn_16x16_evaluation",
        workspace="yehias21",
    )
    
    # Initialize evaluator
    evaluator = Evaluator(
        model_path=model_path,
        input_size=INPUT_SIZE,
        num_actions=NUM_ACTIONS,
        device=device,
        eval_episodes=EVAL_EPISODES,
        max_steps=MAX_STEPS
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Log evaluation results to Comet
    experiment.log_metrics({
        "completion_rate": results["completion_rate"],
        "average_reward": results["average_reward"],
        "average_steps": results["average_steps"],
        "exploration_method": exploration_method
    })
    
    experiment.end()

if __name__ == "__main__":
    # Example usage
    model_path = "dqn_16x16_model_count_based.pth"  # Path to your trained model
    exploration_method = "count_based"  # The exploration method used during training
    evaluate_model(model_path, exploration_method) 