# PyExplore: Exploration Strategies for Deep Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

**PyExplore** is a Python library inspired by recent advancements in exploration methods in Deep Reinforcement Learning (RL), facilitating the implementation and comparison of diverse exploration strategies, including both classical and advanced intrinsic motivation approaches.

---

## 🚀 Overview

Effective exploration is critical in reinforcement learning, especially in scenarios with sparse rewards. This library provides implementations of multiple exploration strategies based on recent literature, simplifying experimentation and benchmarking.

**PyExplore** includes:
- Basic strategies: ε-greedy, Boltzmann Exploration
- Intrinsic motivation strategies:
  - **Intrinsic Curiosity Module (ICM)**
  - **Random Network Distillation (RND)**
  - **Count-based Exploration**

This selection is inspired by the comprehensive survey paper:  
> Pawel Ladosz, Lilian Weng, Minwoo Kim, Hyondong Oh. *Exploration in Deep Reinforcement Learning: A Survey* (2022).

---

## 📂 Structure

```
pyexplore/
├── pyexplore/          # Core library components
│   ├── envs/           # Environment wrappers
│   ├── exploration/    # Exploration strategies
│   ├── models/         # Neural network architectures
│   └── utils/          # Helpers and utilities
├── examples/           # Example scripts for library usage
├── tests/              # Unit tests
├── setup.py            # Installation setup
├── requirements.txt    # Project dependencies
└── README.md
```

---

## ⚙️ Installation

Clone and install the library locally:

```bash
git clone https://github.com/yourusername/pyexplore.git
cd pyexplore
pip install .
```

Or install directly from source:

```bash
pip install git+https://github.com/yourusername/pyexplore.git
```

---

## 🎯 Usage Example

Here's a simple usage example employing MiniGrid with the Intrinsic Curiosity Module (ICM):

```python
from pyexplore.envs import MiniGridEnv
from pyexplore.exploration import ICMExplorer
from pyexplore.models import PolicyNetwork

# Set up environment and exploration strategy
env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
strategy = ICMExplorer(state_dim=env.state_dim, action_dim=env.action_dim)
policy_net = PolicyNetwork(env.state_dim, env.action_dim)

state, _ = env.reset()
done = False

while not done:
    q_values = policy_net.predict(state)
    action = strategy.select_action(q_values, state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    policy_net.update(state, action, reward, next_state, done)
    state = next_state
```

---

## 📈 Benchmarking

Experiments and comparison scripts are provided under the `examples/` directory to replicate and compare exploration strategies. Evaluate on benchmarks such as:
- **MiniGrid** (recommended for sparse reward tasks)
- **Atari** (e.g., Montezuma's Revenge)
- **MuJoCo** (continuous control tasks)

Run the provided scripts:

```bash
python examples/run_icm.py
python examples/run_rnd.py
python examples/run_baselines.py
```

---

## 📚 References

- **ICM**: Pathak et al. (2017). "Curiosity-driven Exploration by Self-supervised Prediction."
- **RND**: Burda et al. (2018). "Exploration by Random Network Distillation."
- **Count-based**: Bellemare et al. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation."

For a comprehensive understanding, refer to the original survey:

> Pawel Ladosz et al. "Exploration in Deep Reinforcement Learning: A Survey," *Information Fusion*, 2022.

---

## ✅ Testing

Run tests with pytest:

```bash
pip install pytest
pytest tests/
```

---

## 🤝 Contributions

Contributions are welcome! Please open issues or submit pull requests directly through GitHub.

---

## 📜 License

PyExplore is released under the MIT License. See [LICENSE](LICENSE) for more details.


