import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyexplore.main import main

if __name__ == "__main__":
    # Run training and evaluation with the example configuration
    main(
        mode="both",
        config="examples/config.json",
        model_path=None  # Will be created during training
    ) 