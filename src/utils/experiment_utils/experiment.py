import os
from datetime import datetime

def create_experiment(config):
    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    exp_id = len(os.listdir(base_dir)) + 1
    exp_name = f"exp_{exp_id:03d}_{config.name}"

    exp_path = os.path.join(base_dir, exp_name)

    os.makedirs(exp_path)
    os.makedirs(f"{exp_path}/logs")
    os.makedirs(f"{exp_path}/checkpoints")
    os.makedirs(f"{exp_path}/outputs")

    return exp_path