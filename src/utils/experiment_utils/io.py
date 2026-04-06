import torch
import json
import sys

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_environment(exp_path):
    with open(f"{exp_path}/env.txt", "w") as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"Torch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")

        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")