import torch

def get_device(config):
    use_gpu = getattr(config, "use_gpu", True)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")

    return device