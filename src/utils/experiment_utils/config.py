import yaml

class Config:
    def __init__(self, dict_):
        for k, v in dict_.items():
            if isinstance(v, dict):
                v = Config(v)   # 🔥 recursive conversion
            setattr(self, k, v)

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict)