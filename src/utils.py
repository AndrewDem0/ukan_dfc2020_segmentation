import os
import random
import torch
import numpy as np

def seed_everything(seed: int = 42):
    """
    Апаратна та програмна фіксація seed-значень.
    Гарантує детермінованість обчислень на CUDA при повторних запусках.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(dir_path: str):
    """Безпечне створення системних директорій у Debian."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)