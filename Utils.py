import torch
import numpy as np
import random


class Utils:
    @staticmethod
    def fix_randomness(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Randomness Fixed to seed {seed}")
