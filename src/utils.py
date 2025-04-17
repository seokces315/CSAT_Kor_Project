import os
import random
import numpy as np
import torch


# Reproducibility
def set_seed(seed):
    # os.environ["PYTHONHASHEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
