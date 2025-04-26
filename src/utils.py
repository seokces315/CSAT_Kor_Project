import os
import random
import numpy as np
import torch


# Reproducibility
def set_seed(seed):
    os.environ["PYTHONHASHEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Option
    if hasattr(torch, "use_deterministic_algorithm"):
        torch.use_deterministic_algorithms(True)


# Function to check cuda's quality
def check_cuda_capability():
    cuda_capability = torch.cuda.get_device_capability()[0]
    return True if cuda_capability >= 8 else False
