import torch
import numpy as np

def pil_to_tensor(pil_image, dtype):
    return torch.as_tensor(np.array(pil_image), dtype=dtype)
