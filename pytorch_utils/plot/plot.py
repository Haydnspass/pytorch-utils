import torch
import matplotlib.pyplot as plt

def tshow(t: torch.Tensor, autosqueeze: bool = True, *args, **kwargs):
    if autosqueeze:
        t = t.squeeze()
    
    if t.dim() == 3:
        t = t.permute(1, 2, 0)

    plt.imshow(t, *args, **kwargs)