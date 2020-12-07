import torch
import matplotlib.pyplot as plt


def tshow(t: torch.Tensor, autosqueeze: bool = True, *args, **kwargs):
    """Tensor friendly plt.imshow"""

    t = t.detach().cpu()
    if autosqueeze:
        t = t.squeeze()
    
    if t.dim() == 3:
        t = t.permute(1, 2, 0)

    return plt.imshow(t, *args, **kwargs)
