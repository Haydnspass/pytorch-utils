import torch


def view_to_dim(x: torch.Tensor, ndim: int, dim: int,
                squeeze: bool = True, unsqueeze: bool = True):
    while x.dim() > ndim and squeeze:
        if x.size(dim) != 1:
            raise ValueError
        x = x.squeeze(dim)

    while x.dim() < ndim and unsqueeze:
        x = x.unsqueeze(dim)

    return x
