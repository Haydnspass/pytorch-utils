from functools import wraps

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


def cycle_view(ndim: int, dim: int, squeeze: bool = True, unsqueeze: bool = True):
    def decorator_cycle_view(func):
        @wraps(func)
        def wrapper_cycle_view(x, *args, **kwargs):
            return view_to_dim(
                func(
                    view_to_dim(x, ndim=ndim, dim=dim, squeeze=squeeze, unsqueeze=unsqueeze),
                    *args,
                    **kwargs
                ), ndim=x.dim(), dim=dim, squeeze=squeeze, unsqueeze=unsqueeze)
        return wrapper_cycle_view
    return decorator_cycle_view
