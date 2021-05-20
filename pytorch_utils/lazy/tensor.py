from functools import wraps

import torch


def view_to_dim(x: torch.Tensor, ndim: int, dim: int, squeeze: bool = True, unsqueeze: bool = True):
    """
    Make tensor to be of specific dimension

    Args:
        x: tensor
        ndim: target dim
        dim: index of dimension to inflate / deflate
        squeeze: allow squeeze
        unsqueeze: allow unsqueeze

    """
    while x.dim() > ndim and squeeze:
        if x.size(dim) != 1:
            raise ValueError
        x = x.squeeze(dim)

    while x.dim() < ndim and unsqueeze:
        x = x.unsqueeze(dim)

    return x


def cycle_view(ndim: int, ndim_out: int = None, dim: int = 0, squeeze: bool = True, unsqueeze: bool\
    = True):
    """
    Decorator that makes sure tensor to be of specific size, run through function (first argument)
    and sizes single return output back to its original dimensionality or another target dim.

    Args:
        ndim: target dimension
        ndim_out: target output dimension (set only if differs from original dimension)
        dim: which dimension to inflate / deflate
        squeeze: allow squeeze
        unsqueeze: allow unsqueeze

    """
    def decorator_cycle_view(func):
        @wraps(func)
        def wrapper_cycle_view(x, *args, **kwargs):
            return view_to_dim(
                func(
                    view_to_dim(x, ndim=ndim, dim=dim, squeeze=squeeze, unsqueeze=unsqueeze),
                    *args,
                    **kwargs
                ), ndim=x.dim() if ndim_out is None else ndim_out,
                dim=dim,
                squeeze=squeeze,
                unsqueeze=unsqueeze
            )
        return wrapper_cycle_view
    return decorator_cycle_view
