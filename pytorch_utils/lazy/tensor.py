from functools import wraps
from typing import List

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


def cycle_view(ndim: int,
               ndim_diff: int = 0, dim: int = 0, squeeze: bool = True, unsqueeze: bool = True):
    """
    Decorator that makes sure tensor to be of specific size, run through function (first argument)
    and sizes single return output back to its original dimensionality or a specified delta.

    Args:
        ndim: target dimension
        ndim_diff: difference in target output dimension
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
                ),
                ndim=x.dim() + ndim_diff,
                dim=dim,
                squeeze=squeeze,
                unsqueeze=unsqueeze
            )
        return wrapper_cycle_view
    return decorator_cycle_view


def invert_permutation(*dims: int) -> List[int]:
    """
    Inverts a permutation defined by integer indices. Even if the indices are (partly) negative,
    the result will always be an equivalent positive set of indices.
    """
    n = len(dims)
    dims = [d if d >= 0 else d + n for d in dims]
    return torch.argsort(torch.LongTensor(dims)).tolist()


def inverse_permute(t: torch.Tensor, dims):
    """Convenience function to invert a known permutation on a tensor."""
    return t.permute(*invert_permutation(*dims))
