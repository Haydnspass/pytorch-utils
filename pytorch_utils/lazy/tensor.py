from functools import wraps, partial
from typing import List, Optional, Union, Callable

import torch

from . import cycle


def view_to_dim(x: torch.Tensor, ndim: int, dim: int, squeeze: bool = True, unsqueeze: bool = True):
    """
    Make tensor to be of specific dimension

    Args:
        x: tensor
        ndim: target dim
        dim: index of dimension to inflate / deflate
        unsqueeze: allow unsqueeze
        squeeze: allow squeeze

    """
    while x.dim() > ndim and squeeze:
        if x.size(dim) != 1:
            raise ValueError
        x = x.squeeze(dim)

    while x.dim() < ndim and unsqueeze:
        x = x.unsqueeze(dim)

    return x


def view_to_dim_dec(ndim: int, dim: int,
                    unsqueeze: bool = True, squeeze: bool = True,
                    arg: Optional[Union[int, str]] = 0) -> Callable:
    """
    Decorator that changes the dimension of a specific argument.

    Args:
        ndim: target dim
        dim: index of dimension to inflate / deflate
        unsqueeze: allow unsqueeze
        squeeze: allow squeeze
        arg: which argument to apply to

    """
    view_to_dim_part = partial(view_to_dim, ndim=ndim, dim=dim, unsqueeze=unsqueeze, squeeze=squeeze)

    return cycle.cycle(view_to_dim_part, None, arg, None)


def cycle_view(ndim: int, ndim_diff: int = 0, dim: int = 0,
               squeeze: bool = True, unsqueeze: bool = True):
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


@cycle_view(4, 0)
def _to_chw(x: torch.Tensor):
    return x.permute(0, -1, 1, 2)


@cycle_view(4, 0)
def _to_hwc(x: torch.Tensor):
    return x.permute(0, 2, 3, 1)


def chw_hwc_cycle(arg: Optional[Union[int, str]] = 0, return_arg: Optional[int] = 0):
    """
    Decorator. Changes channel convention from NxHxWxC to NxCxHxW
    """
    return cycle.cycle(_to_hwc, _to_chw, arg, return_arg)


def hwc_chw_cycle(arg: Optional[Union[int, str]] = 0, return_arg: Optional[int] = 0):
    """
    Decorator. Changes channel convention from NxCxHxW to NxHxWxC
    """
    return cycle.cycle(_to_chw, _to_hwc, arg, return_arg)
