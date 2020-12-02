"""
Transform, do something, transform it back.
"""
import functools
from typing import Optional, Callable

import torch


def cycle(trafo_a: Optional[Callable], trafo_b: Optional[Callable],
          arg: Optional[int] = 0, return_arg: Optional[int] = 0):
    """

    Args:
        trafo_a: must be inplace
        trafo_b: must be inplace
        arg: which argument should be modified
        return_arg: which return argument should be modified

    Returns:

    """

    def decorator_cycle(func):

        @functools.wraps(func)
        def wrapper_cycle(*args):

            if trafo_a is not None:
                args = list(args)
                args[arg] = trafo_a(args[arg])

            out = func(*args)

            if trafo_b is None:
                return out

            elif not isinstance(out, tuple):
                return trafo_b(out)

            else:
                out = list(out)
                out[return_arg] = trafo_b(out[return_arg])
                return tuple(out)

        return wrapper_cycle

    return decorator_cycle


def torch_np_cycle(arg: Optional[int] = 0, return_arg: Optional[int] = 0):
    """
    Decorator for numpy only functions that converts a tensor to np.ndarray,
    exectues function and converts it back.
    """
    def to_np(x):
        return x.numpy()

    return cycle(to_np, torch.from_numpy, arg, return_arg)
