"""
Transform, do something, transform it back.
"""
import functools
from types import SimpleNamespace
from typing import Optional, Callable, Union

import torch


def cycle(trafo_a: Optional[Callable], trafo_b: Optional[Callable],
          arg: Optional[Union[int, str]] = 0, return_arg: Optional[int] = 0):
    """
    Decorator. Applies an input transformation to a specific input argument
    (if specified) and an output transformation (if specified) to the output or
    one of the outputs if a tuple is returned (i.e. mostly multiple outputs).

    Args:
        trafo_a: input transformation
        trafo_b: output transformation
        arg: index of the input argument which argument should be modified
        return_arg: None if output transformation should be applied to the whole
         output or index of the output argument that should be modified (only
         works if the return is a tuple)

    Returns:
        decorated function
    """

    def decorator_cycle(func):

        @functools.wraps(func)
        def wrapper_cycle(*args, **kwargs):

            if trafo_a is not None:
                if isinstance(arg, int):
                    args = list(args)
                    args[arg] = trafo_a(args[arg])
                elif isinstance(arg, str):
                    kwargs[arg] = trafo_a(kwargs[arg])
                else:
                    raise TypeError(f"Argument type must be int for positional arguments or kwargs "
                                    f"for keyword arguments but not {type(arg)}.")

            out = func(*args, **kwargs)

            if trafo_b is None:
                return out

            if not isinstance(out, tuple) or return_arg is None:
                return trafo_b(out)

            else:
                out = list(out)
                out[return_arg] = trafo_b(out[return_arg])
                return tuple(out)

        return wrapper_cycle

    return decorator_cycle


def torch_np_cycle(arg: Optional[Union[int, str]] = 0, return_arg: Optional[int] = 0):
    """
    Decorator for numpy only functions that converts a tensor to np.ndarray,
    executes function and converts it back.
    """
    def to_np(x):
        return x.numpy()

    return cycle(to_np, torch.from_numpy, arg, return_arg)


def _from_dict(x: Union[dict, SimpleNamespace]) -> SimpleNamespace:
    if isinstance(x, dict):
        return SimpleNamespace(**x)
    else:
        return x


def dict_dot_cycle(arg: Optional[Union[int, str]] = 0, return_arg: Optional[int] = 0,
                   convert_back: bool = True):
    """
    Decorator that changes a dictionary type argument to dot notation
    (via SimpleNamespace) and back.

    Args:
        arg: index of argument
        return_arg: index of return argument or None for whole
        convert_back: convert back to dict

    """

    if convert_back:
        return cycle(_from_dict, vars, arg, return_arg)
    else:
        return cycle(_from_dict, None, arg, None)
