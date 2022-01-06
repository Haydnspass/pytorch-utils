from typing import Sequence, Callable
import torch


def unbind(t: torch.Tensor, dim: int = 0, clone: bool = False) -> tuple:
    """
    Unbind with clone option

    Args:
        t: tensor
        dim: dimension to unbind
        clone: clone elements such that they don't share the same base

    Returns:
        tuple of tensors
    """
    b = torch.unbind(t, dim=dim)
    if clone:
        b = [bb.clone() for bb in b]

    return b


def recurse(x, fn: Callable, max_depth: int):
    if isinstance(x, torch.Tensor):
        return fn(x)
    if max_depth == 0:
        return x
    if isinstance(x, Sequence):
        return [recurse_numpy(xx, max_depth - 1) for xx in x]
    if isinstance(x, dict):
        return {k: recurse_numpy(v, max_depth - 1) for k, v in x.items()}

    return x


def _to_numpy(x: torch.Tensor):
    return x.numpy()


def recurse_numpy(x, max_depth: int):
    return recurse(x, _to_numpy, max_depth=max_depth)


def recurse_torch(x, max_depth: int):
    return recurse(x, torch.from_numpy, max_depth=max_depth)
