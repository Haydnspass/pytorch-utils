import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def fract_random_split(ds: Dataset, val_fraction: float, generator: Optional = None) -> list:
    """
    Split dataset by specifying relative fraction of the validation set. Length of validation set will be rounded up

    Args:
        ds: dataset
        val_fraction: target relativ fraction of the validation set

    """
    val_len = math.ceil(len(ds) * val_fraction)
    train_len = len(ds) - val_len

    if generator is not None:
        return torch.utils.data.random_split(ds, (train_len, val_len), generator=generator)

    return torch.utils.data.random_split(ds, (train_len, val_len))


def split_to_fill(values, n_tar: int, n_tol: int, n_tol_elem: int, order="random", seed=None) -> torch.BoolTensor:
    """
    This is helpful to split something with values until a certain sum is reached.

    Returns indices to split an indexable (values) into two sets. It cum-counts the values
    until a target with a certain tolerance is reached. The order is either random, consecutive or in order of the specified
    order tensor.

    Args:
        values: indexable with values
        n_tar: minimum split sum
        n_tol: overshoot tolerance
        n_tol_elem: individual value max
        order: "random", "linear" or torch.Tensor of size==len(values) with the order of indices
        seed: deterministic seed for numpy
    """
    if seed is not None:
        np.random.seed(seed)

    if order == "random":
        ix_total = np.random.choice(len(values), len(values), replace=False).tolist()
    elif order == "linear":
        ix_total = np.arange(len(values)).tolist()
    else:
        ix_total = order.numpy()

    mask = torch.zeros(len(values), dtype=torch.bool).numpy()

    n = 0
    while (n_tar - n) > 0:
        ix = ix_total.pop(0)
        if n_tol_elem is None or values[ix] <= n_tol_elem:
            mask[ix] = True
            n += values[ix]

        if (n - n_tar) > n_tol:
            raise ValueError("Failed because of overshot tolerance.")

        if len(ix_total) == 0:
            raise ValueError("Failed because there are no elements left.")

    return torch.from_numpy(mask), n
