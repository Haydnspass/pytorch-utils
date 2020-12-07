import math
from typing import Optional

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
