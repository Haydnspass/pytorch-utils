import pytest

import torch

from pytorch_utils.label import label


def test_get_all_labels():
    x = torch.rand(100, 3, 32, 32)
    y = torch.arange(100)

    ds = torch.utils.data.TensorDataset(x, y)
    yout = label.get_all_labels(ds)

    assert (y == yout).all()
