import pytest
import torch
import numpy as np

from pytorch_utils.data import unbind, recurse_numpy


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """True if tensors share the same storage"""
    return x.storage().data_ptr() == y.storage().data_ptr()


@pytest.mark.parametrize("clone", [True, False])
def test_unbind(clone):
    x = torch.rand(5, 32, 32)
    xx = unbind(x, 0, clone)

    if clone:
        assert not same_storage(xx[0], xx[1])
    else:
        assert same_storage(xx[0], xx[1])


@pytest.mark.parametrize("x,expct", [
    (torch.tensor(5), np.array(5)),
    ([[torch.tensor(5)], torch.tensor(5)], [[torch.tensor(5)], np.array(5)]),
    ({"a": torch.tensor(5)}, {"a": np.array(5)}),
    ({"a": {"n": torch.tensor(5)}, "b": torch.tensor(5)}, {"a": {"n": torch.tensor(5)}, "b": np.array(5)}),
])
def test_recurse_numpy(x, expct):
    out = recurse_numpy(x, 1)

    assert out == expct
