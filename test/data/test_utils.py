import pytest
import torch

from pytorch_utils.data import unbind


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
