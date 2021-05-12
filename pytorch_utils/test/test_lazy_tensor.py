import pytest
import torch

from pytorch_utils import lazy


@pytest.mark.parametrize("x,dim,ndim", [
    (torch.rand(1, 5, 3), 0, 2),
    (torch.rand(1, 5, 4), 0, 3),
    (torch.rand(5, 4), -1, 3),
])
def test_view_to_dim(x, dim, ndim):

    out = lazy.tensor.view_to_dim(x.clone(), ndim, dim)
    assert out.dim() == ndim


def test_view_to_dim_expct():
    # to few dims of x
    with pytest.raises(ValueError):
        lazy.tensor.view_to_dim(torch.rand(5, 4), 1, 0)
