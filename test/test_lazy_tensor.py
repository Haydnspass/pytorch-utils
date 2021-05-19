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


def test_cycle_view():
    @lazy.tensor.cycle_view(ndim=4, dim=0)
    def func(x):
        assert x.dim() == 4
        return x

    # unsqueeze - squeeze
    x = torch.rand(10, 11)
    out = func(x.clone())
    assert (x == out).all()

    # squeeze - unsqueeze
    x = torch.rand(1, 1, 10, 11, 1)
    out = func(x.clone())
    assert (x == out).all()
