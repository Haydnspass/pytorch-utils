from unittest import mock

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


def test_view_to_dim_dec():
    def _dummy(arg1, x: torch.Tensor, **kwargs):
        if x.dim() != 2:
            raise ValueError

    lazy.tensor.view_to_dim_dec(
        ndim=2,
        dim=0,
        unsqueeze=True,
        squeeze=True,
        arg=1
    )(_dummy)(None, torch.rand(5))


def test_cycle_view():
    @lazy.tensor.cycle_view(ndim=4, dim=0)
    def func(x):
        assert x.dim() == 4
        return x

    @lazy.tensor.cycle_view(ndim=4, ndim_diff=-1, dim=0)
    def reducer(x):
        assert x.dim() == 4
        return x[:, 1]

    # unsqueeze - squeeze
    x = torch.rand(10, 11)
    out = func(x.clone())
    assert (x == out).all()

    # squeeze - unsqueeze
    x = torch.rand(1, 1, 10, 11, 1)
    out = func(x.clone())
    assert (x == out).all()

    # test on reducing function (i.e. input ndim and target ndim are not the same)
    assert reducer(torch.rand(2, 3, 4)).size() == torch.Size([3, 4])


@pytest.mark.parametrize("p_in,p_expct", [
    ([0, 1, 2], [0, 1, 2]),  # identity
    ([0, 2, 1], [0, 2, 1]),  # swap last 2
    ([-3, -2, 1], [0, 1, 2]),  # identity with neg. ix
    ([0, -1, -2], [0, 2, 1]),  # swap last 2
])
def test_invert_permutation(p_in, p_expct):
    assert lazy.tensor.invert_permutation(*p_in) == p_expct

    # test alias function
    with mock.patch.object(torch.Tensor, "permute") as mock_permute:
        lazy.tensor.inverse_permute(torch.rand(2, 3, 4), p_in)

    mock_permute.assert_called_once_with(*p_expct)


@pytest.mark.parametrize("a,b", [
    ([3, 4, 5], [4, 5, 3]),
    ([2, 3, 4, 5], [2, 4, 5, 3]),
    ([9, 3, 38, 2], [9, 38, 2, 3])
])
def test_chw_hwc_reverse(a, b):
    @lazy.tensor.hwc_chw_cycle()
    def _chw(x):
        assert x.size(-3) == 3
        return x

    @lazy.tensor.chw_hwc_cycle()
    def _hwc(x):
        assert x.size(-1) == 3
        return x

    x = _chw(torch.rand(*b))
    assert x.size() == torch.Size(b)
    x = _hwc(torch.rand(*a))
    assert x.size() == torch.Size(a)

