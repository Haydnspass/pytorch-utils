from hypothesis import given, strategies, assume
import torch
import pytest
from unittest import mock

from pytorch_utils import lazy

@pytest.mark.parametrize("ds_len,val_fraction,expct", [(10, 0.1, (9, 1)),
                                                       (11, 0.1, (9, 2))])
def test_fract_random_split(ds_len, val_fraction, expct):

    ds = torch.utils.data.TensorDataset(torch.rand(ds_len, 3, 32, 32))

    ds_train, ds_val = lazy.data.fract_random_split(ds, val_fraction)

    assert len(ds_train) == expct[0]
    assert len(ds_val) == expct[1]

    # test with manual generator
    with mock.patch('torch.utils.data.random_split') as mock_split:
        _ = lazy.data.fract_random_split(ds, val_fraction, generator='dummy')

    mock_split.assert_called_once_with(ds, expct, generator='dummy')


@given(strategies.lists(strategies.integers(min_value=0, max_value=500), min_size=10))
def test_split_to_fill(vals):
    assume(torch.tensor(vals).sum() >= 500)

    try:
        mask, n = lazy.data.split_to_fill(vals, 500, 100, 100)
        assert 400 <= n <= 600
        assert isinstance(mask, torch.BoolTensor)
    except ValueError as err:
        if str(err) == "Failed because there are no elements left.":
            return
        if str(err) == "Failed because of overshot tolerance.":
            return
        assert False

