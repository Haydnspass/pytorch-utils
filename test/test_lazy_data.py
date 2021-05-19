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
