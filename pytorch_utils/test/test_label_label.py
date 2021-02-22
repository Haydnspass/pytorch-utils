import pytest
from unittest import mock

import sklearn.preprocessing
import torch

from pytorch_utils.label import label


def test_get_all_labels():
    x = torch.rand(100, 3, 32, 32)
    y = torch.arange(100)

    ds = torch.utils.data.TensorDataset(x, y)
    yout = label.get_all_labels(ds)

    assert (y == yout).all()


def test_get_all_labels_complex():
    class AdvancedLabelDS(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, ix):
            return torch.rand(3, 32, 32), {'objects': torch.arange(ix, ix + 3)}

    ds = AdvancedLabelDS()
    y_out = label.get_all_labels(AdvancedLabelDS(), extract_fn=lambda x: x['objects'])

    assert (y_out == torch.arange(102)).all()


def test_get_all_labels_nonnumeric():
    class AdvancedLabelDS(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, ix):
            return torch.rand(3, 32, 32), ['bike', 'screen']

    ds = AdvancedLabelDS()
    y_out = label.get_all_labels(AdvancedLabelDS())

    assert y_out == {'bike', 'screen'}
