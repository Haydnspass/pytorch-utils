import torch

from pytorch_utils.label import label


def test_get_all_labels():
    x = torch.rand(100, 3, 32, 32)
    y = torch.arange(100)

    ds = torch.utils.data.TensorDataset(x, y)
    yout = label.get_all_labels(ds)

    assert (y == yout).all()


class _LabelDSDict(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, ix):
        return torch.rand(3, 32, 32), {'objects': torch.arange(ix, ix + 3)}


def _label_ds_dict_extract_fn(x):
    return x[1]['objects']


def test_get_all_labels_complex():

    ds = _LabelDSDict()
    y_out = label.get_all_labels(_LabelDSDict(), extract_fn=_label_ds_dict_extract_fn)

    assert (y_out == torch.arange(102)).all()


class _LabelDSUnequal(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, ix):
        if ix == 0:
            return ['a']
        else:
            return ['a', 'b']


def test_get_all_label_unequal():

    y_out = label.get_all_labels(_LabelDSUnequal(), label_ix=None, num_workers=4)

    assert sorted(y_out) == ['a', 'b']


class _LabelDSNonNumeric(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, ix):
        return torch.rand(3, 32, 32), ['bike', 'screen']


def test_get_all_labels_nonnumeric():

    ds = _LabelDSNonNumeric()
    y_out = label.get_all_labels(_LabelDSNonNumeric())

    assert y_out == {'bike', 'screen'}
