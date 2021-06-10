from pathlib import Path

import torch
import torch.utils.data
import yaml
import pytest

from pytorch_utils.data import dataset as ds_utils


def _equal_ds(ds_a, ds_b):
    assert len(ds_a) == len(ds_b), "Unequal length."

    for sa, sb in zip(ds_a, ds_b):
        if not isinstance(sa, torch.Tensor):
            return _equal_ds(sa, sb)

        return (sa == sb).all()


class TestDumpLoadDataset:
    @pytest.fixture()
    def ds_basic(self):
        ds = torch.utils.data.dataset.TensorDataset(
            torch.rand(5, 3, 32, 40),
            torch.rand(5)
        )
        return ds

    @pytest.mark.parametrize('dump_fn', ['torch', 'dill'])
    def test_dump_dataset(self, dump_fn, ds_basic, tmpdir):
        # basic
        ds_utils.dump_dataset(ds_basic, path=tmpdir, dump_fn=dump_fn)

        if dump_fn == 'dill':
            pttrn = '*.dill'
        elif dump_fn == 'torch':
            pttrn = '*.pt'
        else:
            raise ValueError

        assert len(list(Path(tmpdir).glob(pttrn))) == 5

        # meta
        with (Path(tmpdir) / 'meta.yaml').open('r') as f:
            meta = yaml.safe_load(f)

        assert meta['len'] == 5

    @pytest.mark.parametrize('dump_fn', ['torch', 'dill'])
    @pytest.mark.parametrize('mode', ['mapped', 'static'])
    def test_dump_load_dataset(self, dump_fn, mode, ds_basic, tmpdir):
        ds_utils.dump_dataset(ds_basic, path=tmpdir, dump_fn=dump_fn)
        ds_re = ds_utils.load_from_dump(tmpdir, mode)

        assert _equal_ds(ds_basic, ds_re)
