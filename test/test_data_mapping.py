from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing
import pytest
import torch

from pytorch_utils.data import mapping


@pytest.fixture
def sample_numpy(tmpdir):
    path = Path(tmpdir / 'sample_batch.npy')

    map = np.memmap(path, dtype='float32', mode='w+', shape=(1000, 32, 40))
    map[:] = torch.rand(1000, 32, 40).numpy()
    map.flush()

    return path


@pytest.fixture
def sample_pngs(tmpdir):
    img_dir = Path(tmpdir / 'samples')
    img_dir.mkdir()

    for i in range(10):
        plt.imsave(img_dir / f'sample_{i}.png', np.random.rand(32, 40, 4))

    return img_dir


@pytest.fixture
def sample_pngs_discontinous(sample_pngs):
    for ix in [2, 8]:
        (sample_pngs / f"sample_{ix}.png").unlink(missing_ok=False)

    return sample_pngs


class _TestFileMappedTensor(ABC):

    def test_dim(self, tensor):
        assert isinstance(tensor.dim(), int)

    def test_load(self, tensor):

        assert isinstance(tensor._load(1), torch.Tensor), "Failed on single index load."
        assert isinstance(tensor._load(slice(1, 10, 2)), torch.Tensor), "Failed on sliced load."

    def test_getitem_singular(self, tensor):

        assert isinstance(tensor[1], torch.Tensor), "Failed on single index getitem"
        assert isinstance(tensor[1, 10:20, 5:30], torch.Tensor), "Failed on single index getitem followed by slice"

    def test_getitem_slice(self, tensor):
        assert isinstance(tensor[1:5], torch.Tensor), "Failed on sliced index"
        assert isinstance(tensor[1:5, 10:20, 5:30], torch.Tensor), "Failed on sliced index followed by slice"


class TestMultiMappedList(_TestFileMappedTensor):
    @pytest.fixture()
    def tensor(self, sample_pngs_discontinous):
        def loader(f):
            if f.is_file():
                return torch.from_numpy(plt.imread(f))
            else:
                return None

        return mapping.MultiMapped([sample_pngs_discontinous / f"sample_{i}.png" for i in range(10)], loader)

    def test_dim(self, tensor):
        return

    def test_len(self, tensor):
        assert len(tensor) == 10

    def test_load(self, tensor):
        assert isinstance(tensor._load(0), torch.Tensor)
        assert tensor._load(2) is None

    def test_getitem_slice(self, tensor):
        return

    def test_getitem_singular(self, tensor):
        return


class TestMultiMappedTensor(_TestFileMappedTensor):
    @pytest.fixture()
    def tensor(self, sample_pngs):
        def loader(f):
            return torch.from_numpy(plt.imread(f))

        return mapping.MultiMappedTensor(list(sample_pngs.glob('*.png')), loader)

    def test_len(self, tensor, sample_pngs):
        assert len(tensor) == len(list(sample_pngs.glob('*.png')))

    def test_dim(self, tensor):
        super().test_dim(tensor)

        assert tensor.dim() == 4

    def test_size(self, tensor, sample_pngs):
        assert tensor.size() == torch.Size([len(list(sample_pngs.glob('*.png'))), 32, 40, 4])

    def test_getitem_slice(self, tensor):
        super().test_getitem_slice(tensor)
        # test against native impl
        tensor_nat = tensor[:]

        assert tensor.size() == tensor_nat.size()
        assert tensor[:].size() == tensor_nat[:].size()
        assert tensor[0].size() == tensor_nat[0].size()
        assert tensor[0, :].size() == tensor_nat[0, :].size()
        assert tensor[:, 0, :].size() == tensor_nat[:, 0, :].size()
        assert tensor[0, 0].size() == tensor_nat[0, 0].size()
        assert tensor[:, :-1].size() == tensor_nat[:, :-1].size()

        # ToDo: Implement ellipsis
        # assert tensor[..., 0].size() == tensor_nat[..., 0].size()

    def test_load(self, tensor):
        super().test_load(tensor)

        assert tensor._load(1).size() == torch.Size([32, 40, 4])
        assert tensor._load(slice(1, 10, 2)).size() == torch.Size([5, 32, 40, 4])


@pytest.mark.parametrize("x,y,fn", [
    (torch.rand(20, 3, 40, 32), torch.rand(20, 2, 40, 32), lambda a, b: torch.cat([a, b], dim=1)),
    (torch.rand(20, 3, 40, 32), torch.rand(20, 3, 40, 32), lambda a, b: torch.add(a, b)),
])
def test_delayed(x, y, fn):

    expct = fn(x, y)
    cat_delayed = mapping.Delayed([x, y], fn)

    numpy.testing.assert_array_almost_equal(
        cat_delayed[:].numpy(),
        expct[:].numpy()
    )

    numpy.testing.assert_array_almost_equal(
        cat_delayed[0].numpy(),
        expct[0].numpy()
    )

    numpy.testing.assert_array_almost_equal(
        cat_delayed[5:10].numpy(),
        expct[5:10].numpy()
    )