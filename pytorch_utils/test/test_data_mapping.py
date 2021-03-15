from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


class TestFileMappedTensor(ABC):

    def test_load(self, tensor):

        assert isinstance(tensor._load(1), torch.Tensor), "Failed on single index load."
        assert isinstance(tensor._load(slice(1, 10, 2)), torch.Tensor), "Failed on sliced load."

    def test_getitem(self, tensor):

        assert isinstance(tensor[1], torch.Tensor), "Failed on single index getitem"
        assert isinstance(tensor[1, 10:20, 5:30], torch.Tensor), "Failed on single index getitem followed by slice"
        assert isinstance(tensor[1:5], torch.Tensor), "Failed on sliced index"
        assert isinstance(tensor[1:5, 10:20, 5:30], torch.Tensor), "Failed on sliced index followed by slice"


class TestMultiMappedTensor(TestFileMappedTensor):
    @pytest.fixture()
    def tensor(self, sample_pngs):
        def loader(f):
            return torch.from_numpy(plt.imread(f))

        return mapping.MultiMappedTensor(list(sample_pngs.glob('*.png')), loader)

    def test_len(self, tensor, sample_pngs):
        assert len(tensor) == len(list(sample_pngs.glob('*.png')))

    def test_load(self, tensor):
        super().test_load(tensor)

        assert tensor._load(1).size() == torch.Size([32, 40, 4])
        assert tensor._load(slice(1, 10, 2)).size() == torch.Size([5, 32, 40, 4])

