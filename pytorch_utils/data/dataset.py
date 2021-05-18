from pathlib import Path

import dill
import torch.utils.data
import yaml

from tqdm import tqdm


class _DillMappedDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self._file_list = file_list

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, item):
        with self._file_list[item].open('rb') as f:
            return dill.load(f)


class _DillStaticDataset(_DillMappedDataset):
    def __init__(self, file_list):
        super().__init__(file_list)
        self._cache = [
            super(_DillStaticDataset, self).__getitem__(i)
            for i in range(len(self))
        ]

    def __getitem__(self, item):
        return self._cache[item]


def dump_dataset(ds: torch.utils.data.Dataset, path):
    """
    Iterates through dataset and dumps samples via dill to individual file.
    Convenience to make a dataset that processes on the fly process once and load afterwards.

    Args:
        ds: dataset
        path: directory where to save meta file and samples
    """
    path = Path(path) if not isinstance(path, Path) else path
    assert path.is_dir()

    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        with (path / f'sample_{i}.dill').open('wb') as f:
            dill.dump(sample, file=f)

    meta = {
        'len': len(ds),
        'namespace': 'sample_',
        'file_extension': '.dill',
    }
    with (path / 'meta.yaml').open('w') as f:
        yaml.dump(meta, stream=f)


def load_from_dump(path: Path, mode: str = 'mapped') -> torch.utils.data.Dataset:
    """
    Construct dataset from dumped one. Behaves like the original one.

    Args:
        path: directory of meta and samples
        mode: 'mapped' or 'static' (ally) loaded
    """
    path = Path(path) if not isinstance(path, Path) else path

    assert path.is_dir()
    with (path / 'meta.yaml').open('r') as f:
        meta = yaml.safe_load(f)

    file_base = meta['namespace']
    file_ext = meta['file_extension']
    file_list = [path / (f'{file_base}{i}{file_ext}') for i in range(meta['len'])]

    if mode == 'mapped':
        return _DillMappedDataset(file_list)
    elif mode == 'static':
        return _DillStaticDataset(file_list)
    else:
        raise ValueError
