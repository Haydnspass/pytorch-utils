from pathlib import Path

import dill
import torch.utils.data
import yaml

from tqdm import tqdm


def _dump_dill(sample, file):
    with file.open('wb') as f:
        dill.dump(sample, file=f)


def _load_dill(file):
    with file.open('rb') as f:
        return dill.load(f)


class _MappedDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, load_fn):
        self._file_list = file_list
        self._load_fn = load_fn

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, item):
        if self._load_fn == 'torch':
            return torch.load(self._file_list[item])
        elif self._load_fn == 'dill':
            return _load_dill(self._file_list[item])
        elif callable(self._load_fn):
            return self._load_fn(self._file_list[item])
        else:
            raise ValueError


class _StaticDataset(_MappedDataset):
    def __init__(self, file_list, load_fn):
        super().__init__(file_list, load_fn)
        self._cache = [
            super(_StaticDataset, self).__getitem__(i)
            for i in range(len(self))
        ]

    def __getitem__(self, item):
        return self._cache[item]


def dump_dataset(ds: torch.utils.data.Dataset, path, dump_fn: 'torch', file_extension=''):
    """
    Iterates through dataset and dumps samples via dill to individual file.
    Convenience to make a dataset that processes on the fly process once and load afterwards.

    Args:
        ds: dataset
        path: directory where to save meta file and samples
    """

    path = Path(path) if not isinstance(path, Path) else path
    assert path.is_dir()

    for i, sample in tqdm(enumerate(ds), total=len(ds), desc='Dumping Dataset'):
        fname = path / f'sample_{i}'
        if dump_fn == 'torch':
            file_extension = '.pt'
            fname = fname.with_suffix(file_extension)
            torch.save(sample, fname)
        elif dump_fn == 'dill':
            file_extension = '.dill'
            fname = fname.with_suffix(file_extension)
            _dump_dill(sample, fname)
        elif callable(dump_fn):
            fname = fname.with_suffix(file_extension)
            dump_fn(sample, fname)
        else:
            raise ValueError(f"Non supported dump_fn. Supported are 'torch', 'dill' or specifying"
                             f"a callable directly.")

    meta = {
        'len': len(ds),
        'namespace': 'sample_',
        'file_extension': file_extension,
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
    if file_ext == '.dill':
        load_fn = 'dill'
    elif file_ext == '.pt':
        load_fn = 'torch'
    else:
        raise ValueError

    file_list = [path / (f'{file_base}{i}{file_ext}') for i in range(meta['len'])]

    if mode == 'mapped':
        return _MappedDataset(file_list, load_fn=load_fn)
    elif mode == 'static':
        return _StaticDataset(file_list, load_fn=load_fn)
    else:
        raise ValueError
