import yaml
import dill
import torch.utils.data
from pathlib import Path


class _DillDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self._file_list = file_list

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, item):
        with self._file_list[item].open('rb') as f:
            return dill.load(f)


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

    for i, sample in enumerate(ds):
        with (path / f'sample_{i}.dill').open('wb') as f:
            dill.dump(sample, file=f)

    meta = {
        'len': len(ds),
        'namespace': 'sample_',
        'file_extension': '.dill',
    }
    with (path / 'meta.yaml').open('w') as f:
        yaml.dump(meta, stream=f)


def load_from_dump(path: Path) -> torch.utils.data.Dataset:
    """
    Construct dataset from dumped one. Behaves like the original one.

    Args:
        path: directory of meta and samples
    """
    path = Path(path) if not isinstance(path, Path) else path

    assert path.is_dir()
    with (path / 'meta.yaml').open('r') as f:
        meta = yaml.safe_load(f)

    file_base = meta['namespace']
    file_ext = meta['file_extension']
    file_list = [path / (f'{file_base}{i}{file_ext}') for i in range(meta['len'])]

    return _DillDataset(file_list)
