import collections
import itertools
import warnings

from pytorch_utils import label
from typing import Optional, Callable

import torch
from torch.utils.data._utils.collate import default_collate


def get_all_labels(ds: torch.utils.data.Dataset, label_ix: int = 1,
                    extract_fn: Optional[Callable] = None,
                    batch_size=64, num_workers=4) -> torch.Tensor:
    """
    Get all unique labels of a ds

    Args:
        ds: dataset
        label_index: tuple index of the labels. Defaults to 1 because most stereo-type return is x, y
        extract_fn: extract label from dataset getitem (e.g. useful if numeric label is somewhere in a dictionary)
        batch_size:
        num_workers: number of workers for auxiliary dl

    """
    class LabelOnlyDS(torch.utils.data.Dataset):
        """In order to be able to ignore the non label stuff which could confuse the temporary dataloader"""
        def __init__(self, ds_core, label_ix, extract_fn):
            super().__init__()
            self.ds_core = ds_core
            self.label_ix = label_ix
            self.extract_fn = extract_fn

        def __len__(self):
            return self.ds_core.__len__()

        def __getitem__(self, ix):
            if self.extract_fn is not None:
                return self.extract_fn(self.ds_core.__getitem__(ix))
            elif self.label_ix is not None:
                return self.ds_core.__getitem__(ix)[self.label_ix]
            else:
                return self.ds_core.__getitem__(ix)

    if label_ix is not None and extract_fn is not None:
        warnings.warn("Ignoring label_ix because extract_fn was defined. Pass label_ix=None to avoid this warning.")
        label_ix = None

    ds = LabelOnlyDS(ds_core=ds, label_ix=label_ix, extract_fn=extract_fn)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=flexible_collate)

    label_cache = set()

    for y in dl:
        numeric = True if isinstance(y, torch.LongTensor) else False

        if numeric:
            assert isinstance(y, torch.LongTensor)
            label_cache = label_cache | set(y.unique().tolist())
        else:
            label_cache = label_cache | {yi for yi in itertools.chain(*y)}


    if numeric:
        label_cache = torch.LongTensor(list(label_cache))

    return label_cache


def flexible_collate(batch):
    elem = batch[0]

    if isinstance(elem, collections.Sequence):
        return batch

    else:
        return default_collate(batch)