from abc import ABC, abstractmethod
from typing import Union, Iterable, List, Tuple

import torch


class FileMappedTensor(ABC):
    def __init__(self, file):
        """
        Memory-mapped tensor. Note that data is loaded only to the extent to which the object is accessed through brackets '[ ]'
        Therefore, this tensor has no value and no state until it is sliced and then returns a torch tensor.
        You can of course enforce loading the whole tiff by tiff_tensor[:]
        Args:
            file: path to tiff file
            dtype: data type to which to convert
        """
        self._file = file

    def __getitem__(self, pos) -> torch.Tensor:

        # convert to tuple if not already
        if not isinstance(pos, tuple):
            pos = tuple([pos])

        return self._load(pos[0])[pos[1:]]

    def __setitem(self, key, value):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def _load(self, pos) -> torch.Tensor:
        raise NotImplementedError


class MultiMappedTensor(FileMappedTensor):
    def __init__(self, files: Union[list, tuple, dict], loader):
        super().__init__(file=None)

        self._files = files
        self._loader = loader

    def __len__(self):
        return len(self._files)

    def _load(self, pos) -> torch.Tensor:
        if isinstance(pos, int):
            pos = (pos, )
        else:
            pos = torch.arange(len(self))[pos]

        return torch.stack([self._loader(self._files[k]) for k in pos], 0)
