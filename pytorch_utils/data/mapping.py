from abc import ABC, abstractmethod
from typing import Union, Iterable, List, Tuple, Callable, Optional

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

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def _load(self, pos) -> torch.Tensor:
        raise NotImplementedError


class MultiMappedTensor(FileMappedTensor):
    def __init__(self, files: Union[list, tuple, dict], loader: Callable):
        """
        Map multiple files to single pytorch tensor (e.g. a list of .png files).

        Args:
            files: paths
            loader: callable that is able to load an individual file and returns a tensor

        """
        super().__init__(file=None)

        self._files = files
        self._loader = loader

    def size(self, dim: Optional[int] = None):
        s = torch.Size([len(self), *self[0].size()])
        return s[dim] if dim is not None else s

    def __len__(self):
        return len(self._files)

    def _load(self, pos) -> torch.Tensor:
        if isinstance(pos, int):
            pos = (pos, )
            squeeze_batch_dim = True
        else:
            pos = torch.arange(len(self))[pos]
            squeeze_batch_dim = False

        data = torch.stack([self._loader(self._files[k]) for k in pos], 0)

        if squeeze_batch_dim:
            return data.squeeze(0)
        else:
            return data


class Delayed:
    def __init__(self, tensors: Iterable[Union[MultiMappedTensor, torch.Tensor]], fn: Callable):
        """
        Delays a function that operates on (most interestingly) mapped tensors.
        For example you have two mapped tensors a (size Nx2xHxW), b (size Nx3xHxW)
        which should be concatenated such that the result is Nx5xHxW but a and b
        do not fit into memory.

        Warning:
            The (delayed) function must not operate on the batch dimension

        Args:
            tensors: iterable of tensors
            fn: delayed function
        """
        self._tensors = tensors
        self._fn = fn

    def __getitem__(self, pos) -> torch.Tensor:
        return self._fn(*[t[pos] for t in self._tensors])
