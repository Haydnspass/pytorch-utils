import pytest
import torch
import numpy as np
import sklearn

from pytorch_utils.lazy import cycle


def test_cycle_minimal():
    def minimal(x):
        return 2 * x

    trafo_a = lambda x: x ** 2
    trafo_b = lambda x: x ** 0.5

    assert cycle.cycle(trafo_a, trafo_b, 0, 0)(minimal)(5) == pytest.approx(50**0.5, 1e-8)
    assert cycle.cycle(None, trafo_b, None, 0)(minimal)(5) == pytest.approx(10 ** 0.5, 1e-8)
    assert cycle.cycle(trafo_a, None, 0, None)(minimal)(5) == pytest.approx(50., 1e-8)


def test_cycle_on_tuple_return():
    def minimal(x):
        return x, 1, 2

    trafo_a = lambda x: x * 2
    trafo_b = lambda x: x[0]

    assert cycle.cycle(trafo_a, trafo_b, return_arg=0)(minimal)([10]) == (10, 1, 2)
    assert cycle.cycle(trafo_a, trafo_b, return_arg=None)(minimal)([10]) == [10, 10]


def test_torch_cycle():
    def numpy_only_func(x):
        assert isinstance(x, np.ndarray)
        return x

    assert isinstance(cycle.torch_np_cycle(0, 0)(numpy_only_func)(torch.rand(5, 5)), torch.Tensor)

