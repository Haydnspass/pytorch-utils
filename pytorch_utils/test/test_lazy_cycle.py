import pytest
import torch
import numpy as np
import sklearn

from pytorch_utils.lazy import cycle


class TestCycle:

    @pytest.fixture()
    def minimal(self):
        def minimal_func(x):
            return 2 * x

        return minimal_func

    @pytest.fixture()
    def numpy_only_func(self):
        def np_func(x: np.array):
            assert isinstance(x, np.ndarray)
            return x

        return np_func

    def test_minimal(self, minimal):
        trafo_a = lambda x: x ** 2
        trafo_b = lambda x: x ** 0.5

        assert cycle.cycle(trafo_a, trafo_b, 0, 0)(minimal)(5) == pytest.approx(50**0.5, 1e-8)
        assert cycle.cycle(None, trafo_b, None, 0)(minimal)(5) == pytest.approx(10 ** 0.5, 1e-8)
        assert cycle.cycle(trafo_a, None, 0, None)(minimal)(5) == pytest.approx(50., 1e-8)

    def test_torch_cycle(self, numpy_only_func):

        assert isinstance(cycle.torch_np_cycle(0, 0)(numpy_only_func)(torch.rand(5, 5)), torch.Tensor)

