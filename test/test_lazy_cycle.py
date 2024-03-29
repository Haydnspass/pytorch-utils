import pytest
import torch
import numpy as np

from pytorch_utils.lazy import cycle


def test_cycle_minimal():
    def minimal(x):
        return 2 * x

    trafo_a = lambda x: x ** 2
    trafo_b = lambda x: x ** 0.5

    assert cycle.cycle(trafo_a, trafo_b, 0, 0)(minimal)(5) == pytest.approx(50**0.5, 1e-8)
    assert cycle.cycle(None, trafo_b, None, 0)(minimal)(5) == pytest.approx(10 ** 0.5, 1e-8)
    assert cycle.cycle(trafo_a, None, 0, None)(minimal)(5) == pytest.approx(50., 1e-8)


def test_cycle_kwargs():
    def minimal(*, x, y, z):
        return x + y * z

    trafo_a = lambda x: x ** 2
    trafo_b = lambda x: x ** 4

    assert cycle.cycle(trafo_a, trafo_b, 'z', 0)(minimal)(x=5, y=6, z=7) == pytest.approx((5 + 6 * 7**2)**4, 1e-8)


def test_cycle_on_tuple_return():
    def minimal(x):
        return x, 1, 2

    trafo_a = lambda x: x * 2
    trafo_b = lambda x: x[0]

    assert cycle.cycle(trafo_a, trafo_b, return_arg=0)(minimal)([10]) == (10, 1, 2)
    assert cycle.cycle(trafo_a, trafo_b, return_arg=None)(minimal)([10]) == [10, 10]


def test_torchify():
    def torch_only_func(x):
        assert isinstance(x, torch.Tensor)
        return x

    cycle.torchify(0)(torch_only_func)(np.random.rand(5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Makes no sense with only cpu")
@pytest.mark.parametrize("device_in", ['cpu', 'cuda:0'])
@pytest.mark.parametrize("device_out", ['cpu', 'cuda:0'])
def test_device(device_in, device_out):
    def device_only_func(x: torch.Tensor):
        assert str(x.device) == device_out

    cycle.auto_device(device_out, arg=0)(device_only_func)(torch.rand(5, device=device_in))


def test_torch_np_cycle():
    def numpy_only_func(x):
        assert isinstance(x, np.ndarray)
        return x

    assert isinstance(cycle.torch_np_cycle(0, 0)(numpy_only_func)(torch.rand(5, 5)), torch.Tensor)


def test_torch_list_cycle():
    def list_only_func(x):
        assert isinstance(x, list)
        return x

    assert isinstance(cycle.torch_list_cycle(0, 0)(list_only_func)(torch.rand(5, 5)), torch.Tensor)


def test_dict_dot_cycle():
    def dummy(x):
        assert x.a == 42
        return x

    assert cycle.dict_dot_cycle(0, None)(dummy)({'a': 42}) == {'a': 42}
    assert cycle.dict_dot_cycle(0, None, False)(dummy)({'a': 42}).a == 42
