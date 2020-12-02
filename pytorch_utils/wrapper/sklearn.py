from pytorch_utils.lazy import cycle


def torchify(sklearn_module):
    """Let's return torch tensor from common sklearn modules"""

    sklearn_module.inverse_transform = cycle.torch_np_cycle(0, 0)(sklearn_module.inverse_transform)
    sklearn_module.fit_transform = cycle.torch_np_cycle(0, 0)(sklearn_module.fit_transform)
    sklearn_module.transform = cycle.torch_np_cycle(0, 0)(sklearn_module.transform)

    return sklearn_module
