import torch


def unbind(t: torch.Tensor, dim: int = 0, clone: bool = False) -> tuple:
    """
    Unbind with clone option

    Args:
        t: tensor
        dim: dimension to unbind
        clone: clone elements such that they don't share the same base

    Returns:
        tuple of tensors
    """
    b = torch.unbind(t, dim=dim)
    if clone:
        b = [bb.clone() for bb in b]

    return b