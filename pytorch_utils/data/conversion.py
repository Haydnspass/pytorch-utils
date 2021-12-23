import numpy as np
import torch


def torch_cv2(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        return img.permute(1, 2, 0)[..., [-1, 1, 0]].numpy()
    return img.numpy()


def cv2_torch(img: np.array) -> torch.Tensor:
    if img.ndim == 3:
        return torch.from_numpy(img).permute(-1, 0, 1)[[-1, 1, 0]]
    return torch.from_numpy(img)