from typing import Optional

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import label


def tshow(t: torch.Tensor, autosqueeze: bool = False, ax=None, *args, **kwargs):
    """Tensor friendly plt.imshow"""

    t = t.detach().cpu()

    if t.dtype == torch.float16:  # short precision does not work for imshow
        t = t.to(torch.float32)

    if autosqueeze:
        t = t.squeeze()

    if t.dim() == 3:
        t = t.permute(1, 2, 0)

    if ax is None:
        return plt.imshow(t, *args, **kwargs)
    else:
        return ax.imshow(t, *args, **kwargs)


def plot_bboxes(boxes: torch.Tensor, scores: Optional[torch.Tensor] = None, box_mode='xyxy', order=None,
                ax=None, patch_kwargs=None, annotate_kwargs=None):
    """
    Plot bounding boxes on axis

    Args:
        boxes: size N x 4
        scores: size N
        box_mode: either 'xyxy' or 'xywh'
        order: None (mpl default) or 'swapped'
        ax: axis

    """

    if patch_kwargs is None:
        patch_kwargs = dict()

    if annotate_kwargs is None:
        annotate_kwargs = {'weight': 'bold', 'fontsize': 8}

    if ax is None:
        ax = plt.gca()

    if order == 'swapped':
        boxes = boxes[:, [1, 0, 3, 2]]

    boxes_xywh = label.bbox.convert_bbox(boxes, box_mode, 'xywh')

    rectangles = [mpl.patches.Rectangle(b[[1, 0]], b[3], b[2], fill='', **patch_kwargs) for b in boxes_xywh]

    for i, r in enumerate(rectangles):
        ax.add_patch(r)

        if scores is not None:
            rx, ry = r.get_xy()
            cx = rx
            cy = ry + r.get_height()

            ax.annotate(f'p: {scores[i]:.2f}', (cx, cy), **annotate_kwargs)

    return ax


def plot_keypoints(keypoints: torch.Tensor, graph: Optional[dict] = None, plot_ix=True, ix_prefix: str = '', ax=None):
    if ax is None:
        ax = plt.gca()

    if graph is not None:
        for k, v in graph.items():
            if v is not None:
                ax.plot(keypoints[[k, v], 0], keypoints[[k, v], 1], 'blue')

    ax.plot(keypoints[:, 0], keypoints[:, 1], 'ro')

    if plot_ix:
        for i, kp in enumerate(keypoints):
            plt.text(kp[0], kp[1], ix_prefix + str(i))

    return ax
