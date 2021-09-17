from typing import Optional, List

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import label
from .. import lazy


@lazy.cycle.torchify(0)
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


@lazy.cycle.torchify(0)
@lazy.cycle.auto_device('cpu', 0)
@lazy.tensor.view_to_dim_dec(2, 0, arg=0)
def plot_bboxes(boxes: torch.Tensor, /, scores: Optional[torch.Tensor] = None, *,
                box_mode='xyxy', order=None, ax=None, patch_kwargs=None, annotate_kwargs=None):
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


@lazy.cycle.torchify(0)
def plot_keypoints(keypoints: torch.Tensor, graph: Optional[List[tuple]] = None, plot_ix=True,
                   ix_prefix: str = '', ax=None):
    if ax is None:
        ax = plt.gca()

    keypoints = keypoints.detach().cpu()

    if graph is not None:
        for k, v in graph:
            if v is not None:
                ax.plot(keypoints[[k, v], 0], keypoints[[k, v], 1], 'blue')

    ax.plot(keypoints[:, 0], keypoints[:, 1], 'ro')

    if plot_ix:
        for i, kp in enumerate(keypoints):
            ax.text(kp[0], kp[1], ix_prefix + str(i))

    return ax


class PlotKeypoints:
    def __init__(self, graph, plot_ix: bool = True, ix_prefix: str = ''):
        self.graph = graph
        self.plot_ix = plot_ix
        self.ix_prefix = ix_prefix

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self, keypoints, ax=None):
        plot_keypoints(keypoints, self.graph, plot_ix=self.plot_ix, ix_prefix=self.ix_prefix, ax=ax)