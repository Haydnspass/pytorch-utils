import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import label


def tshow(t: torch.Tensor, autosqueeze: bool = False, ax=None, *args, **kwargs):
    """Tensor friendly plt.imshow"""

    t = t.detach().cpu()
    if autosqueeze:
        t = t.squeeze()

    if t.dim() == 3:
        t = t.permute(1, 2, 0)

    if ax is None:
        return plt.imshow(t, *args, **kwargs)
    else:
        ax.imshow(t, *args, **kwargs)
        return ax


def plot_bboxes(boxes: torch.Tensor, box_mode='xyxy', order=None, ax=plt.gca()):
    """
    Plot bounding boxes on axis

    Args:
        boxes: size N x 4
        box_mode: either 'xyxy' or 'xywh'
        order: None (mpl default) or 'swapped'
        ax: axis

    """

    if order == 'swap':
        boxes = boxes[:, [1, 0, 3, 2]]

    boxes_xywh = label.bbox.convert_bbox(boxes, box_mode, 'xywh')

    for b in boxes_xywh:
        ax.add_patch(mpl.patches.Rectangle(b[[1, 0]], b[3], b[2], fill=''))

    return ax
