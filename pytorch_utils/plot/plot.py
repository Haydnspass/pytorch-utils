from typing import Optional, List

import cv2
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import label
from .. import lazy


@lazy.cycle.torchify(0)
def tshow(t: torch.Tensor, autosqueeze: bool = False, ax=None, im=None, backend: str = "matplotlib",
          *args, **kwargs):
    """
    Tensor friendly plt.imshow

    Args:
        backend: matplotlib or cv2
    """

    t = t.detach().cpu()

    if t.dtype == torch.float16:  # short precision does not work for imshow
        t = t.to(torch.float32)

    if autosqueeze:
        t = t.squeeze()

    if t.dim() == 3:
        t = t.permute(1, 2, 0)

    if backend == "cv2":
        t = t.numpy()
        if t.ndim == 3:
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        return cv2.imshow("Figure", t)

    if im is not None:
        return im.set_data(t, *args, **kwargs)

    if ax is None:
        return plt.imshow(t, *args, **kwargs)
    else:
        return ax.imshow(t, *args, **kwargs)


@lazy.cycle.torchify(0)
@lazy.cycle.auto_device('cpu', 0)
@lazy.tensor.view_to_dim_dec(2, 0, arg=0)
def plot_bboxes(boxes: torch.Tensor, /, scores: Optional[torch.Tensor] = None, *,
                box_mode='xyxy', order=None, ax=None,
                img: torch.Tensor = None, img_mode: str = "torch",
                patch_kwargs=None, annotate_kwargs=None):
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

    if img is None:
        boxes_xywh = label.bbox.convert_bbox(boxes, box_mode, 'xywh')

        rectangles = [mpl.patches.Rectangle(b[[1, 0]], b[3], b[2], fill='', **patch_kwargs) for b in boxes_xywh]

        for i, r in enumerate(rectangles):
            ax.add_patch(r)

            if scores is not None:
                rx, ry = r.get_xy()
                cx = rx
                cy = ry + r.get_height()

                scores = scores.unsqueeze(0) if scores.dim() == 0 else scores
                ax.annotate(f'p: {scores[i]:.2f}', (cx, cy), **annotate_kwargs)

        return ax
    else:
        img = torch_cv2(img) if img_mode == "torch" else img
        color = [255, 0, 0] if "color" not in patch_kwargs else patch_kwargs["color"]
        thickness_rect = 1 if "thickness" not in patch_kwargs else patch_kwargs["thickness"]

        boxes = boxes[..., [1, 0, 3, 2]]
        boxes = boxes.to(torch.int)
        for b in boxes:
            img = cv2.rectangle(img, b[:2].tolist(), b[2:].tolist(), color=color, thickness=thickness_rect)

        img = cv2_torch(img) if img_mode == "torch" else img
        return img


@lazy.cycle.torchify(0)
def plot_keypoints(keypoints: torch.Tensor, graph: Optional[List[tuple]] = None, plot_ix=True,
                   ix_prefix: str = '', ax=None, plot_3d: bool = False,
                   img=None, img_mode="torch"):
    if ax is None:
        ax = plt.gca()

    if img is not None and img_mode == "torch":
        img = torch_cv2(img)

    keypoints = keypoints.detach().cpu()

    if graph is not None:
        for k, v in graph:
            if v is not None:
                if img is not None:
                    pt0 = keypoints[[k, v]][0].to(torch.int).tolist()
                    pt1 = keypoints[[k, v]][1].to(torch.int).tolist()
                    img = cv2.line(img, pt0, pt1, color=[255, 0, 0], thickness=2)
                elif not plot_3d:
                    ax.plot(keypoints[[k, v], 0], keypoints[[k, v], 1], color='blue')
                else:
                    ax.plot(
                        keypoints[[k, v], 0],
                        keypoints[[k, v], 1],
                        keypoints[[k, v], 2],
                        color='blue')

    if img is not None:
        for p in keypoints:
            img = cv2.circle(img, p[:2].to(torch.int).tolist(),
                             radius=3, color=[0, 0, 255], thickness=-1)
    elif not plot_3d:
        ax.plot(keypoints[:, 0], keypoints[:, 1], 'ro')
    else:
        ax.plot(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 'ro')

    if plot_ix:
        for i, kp in enumerate(keypoints):
            if img is not None:
                pass
            elif not plot_3d:
                ax.text(kp[0], kp[1], ix_prefix + str(i))
            else:
                ax.text(kp[0], kp[1], kp[2], ix_prefix + str(i))

    if img is None:
        return ax
    if img_mode == "torch":
        img = cv2_torch(img)
    return img


def torch_cv2(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3:
        return img.permute(1, 2, 0)[..., [-1, 1, 0]].numpy()
    return img.numpy()


def cv2_torch(img: np.array) -> torch.Tensor:
    if img.ndim == 3:
        return torch.from_numpy(img).permute(-1, 0, 1)[[-1, 1, 0]]
    return torch.from_numpy(img)


class PlotKeypoints:
    def __init__(self, graph, plot_ix: bool = True, ix_prefix: str = ''):
        self.graph = graph
        self.plot_ix = plot_ix
        self.ix_prefix = ix_prefix

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self, keypoints, ax=None):
        plot_keypoints(keypoints, self.graph, plot_ix=self.plot_ix, ix_prefix=self.ix_prefix, ax=ax)