from typing import Tuple

import torch

import pytorch_utils.lazy.tensor


class BBox:
    def __init__(self, data: torch.Tensor, mode: str = 'xyxy'):
        self._data = convert_bbox(
            data if isinstance(data, torch.Tensor) else torch.Tensor(data),
            mode_in=mode,
            mode_out='xyxy'
        )

    @property
    def xyxy(self):
        return convert_bbox(self._data, mode_in='xyxy', mode_out='xyxy')

    @property
    def xywh(self):
        return convert_bbox(self._data, mode_in='xyxy', mode_out='xywh')

    @property
    def cxcywh(self):
        return convert_bbox(self._data, mode_in='xyxy', mode_out='cxcywh')

    @property
    def area(self):
        return self.xywh[..., 2] * self.xywh[..., 3]

    def __eq__(self, other) -> bool:
        return (self.xyxy == other.xyxy).all()

    def check(self):
        self.check_area()

    def check_area(self):
        if self.area <= 0:
            raise ValueError("Bounding box(es) are not of valid area.")

        return self

    def check_fits_img(self, img_size: torch.Size, order: str = 'matplotlib', eps_border: float = 1e-6):
        img_size = _unified_img_size(img_size, order=order)

        if (self.xyxy < 0).any() or \
           (self.xyxy[..., [0, 2]] > img_size[0] - eps_border).any() or \
           (self.xyxy[..., [1, 3]] > img_size[1] - eps_border).any():
            raise ValueError("Bounding box(es) are outside of the specified image size.")

        return self


def convert_bbox(box: torch.Tensor, mode_in: str, mode_out: str) -> torch.Tensor:
    """
    Convert bounding boxes in format (N x )4 between xyxy, xywh, cxcywh formats.

    Formats:
        xyxy: upper left, lower right corner
        xywh: upper left, width and height
        cxcywh: center x/y, width and height

    Args:
        box: input boxes
        mode_in: xyxy or xywh or cxcywh
        mode_out: xyxy or xywh or cxcywh

    """
    if mode_in == mode_out:
        return box

    # always convert to xyxy first and derive from that base
    box_xyxy = _bbox_arbitrary_to_xyxy(box, mode=mode_in)
    box_out = _bbox_xyxy_to_arbitrary(box_xyxy, mode=mode_out)

    return box_out


def limit_bbox(box: torch.Tensor, img_size: torch.Size, mode='xyxy', eps_border=1e-6,
               order='matplotlib'):
    """
    Limit bounding boxes to image (size).

    Args:
        box: bounding boxes
        img_size: image size to which to limit to
        mode: mode of input bbox
        eps_border: border to next pixel
        order: 'matplotlib' interprets the x coordinate belonging to the last element of
         image size and y to the second last. Alternatively "None" which does not alter
         orders.
    """
    box = convert_bbox(box, mode, 'xyxy')
    img_size = _unified_img_size(img_size, order)

    box = box.clamp(0.)
    box[..., [0, 2]] = box[..., [0, 2]].clamp(max=img_size[-2] - eps_border)
    box[..., [1, 3]] = box[..., [1, 3]].clamp(max=img_size[-1] - eps_border)

    check_bbox(box, mode='xyxy')

    return convert_bbox(box, 'xyxy', mode)


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def shift_bbox_inside_img(box, img_size: torch.Size, mode='xyxy', eps_border=1e-6,
                          order: str = 'matplotlib'):
    """Shift bounding boxes inside image (without altering their size)."""

    box_xyxy = convert_bbox(box, mode, 'xyxy')
    img_size = _unified_img_size(img_size, order)

    shift_min = torch.max(
        -box_xyxy[:, :2],
        torch.zeros_like(box_xyxy[:, :2])
    )
    shift_max = torch.min(
        (img_size - eps_border) - box_xyxy[:, 2:],
        torch.zeros_like(box_xyxy[:, :2])
    )

    # shifts must not be non-zero at the same positions
    if (shift_min * shift_max).any():
        raise ValueError("Bounding boxes can not be shifted such that they fit inside the specified"
                         "image size.")

    box_xyxy += shift_min.repeat(1, 2)
    box_xyxy += shift_max.repeat(1, 2)

    check_bbox(box_xyxy, mode='xyxy')
    box_out = convert_bbox(box_xyxy, 'xyxy', mode)

    return box_out


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def resize_boxes(box, wh: Tuple[float, float], mode: str = 'xyxy'):
    """Resize boxes to a specific size."""
    box_cxywh = convert_bbox(box, mode, 'cxcywh')
    box_cxywh[:, 2] = wh[0]
    box_cxywh[:, 3] = wh[1]

    return convert_bbox(box_cxywh, 'cxcywh', mode)


def square_boxes(box, mode: str = 'xyxy'):
    if not box.dim() == 1:
        raise ValueError("Square boxes currently only supported for a single box.")

    box_cxywh = convert_bbox(box, mode, 'cxcywh')
    wh_max = box_cxywh[2:].max().item()
    box_aug_cxywh = resize_boxes(box_cxywh, (wh_max, wh_max), mode='cxcywh')

    return convert_bbox(box_aug_cxywh, mode_in='cxcywh', mode_out=mode)


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def _bbox_arbitrary_to_xyxy(box: torch.Tensor, mode: str) -> torch.Tensor:

    if mode == 'xyxy':
        return box

    box_out = box.clone()
    if mode == 'xywh':
        box_out[..., 2:] = box[..., :2] + box[..., 2:]
    elif mode == 'cxcywh':
        box_out[..., :2] -= box_out[..., 2:] / 2
        return _bbox_arbitrary_to_xyxy(box_out, 'xywh')
    else:
        raise NotImplementedError

    return box_out


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def _bbox_xyxy_to_arbitrary(box: torch.Tensor, mode: str) -> torch.Tensor:

    if mode == 'xyxy':
        return box

    box_out = box.clone()
    if mode == 'xywh':
        box_out[..., 2:] = box[:, 2:] - box[:, :2]
    elif mode == 'cxcywh':
        # nice little recursion
        box_xywh = _bbox_xyxy_to_arbitrary(box, mode='xywh')
        box_out[..., :2] = (box[..., :2] + box[..., 2:]) / 2
        box_out[..., 2:] = box_xywh[:, 2:]
    else:
        raise NotImplementedError

    return box_out


def _unified_img_size(img_size, order: str = 'matplotlib'):
    """Swaps image size depending on coordinate order."""

    img_size = torch.FloatTensor(list(img_size))
    if order == 'matplotlib':
        img_size = img_size[[-1, -2]]
    elif order is not None:
        raise ValueError("Order must be None or 'matplotlib.")

    return img_size
