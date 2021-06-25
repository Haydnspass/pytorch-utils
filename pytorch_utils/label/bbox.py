from typing import Tuple

import torch

import pytorch_utils.lazy.tensor


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


def limit_bbox_to_img(box, img_size: torch.Size, mode='xyxy'):
    box_xyxy = convert_bbox(box, mode, 'xyxy')

    shift_min = box[:, :2] - torch.max(box_xyxy[:, :2], torch.zeros_like(box_xyxy[:, :2]))
    shift_max = box[:, 2:] - torch.min(box_xyxy[:, 2:], torch.tensor(list(img_size)) - 1)

    box_xyxy -= shift_min.repeat(1, 2)
    box_xyxy -= shift_max.repeat(1, 2)

    box_out = convert_bbox(box_xyxy, 'xyxy', mode)

    if (box_out.min() < 0).any() or (box_out[:, 2:] > torch.tensor(list(img_size)) - 1).any():
        raise ValueError

    return box_out


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def resize_boxes(box, wh: Tuple[float, float], mode: str = 'xyxy'):
    """Resize boxes to a specific size."""
    box_cxywh = convert_bbox(box, mode, 'cxcywh')
    box_cxywh[:, 2] = wh[0]
    box_cxywh[:, 3] = wh[1]

    return convert_bbox(box_cxywh, 'cxcywh', mode)


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
