import torch


def convert_bbox(box: torch.Tensor, mode_in: str, mode_out: str) -> torch.Tensor:
    """
    Convert bounding boxes in format N x 4 between xyxy and xywh formats.

    Args:
        box: input boxes
        mode_in: xyxy or xywh
        mode_out: xyxy or xywh

    """
    if mode_in == mode_out:
        return box

    # always convert to xyxy first and derive from that base
    box_xyxy = _bbox_arbitrary_to_xyxy(box, mode=mode_in)
    box_out = _bbox_xyxy_to_arbitrary(box_xyxy, mode=mode_out)

    return box_out


def _bbox_arbitrary_to_xyxy(box: torch.Tensor, mode: str) -> torch.Tensor:

    if mode == 'xyxy':
        return box

    box_out = box.clone()
    if mode == 'xywh':
        box_out[:, 2:] = box[:, :2] + box[:, 2:]
    else:
        raise NotImplementedError

    return box_out

def _bbox_xyxy_to_arbitrary(box: torch.Tensor, mode: str) -> torch.Tensor:

    if mode == 'xyxy':
        return box

    box_out = box.clone()
    if mode == 'xywh':
        box_out[:, 2:] = box[:, 2:] - box[:, :2]
    elif mode == 'cxcywh':
        # nice little recursion
        box_xywh = _bbox_xyxy_to_arbitrary(box, mode='xywh')
        box_out[:, :2] = (box[:, :2] + box[:, 2:]) / 2
        box_out[:, 2:] = box_xywh[:, 2:]
    else:
        raise NotImplementedError

    return box_out
