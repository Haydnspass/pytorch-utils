import torch


def convert_bbox(box: torch.Tensor, mode_in, mode_out) -> torch.Tensor:
    if mode_in == mode_out:
        return box

    box_out = box.clone()
    if mode_in == 'xyxy' and mode_out == 'xywh':
        box_out[:, 2:] = box[:, 2:] - box[:, :2]

    elif mode_in == 'xywh' and mode_out == 'xyxy':
        box_out[:, 2:] = box[:, :2] + box[:, 2:]

    else:
        raise NotImplementedError

    return box_out
