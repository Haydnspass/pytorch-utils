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
        return self._data  # convert_bbox(self._data, mode_in='xyxy', mode_out='xyxy')

    @xyxy.setter
    def xyxy(self, val):
        self._data = val

    @property
    def xywh(self):
        return convert_bbox(self._data, mode_in='xyxy', mode_out='xywh')

    @xywh.setter
    def xywh(self, val):
        self.xyxy = convert_bbox(val, mode_in='xywh', mode_out='xyxy')

    @property
    def cxcywh(self):
        return convert_bbox(self._data, mode_in='xyxy', mode_out='cxcywh')

    @cxcywh.setter
    def cxcywh(self, val):
        self.xyxy = convert_bbox(val, mode_in='cxcywh', mode_out='xyxy')

    @property
    def area(self):
        return self.xywh[..., 2] * self.xywh[..., 3]

    def __eq__(self, other) -> bool:
        return (self.xyxy == other.xyxy).all().item()

    def __len__(self) -> int:
        if self.xyxy.dim() == 1:
            return 1
        if self.xyxy.dim() == 2:
            return self.xyxy.size(0)

        raise ValueError("Not supported dim of underlying data.")

    def clone(self):
        return BBox(self.xyxy, mode='xyxy')

    def limit_bbox(self, img_size: torch.Size, eps_border=1e-6, order='matplotlib',
                    check: bool = True):
        return self.clone().limit_bbox(img_size, eps_border, order, check)

    def limit_bbox_(self, img_size: torch.Size, eps_border=1e-6, order='matplotlib',
                    check: bool = True):
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
        img_size = _unified_img_size(img_size, order)

        self.xyxy.clamp_(0.)
        self.xyxy[..., [0, 2]] = self.xyxy[..., [0, 2]].clamp(max=img_size[-2] - eps_border)
        self.xyxy[..., [1, 3]] = self.xyxy[..., [1, 3]].clamp(max=img_size[-1] - eps_border)

        self.check_area()

        return self

    def square_bbox(self, mode: str = 'max'):
        if mode not in ('min', 'max'):
            raise ValueError("Mode must be max or min.")

        self.cxcywh = self._square_bbox(self.cxcywh, mode=mode)
        return self

    def shift_bbox_inside_img(self, img_size: torch.Size, eps_border=1e-6, order='matplotlib'):
        self.xyxy = shift_bbox_inside_img(
            self.xyxy, img_size=img_size, mode='xyxy', eps_border=eps_border, order=order
        )

        return self

    def crop_image(self, img: torch.Tensor, mode: str = 'floor', order: str = 'matplotlib'):
        assert mode in ('ceil', 'floor')
        assert order in (None, 'matplotlib')
        assert len(self) == 1
        assert img.dim() in (2, 3)

        xyxy = self.xyxy.floor() if mode == 'floor' else self.xyxy.ceil()

        img_bbox = BBox.__init__(xyxy, mode='xyxy')
        img_bbox.limit_bbox(img_size=img.size(), )
        xyxy[:2] = xyxy[:2].clamp(min=0)

        if order == 'matplotlib':
            xyxy = xyxy[[1, 0, 3, 2]]

        xyxy[2] = xyxy[2].clamp(max=img.size(-2))
        xyxy[3] = xyxy[3].clamp(max=img.size(-1))

        xyxy = xyxy.long()

        img = img[
            ...,
            slice(xyxy[0], xyxy[2]),
            slice(xyxy[1], xyxy[3])
        ]
        return img, shift

    def repair_order(self):
        xyxy = self.xyxy
        xyxy[..., [0, 2]] = xyxy[..., [0, 2]].sort(dim=-1)[0]
        xyxy[..., [1, 3]] = xyxy[..., [1, 3]].sort(dim=-1)[0]

        self.xyxy = xyxy
        return self

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

    @staticmethod
    @pytorch_utils.lazy.tensor.cycle_view(2, 0)
    def _square_bbox(cxcywh: torch.Tensor, mode) -> torch.Tensor:
        if cxcywh.dim() != 2:
            raise ValueError

        if mode == 'min':
            cxcywh[:, 2:] = cxcywh[..., 2:].min(1, keepdim=True)[0].repeat(1, 2)
        if mode == 'max':
            cxcywh[:, 2:] = cxcywh[..., 2:].max(1, keepdim=True)[0].repeat(1, 2)

        return cxcywh


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


@pytorch_utils.lazy.tensor.cycle_view(2, 0)
def shift_bbox_inside_img(box, img_size: torch.Size,
                          mode='xyxy',
                          eps_border=1e-6,
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
        raise ValueError(
            "Bounding boxes can not be shifted such that they fit inside the specified"
            "image size.")

    box_xyxy += shift_min.repeat(1, 2)
    box_xyxy += shift_max.repeat(1, 2)

    BBox(box_xyxy, mode='xyxy').check_fits_img(
        img_size=img_size,
        order='matplotlib',  # cause this was ensured earlier
        eps_border=eps_border,
    )
    box_out = convert_bbox(box_xyxy, 'xyxy', mode)

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


def _unified_img_size(img_size, order: str = 'matplotlib'):
    """Swaps image size depending on coordinate order."""

    img_size = torch.FloatTensor(list(img_size))
    if order == 'matplotlib':
        img_size = img_size[[-1, -2]]
    elif order is not None:
        raise ValueError("Order must be None or 'matplotlib.")

    return img_size
