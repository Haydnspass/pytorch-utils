import pytest
import torch

from pytorch_utils.label import bbox


@pytest.mark.parametrize("box_in,mode_in,mode_out,box_expct", [
    (torch.Tensor([[1., 2., 3., 4.]]), 'xyxy', 'xyxy', torch.Tensor([[1., 2., 3., 4.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xywh', 'xywh', torch.Tensor([[1., 2., 3., 4.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xyxy', 'xywh', torch.Tensor([[1., 2., 2., 2.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xywh', 'xyxy', torch.Tensor([[1., 2., 4, 6.]])),
])
def test_convert_bbox(box_in, mode_in, mode_out, box_expct):

    box_out = bbox.convert_bbox(box_in, mode_in, mode_out)

    assert (box_out == box_expct).all()
