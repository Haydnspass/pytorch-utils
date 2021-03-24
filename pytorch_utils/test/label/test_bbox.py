from unittest import mock
import pytest
import torch

from pytorch_utils.label import bbox


@pytest.mark.parametrize("mode_in, mode_out", [
    ('xyxy', 'xyxy'),
    ('xyxy', 'xywh'),
    ('xywh', 'xyxy'),
    ('xywh', 'xywh'),
])
def test_convert_bbox(mode_in, mode_out):

    with mock.patch.object(bbox, '_bbox_arbitrary_to_xyxy') as arb_to_xyxy:
        with mock.patch.object(bbox, '_bbox_xyxy_to_arbitrary') as xyxy_to_arb:
            bbox.convert_bbox(torch.LongTensor([[1., 2., 3., 4.]]), mode_in=mode_in, mode_out=mode_out)

    if mode_in == mode_out:
        xyxy_to_arb.assert_not_called()
        arb_to_xyxy.assert_not_called()
    else:
        xyxy_to_arb.assert_called()
        arb_to_xyxy.assert_called()


@pytest.mark.parametrize("box,mode,box_expct", [
    (torch.Tensor([[1., 2., 3., 4.]]), 'xyxy', torch.Tensor([[1., 2., 3., 4.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xywh', torch.Tensor([[1., 2., 4., 6.]])),
])
def test__bbox_arbitrary_to_xyxy(box, mode, box_expct):

    box_out = bbox._bbox_arbitrary_to_xyxy(box, mode=mode)
    assert (box_out == box_expct).all()


@pytest.mark.parametrize("box,mode,box_expct", [
    (torch.Tensor([[1., 2., 3., 4.]]), 'xyxy', torch.Tensor([[1., 2., 3., 4.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xywh', torch.Tensor([[1., 2., 2., 2.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'cxcywh', torch.Tensor([[2., 3., 2., 2.]])),
])
def test__bbox_xyxy_to_arbitrary(box, mode, box_expct):

    box_out = bbox._bbox_xyxy_to_arbitrary(box, mode=mode)
    assert (box_out == box_expct).all()
