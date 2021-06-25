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


@pytest.mark.parametrize("box,box_expct", [
    (torch.Tensor([[-5., -10, 3, 7]]), torch.Tensor([[0., 0., 8, 17]])),
    (torch.Tensor([[10., 20., 37, 41]]), torch.Tensor([[4., 18., 31., 39.]])),
    (torch.Tensor([[0., 0., 40., 48]]), 'err'),
])
def test_limit_bbox_to_img(box, box_expct):
    if box_expct == 'err':
        with pytest.raises(ValueError):
            bbox.limit_bbox_to_img(box, torch.Size([32, 40]), 'xyxy')

    else:
        box_out = bbox.limit_bbox_to_img(box, torch.Size([32, 40]), 'xyxy')
        assert (box_out == box_expct).all()


def test_resize_bbox():
    box = torch.Tensor([[-20, -30, 20, 30]])

    box_out = bbox.resize_boxes(box, (50, 20), 'xyxy')
    assert (box_out == torch.Tensor([-25, -10, 25, 10])).all()


@pytest.mark.parametrize("box,mode,box_expct", [
    (torch.Tensor([[1., 2., 3., 4.]]), 'xyxy', torch.Tensor([[1., 2., 3., 4.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'xywh', torch.Tensor([[1., 2., 4., 6.]])),
    (torch.Tensor([[1., 2., 3., 4.]]), 'cxcywh', torch.Tensor([[-.5, 0, 2.5, 4.]])),
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


def test_auto_inflate():

    box = torch.Tensor([1., 2., 3., 4.])  # 1D box

    bbox._bbox_arbitrary_to_xyxy(box, mode='xywh')
    bbox._bbox_xyxy_to_arbitrary(box, mode='xywh')
    bbox.resize_boxes(box, (10, 10))
