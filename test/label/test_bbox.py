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


@pytest.mark.parametrize("box,expct", [
    (torch.Tensor([10, 20, 30, 40]), None),
    (torch.Tensor([[10, 20, 10, 40]]), 'At least one bounding box has width or height 0.')
])
def test_check_box(box, expct):
    if expct is None:
        bbox.check_bbox(box)
        return

    with pytest.raises(ValueError) as err:
        bbox.check_bbox(box)

    assert str(err.value) == expct


@pytest.mark.parametrize("box,box_expct", [
    (torch.Tensor([[-5., -10, 3, 7]]), torch.Tensor([[0., 0., 3, 7]])),
    (torch.Tensor([[0., 0., 31.99, 39.99]]), torch.Tensor([[0., 0., 31.99, 39.99]])),
    (torch.Tensor([-5, -1, 100, 200]), torch.Tensor([0., 0., 32 - 1e-6, 40 - 1e-6])),
    (torch.Tensor([[-5, 2, -1, 7]]), 'err'),
    (torch.Tensor([[100, 200, 300, 400]]), 'err'),
])
def test_limit_bbox(box, box_expct):
    if box_expct == 'err':
        with pytest.raises(ValueError):
            bbox.limit_bbox(box, torch.Size([32, 40]))
        return

    box_out = bbox.limit_bbox(box, torch.Size([32, 40]))
    assert (box_out == box_expct).all()


@pytest.mark.skip("Deprecated implementation.")
@pytest.mark.parametrize("box,box_expct", [
    (torch.Tensor([[-5., -10, 3, 7]]), torch.Tensor([[0., 0., 8, 17]])),
    (torch.Tensor([[10., 20., 37, 41]]), torch.Tensor([[4., 18., 31., 39.]])),
    (torch.tensor([[5., 5., 8., 10.]]), torch.Tensor([[5., 5., 8., 10.]])),
    (torch.Tensor([[1., 1., 32., 40.]]), torch.Tensor([[0., 0., 31., 39.]])),
    (torch.Tensor([[0., 0., 31.99, 39.99]]), torch.Tensor([[0., 0., 31.99, 39.99]])),
    (torch.Tensor([[0., 0., 40., 48]]), 'err'),
])
def test_shift_bbox_inside_img(box, box_expct):
    if box_expct == 'err':
        with pytest.raises(ValueError):
            bbox.shift_bbox_inside_img(box, torch.Size([32, 40]), 'xyxy')

    else:
        box_out = bbox.shift_bbox_inside_img(box, torch.Size([32, 40]), 'xyxy')
        assert (box_out == box_expct).all()


def test_resize_bbox():
    box = torch.Tensor([[-20, -30, 20, 30]])

    box_out = bbox.resize_boxes(box, (50, 20), 'xyxy')
    assert (box_out == torch.Tensor([-25, -10, 25, 10])).all()


def test_square_box():
    box = torch.Tensor([100, 200, 300, 500])

    box_out = bbox.square_boxes(box, 'xyxy')
    assert (box_out == torch.Tensor([50, 200, 350, 500])).all()

    # raises
    with pytest.raises(ValueError):
        bbox.square_boxes(box.unsqueeze(0), 'xyxy')


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
