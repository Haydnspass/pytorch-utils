from unittest import mock

import numpy as np
import numpy.testing
import pytest
import torch

from pytorch_utils.label import bbox


def test_bbox_clone():
    b = bbox.BBox([1., 2., 3., 4])
    b_ref = b
    b_clone = b.clone()

    b_ref.xyxy = torch.Tensor([2., 3., 4., 5.])

    assert (b.xyxy == torch.Tensor([2., 3., 4., 5.])).all()
    assert (b_clone.xyxy == torch.Tensor([1., 2., 3., 4])).all()


def test_bbox_eq():
    b0 = bbox.BBox([[1., 2., 3., 4.]], 'xyxy')
    b1 = bbox.BBox([[2., 3., 2., 2.]], 'cxcywh')

    assert isinstance(b0 == b1, bool)
    assert b0 == b1


@pytest.mark.parametrize("box,expct", [
    (torch.Tensor([10, 20, 30, 40]), None),
    (torch.Tensor([[10, 20, 10, 40]]), 'Bounding box(es) are not of valid area.')
])
def test_check_bbox_area(box, expct):
    box = bbox.BBox(box, 'xyxy')
    if expct is None:
        box.check_area()
        return

    with pytest.raises(ValueError) as err:
        box.check_area()

    assert str(err.value) == expct


@pytest.mark.parametrize("box,expct", [
    ([10, 20, 20, 40], None),
    ([10, 30, 20, 80], 'Bounding box(es) are outside of the specified image size.')
])
def test_check_bbox_fits_img(box, expct):
    box = bbox.BBox(box, 'xyxy')
    if expct is None:
        box.check_fits_img([64, 21])
        return

    with pytest.raises(ValueError) as err:
        box.check_fits_img([64, 21])

    assert str(err.value) == expct


@pytest.mark.parametrize("mode_in, mode_out", [
    ('xyxy', 'xyxy'),
    ('xyxy', 'xywh'),
    ('xywh', 'xyxy'),
    ('xywh', 'xywh'),
])
def test_convert_bbox(mode_in, mode_out):
    with mock.patch.object(bbox, '_bbox_arbitrary_to_xyxy') as arb_to_xyxy:
        with mock.patch.object(bbox, '_bbox_xyxy_to_arbitrary') as xyxy_to_arb:
            bbox.convert_bbox(torch.LongTensor([[1., 2., 3., 4.]]), mode_in=mode_in,
                              mode_out=mode_out)

    if mode_in == mode_out:
        xyxy_to_arb.assert_not_called()
        arb_to_xyxy.assert_not_called()
    else:
        xyxy_to_arb.assert_called()
        arb_to_xyxy.assert_called()


@pytest.mark.parametrize("box,box_expct", [
    ([[-5., -10, 3, 7]], [[0., 0., 3, 7]]),
    ([[0., 0., 31.99, 39.99]], [[0., 0., 31.99, 39.99]]),
    ([-5, -1, 100, 200], [0., 0., 32 - 1e-6, 40 - 1e-6]),
    ([[-5, 2, -1, 7]], 'err'),
    ([[100, 200, 300, 400]], 'err'),
])
def test_limit_bbox(box, box_expct):
    if box_expct == 'err':
        with pytest.raises(ValueError):
            bbox.BBox(box).limit_bbox_(torch.Size([32, 40]), order=None)
        return

    box_out = bbox.BBox(box).limit_bbox_(torch.Size([32, 40]), order=None)
    box_expct = bbox.BBox(box_expct)
    assert box_out == box_expct


_scenarios_1d = [
    ([2, 10], [2, 10]),  # totally inside (no-op)
    ([-3, 10], [0, 13]),  # lower end outside
    ([-3, 31], 'err'),
    ([30, 45], [17, 32 - 1e-6]),  # upper end outside
    ([-3, 45], 'err'),  # both outside (=err)
]


@pytest.mark.parametrize("x,x_expct", _scenarios_1d)
@pytest.mark.parametrize("y,y_expct", _scenarios_1d)
def test_shift_bbox_inside_img(x, x_expct, y, y_expct):
    box = [x[0], y[0], x[1], y[1]]

    if x_expct == 'err' or y_expct == 'err':
        with pytest.raises(ValueError):
            bbox.BBox(box).shift_bbox_inside_img(torch.Size([32, 32]))

    else:
        box_out = bbox.BBox(box).shift_bbox_inside_img(torch.Size([32, 32]))

        box_expct = bbox.BBox([x_expct[0], y_expct[0], x_expct[1], y_expct[1]])
        numpy.testing.assert_almost_equal(
            box_out.xyxy.numpy(),
            box_expct.xyxy.numpy(),
            decimal=5,
        )


@pytest.mark.parametrize("order", [None, 'matplotlib'])
@pytest.mark.parametrize("box,mode,img_expct,shift_expct", [
    ([1., 2., 3., 5.], 'floor', [2, 3], [1, 2]),
    ([1.9, 2.1, 3.1, 5.7], 'floor', [2, 3], [1, 2]),
    ([1.89, 2.1, 3.1, 5.7], 'ceil', [2, 3], [2, 3]),
    ([-3.2, 1., 10., 20.], 'floor', [10, 19], [0, 1]),
    ([500, 700, 880, 290], 'floor', [0, 0], [float('nan'), float('nan')])
])
def test_crop_image_unfilled(order, box, mode, img_expct, shift_expct):
    box = bbox.BBox(box)
    img = torch.rand(3, 64, 69)
    shift_expct = torch.tensor(shift_expct)

    img_out, shift_out = box.crop_image(img, mode=mode, order=order)

    if order == 'matplotlib':
        img_expct.reverse()

    assert img_out[0].size() == torch.Size(img_expct)
    assert np.array_equal(shift_out.numpy(), shift_expct.numpy(), equal_nan=True)


@pytest.mark.parametrize("order", [None, 'matplotlib'])
@pytest.mark.parametrize("box,mode,img_size,shift", [
    ([1., 2., 3., 5], 'floor', [2, 3], [1, 2]),
    ([-3.2, 2., 5., 7.], 'floor', [9, 5], [-4, 2]),
    ([5.3, 3., 17., 20.], 'ceil', [11, 17], [6, 3])
])
def test_crop_image_filled(order, box, mode, img_size, shift):
    def _hard_coded_target_fills(box, mode, img_size):
        """Construct target zero-fills from test parametrization."""
        nz = torch.zeros(img_size, dtype=torch.bool)

        if box == [1., 2., 3., 5.] and mode == 'floor':
            pass
        elif box == [-3.2, 2., 5., 7.] and mode == 'floor':
            nz[:4] = 1
        elif box == [5.3, 3., 17., 20.] and mode == 'ceil':
            if order is None:
                nz[4:] = 1
                nz[:, 9:] = 1
            elif order == 'matplotlib':
                nz[6:] = 1
                nz[:, 7:] = 1
            else:
                raise ValueError
        else:
            raise ValueError
        return nz if order is None else nz.transpose(0, 1)

    tar_fill = _hard_coded_target_fills(box, mode, img_size)
    box = bbox.BBox(box)
    img = torch.rand(3, 10, 12).clamp(min=0.1)
    shift = torch.tensor(shift)

    img_out, shift_out = box.crop_image(img, mode=mode, order=order, fill=0.)

    if order == 'matplotlib':
        img_size.reverse()

    assert img_out[0].size() == torch.Size(img_size)
    assert (shift_out == shift).all()

    # check that the correct side is filled with 0.
    is_filled = img_out == 0
    assert (is_filled == tar_fill).all()


@pytest.mark.parametrize("box,box_expct", [
    ([1., 2., 3., 4.], [1., 2., 3., 4.]),
    ([3., 4., 1., 2.], [1., 2., 3., 4.])
])
def test_repair_order(box, box_expct):
    b = bbox.BBox(box)
    b.repair_order()

    assert (b.xyxy == torch.tensor(box_expct)).all()


def test_resize_bbox():
    box = torch.Tensor([[-20, -30, 20, 30]])

    box_out = bbox.resize_boxes(box, (50, 20), 'xyxy')
    assert (box_out == torch.Tensor([-25, -10, 25, 10])).all()


@pytest.mark.parametrize("mode,box_in,box_expct", [
    ("max", [1., 2., 3., 4.], [1., 2., 4., 4.]),
    ("min", [1., 2., 3., 4.], [1., 2., 3., 3.]),
    ("max", [[1., 2., 3., 4.], [-500., -1200., 899., 283]],
     [[1., 2., 4., 4.], [-500., -1200., 899., 899.]])
])
def test_square_boxes(mode, box_in, box_expct):
    box = bbox.BBox(box_in, mode='cxcywh')
    box_expct = bbox.BBox(box_expct, mode='cxcywh')

    assert box.square_bbox_(mode=mode) == box_expct


def test_random_close_bbox():
    box = bbox.BBox([0., 10., 20., 30.], mode='cxcywh')
    box_rand = box.random_close(rel_dist=2.)

    assert (box_rand.xyxy != box.xyxy).all()
    assert (box_rand.cxcywh[2:] == torch.Tensor([20., 30.])).all()
    assert (torch.Tensor([-20., -20.]) <= box_rand.cxcywh[:2]).all()
    assert (box_rand.cxcywh[:2] <= torch.Tensor([20., 40.])).all()


def test_random_zoom_bbox():
    box = bbox.BBox([1., 2., 3., 4], mode='cxcywh')
    zoom_range = (0.5, 3)
    box_rand = box.random_zoom(*zoom_range)

    assert np.array_equal(
        box_rand.cxcywh[..., :2],
        box.cxcywh[..., :2]
    ), "Centers must not change"
    assert pytest.approx(
        box_rand.cxcywh[..., 2] / box.cxcywh[..., 2],
        box_rand.cxcywh[..., 3] / box.cxcywh[..., 3]
    ), "Zoom must be equivalent in w and h"
    assert zoom_range[0] <= (box_rand.cxcywh[..., 2] / box.cxcywh[..., 2]) <= zoom_range[1], \
        "Zoom factor outside range"


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
