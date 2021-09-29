import unittest.mock
import matplotlib.pyplot as plt
import pytest

import torch

import pytorch_utils.plot.plot


def test_tshow():
    x = torch.rand(3, 30, 32)
    xn = x.permute(1, 2, 0).clone().numpy()

    with unittest.mock.patch('matplotlib.pyplot.imshow') as mimshow:
        pytorch_utils.plot.plot.tshow(x, cmap='gray')

    mimshow.assert_called_once()
    imshow_arg, = mimshow.call_args.args
    imshow_kwargs = mimshow.call_args.kwargs

    assert imshow_arg.size() == torch.Size([30, 32, 3])
    assert 'cmap' in imshow_kwargs.keys()


@pytest.mark.graphic
@pytest.mark.manual
@pytest.mark.parametrize("plot_3d", [True, False])
def test_plot_keypoints(plot_3d):
    x = torch.rand(5, 3)

    graph = [
        (0, 1),
        (1, 3),
        (2, 3),
        (4, None),
        (5, None),
    ]

    if plot_3d:
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
    else:
        ax = None

    pytorch_utils.plot.plot.plot_keypoints(x, graph, ax=ax, plot_3d=plot_3d)
    plt.show()


@pytest.mark.graphic
@pytest.mark.manual
def test_plot_bbox():
    # 2D
    pytorch_utils.plot.plot.plot_bboxes(torch.rand(2, 4))
    plt.show()

     # 1D
    pytorch_utils.plot.plot.plot_bboxes(torch.rand(4))
    plt.show()
