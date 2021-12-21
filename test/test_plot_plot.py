import tempfile
import unittest.mock

import cv2
from matplotlib import cbook
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import pytorch_utils.plot.plot


@pytest.fixture()
def rgb_sample():
    with cbook.get_sample_data('grace_hopper.jpg') as image_file:
        return torch.from_numpy(plt.imread(image_file)).permute(-1, 0, 1)


@pytest.mark.parametrize("backend", ["matplotlib", "cv2"])
def test_tshow(backend):
    x = torch.rand(3, 30, 32)
    xn = x.permute(1, 2, 0).clone().numpy()

    if backend == "matplotlib":
        with unittest.mock.patch('matplotlib.pyplot.imshow') as mimshow:
            pytorch_utils.plot.plot.tshow(x, cmap='gray', backend=backend)
    elif backend == "cv2":
        with unittest.mock.patch("cv2.imshow") as mimshow:
            pytorch_utils.plot.plot.tshow(x, backend=backend)

    mimshow.assert_called_once()
    imshow_args = mimshow.call_args.args
    imshow_kwargs = mimshow.call_args.kwargs

    if backend == "matplotlib":
        assert imshow_args[0].size() == torch.Size([30, 32, 3])
        assert 'cmap' in imshow_kwargs.keys()
    if backend == "cv2":
        assert imshow_args[1].shape == (30, 32, 3)


@pytest.fixture()
def graph():
    return [
        (0, 1),
        (1, 3),
        (2, 3),
        (4, None),
        (5, None),
    ]


@pytest.mark.graphic
@pytest.mark.manual
@pytest.mark.parametrize("cv2_backend", [False, True])
def test_plot_keypoints(cv2_backend, rgb_sample, graph):
    x = torch.Tensor([[210, 180], [315, 170], [230, 250], [280, 200], [280, 310]])

    if not cv2_backend:
        pytorch_utils.plot.plot.plot_keypoints(x, graph)
    else:
        rgb_sample = pytorch_utils.plot.plot.plot_keypoints(x, graph, img=rgb_sample)
    pytorch_utils.plot.plot.tshow(rgb_sample)
    plt.show()


@pytest.mark.graphic
@pytest.mark.manual
def test_plot_keypoints_3d(graph):
    x = torch.rand(5, 3)

    f = plt.figure()
    ax = f.add_subplot(projection='3d')
    pytorch_utils.plot.plot.plot_keypoints(x, graph, ax=ax, plot_3d=True)
    plt.show()


@pytest.mark.graphic
@pytest.mark.manual
@pytest.mark.parametrize("cv2_backend", [False, True])
def test_plot_bbox(cv2_backend, rgb_sample):
    b = torch.Tensor([[50, 150, 400, 450], [180, 200, 220, 230]])

    if not cv2_backend:
        pytorch_utils.plot.plot.plot_bboxes(b)
    else:
        rgb_sample = pytorch_utils.plot.plot.plot_bboxes(b, img=rgb_sample)
    pytorch_utils.plot.plot.tshow(rgb_sample)
    plt.show()


@pytest.mark.parametrize("img", [torch.rand(3, 32, 64), torch.rand(32, 64)])
def test_torch_cv2_conversion(img):

    np.testing.assert_array_almost_equal(
        pytorch_utils.plot.plot.cv2_torch(
            pytorch_utils.plot.plot.torch_cv2(img)
        ).numpy(),
        img.numpy()
    )
