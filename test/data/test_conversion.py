import numpy as np
import pytest
import torch

import pytorch_utils.data


@pytest.mark.parametrize("img", [torch.rand(3, 32, 64), torch.rand(32, 64)])
def test_torch_cv2_conversion(img):

    np.testing.assert_array_almost_equal(
        pytorch_utils.data.conversion.cv2_torch(
            pytorch_utils.data.conversion.torch_cv2(img)
        ).numpy(),
        img.numpy()
    )
