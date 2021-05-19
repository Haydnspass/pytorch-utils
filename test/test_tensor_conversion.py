import PIL
import torch
import torchvision

from pytorch_utils.tensor import conversion

def test_pil_to_tensor():
    im = PIL.Image.new(mode='P', size=(32, 32))

    t = conversion.pil_to_tensor(im, torch.long)
    assert t.dtype == torch.long
    assert t.min() >= 0
    assert t.max() < 255
    