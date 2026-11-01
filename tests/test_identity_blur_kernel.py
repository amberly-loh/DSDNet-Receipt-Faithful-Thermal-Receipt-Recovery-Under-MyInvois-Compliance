import torch
from dsdnet_receipt.dsdnet_adapter import gaussian_kernel


def test_zero_blur_is_identity_with_nonzero_taper_border():
    kernel = gaussian_kernel(0)
    assert kernel.shape == (1, 3, 3)
    image = torch.rand(1, 1, 16, 16)
    result = torch.nn.functional.conv2d(image, kernel.unsqueeze(0), padding=1)
    assert torch.equal(image, result)
