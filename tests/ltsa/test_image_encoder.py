import torch

from ltsa.image_encoder import ImageEncoder, ResNetEncoder, SwinEncoder


def test_encoders_can_initialize_without_downloading_weights() -> None:
    resnet_encoder = ResNetEncoder(weights=None)
    swin_encoder = SwinEncoder(weights=None)

    assert isinstance(resnet_encoder, ImageEncoder)
    assert isinstance(swin_encoder, ImageEncoder)
    assert resnet_encoder(torch.randn(1, 3, 64, 64)).shape == (1, resnet_encoder.n_features)
    assert swin_encoder(torch.randn(1, 3, 256, 256)).shape == (1, swin_encoder.n_features)
