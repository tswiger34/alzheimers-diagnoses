import pytest  # noqa


def test_smoke_test():
    """Smoke test that everything can get imported and classes get initialized"""
    from ltsa.image_encoder import ImageEncoder, ResNetEncoder, SwinEncoder

    resnet_encoder = ResNetEncoder()
    swin_encoder = SwinEncoder()
    img_encoder: type[ImageEncoder] = ImageEncoder

    assert isinstance(resnet_encoder, ImageEncoder)
    assert isinstance(swin_encoder, ImageEncoder)
    assert img_encoder is not None
