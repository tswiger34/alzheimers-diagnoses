"""Shared image encoders for survival models."""

from typing import Self, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from ltsa.image_encoder import ImageEncoder
from torch import Tensor
from torchvision.models import ResNet, ResNet101_Weights, resnet101


class OrthogonalSliceResNet101Encoder(ImageEncoder[ResNet]):
    """Encode a 3D MRI as three orthogonal center slices with ResNet-101."""

    def __init__(
        self,
        weights: ResNet101_Weights | None = ResNet101_Weights.IMAGENET1K_V2,
    ) -> None:
        self.weights = weights
        super().__init__()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @override
    def _build_model(self) -> ResNet:
        return resnet101(weights=self.weights)

    @override
    def _extract_n_features(self, model: ResNet) -> int:
        return model.fc.in_features

    @override
    def _set_model_state(self) -> None:
        self._model.fc = nn.Identity()  # ty: ignore[invalid-assignment]

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if mode:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def volume_to_resnet_input(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [batch, channels, depth, height, width], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected one MRI channel, got {x.shape[1]}")

        volume = x[:, 0]
        orthogonal_slices = (
            volume[:, volume.shape[1] // 2, :, :],
            volume[:, :, volume.shape[2] // 2, :],
            volume[:, :, :, volume.shape[3] // 2],
        )
        resized_slices = [
            F.interpolate(
                image_slice.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(1)
            for image_slice in orthogonal_slices
        ]
        resnet_input = torch.stack(resized_slices, dim=1)
        image_min = resnet_input.amin(dim=(1, 2, 3), keepdim=True)
        image_max = resnet_input.amax(dim=(1, 2, 3), keepdim=True)
        resnet_input = (resnet_input - image_min) / (image_max - image_min).clamp_min(1e-6)
        return (resnet_input - self.get_buffer("imagenet_mean")) / self.get_buffer("imagenet_std")

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.volume_to_resnet_input(x))


class CoxRiskModel(nn.Module):
    """Single-MRI risk model using the shared comparison encoder."""

    def __init__(self, image_encoder: OrthogonalSliceResNet101Encoder) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.risk_head = nn.Linear(image_encoder.n_features, 1)

    def forward(self, images: Tensor) -> Tensor:
        return self.risk_head(self.image_encoder(images)).squeeze(-1)
