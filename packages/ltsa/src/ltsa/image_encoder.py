from abc import ABC, abstractmethod
from typing import override

from torch import Tensor, nn
from torchvision.models import ResNet, ResNet18_Weights, Swin_V2_T_Weights, SwinTransformer, resnet18, swin_v2_t


class ImageEncoder[T: nn.Module](nn.Module, ABC):
    """Abstract interface for torchvision image encoders with classifier head removed."""

    def __init__(self) -> None:
        super().__init__()
        self._model: T = self._build_model()
        self._n_features: int = self._extract_n_features(model=self._model)
        self._set_model_state()

    @property
    def model(self) -> T:
        return self._model

    @property
    def n_features(self) -> int:
        return self._n_features

    @abstractmethod
    def _build_model(self) -> T:
        """Builds the pretrained encoder architecture."""
        ...

    @abstractmethod
    def _extract_n_features(self, model: T) -> int:
        """Returns classifier input feature size before replacing the head."""
        ...

    @abstractmethod
    def _set_model_state(self) -> None:
        """Replaces the classifier head with ``nn.Identity`` in-place."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResNetEncoder(ImageEncoder[ResNet]):
    """Image encoder model based on the ResNet architecture"

    Attributes:
        weights (ResNet18_Weights): Pre-trained weights for initialization, uses the `IMAGENET1K_V1` weights
    """

    def __init__(self, weights: ResNet18_Weights | None = ResNet18_Weights.IMAGENET1K_V1) -> None:
        self.weights = weights
        super().__init__()

    @override
    def _build_model(self) -> ResNet:
        return resnet18(weights=self.weights)

    @override
    def _extract_n_features(self, model: ResNet) -> int:
        return model.fc.in_features

    @override
    def _set_model_state(self) -> None:
        self._model.fc = nn.Identity()  # ty: ignore[invalid-assignment]


class SwinEncoder(ImageEncoder[SwinTransformer]):
    """Image encoder model based on the SwinTransformer architecture

    Attributes:
        weights (Swin_V2_T_Weights): Pre-trained weights for initialization, uses the `IMAGENET1K_V1` weights
    """

    def __init__(self, weights: Swin_V2_T_Weights | None = Swin_V2_T_Weights.IMAGENET1K_V1) -> None:
        self.weights = weights
        super().__init__()

    @override
    def _build_model(self) -> SwinTransformer:
        return swin_v2_t(weights=self.weights)

    @override
    def _extract_n_features(self, model: SwinTransformer) -> int:
        return model.head.in_features

    @override
    def _set_model_state(self) -> None:
        self._model.head = nn.Identity()  # ty: ignore[invalid-assignment]
