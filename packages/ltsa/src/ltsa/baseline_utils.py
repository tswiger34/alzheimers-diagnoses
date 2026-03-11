"""Contains useful models for comparing the LTSA architecture to a single image baseline"""

from typing import Any

import torch
from torch import Tensor

from ltsa.image_encoder import ImageEncoder


class BaselineImageSurvivalModel(torch.nn.Module):
    """Baseline model to use when comparing
    Args:
        image_encoder (ImageEncoder): The image encoder to use for feature extraction
        n_classes (int): The number of classification options
        dropout (float, optional): Optionally define the rate rate used in the forward pass. Defaults to 0.25
    """

    def __init__(self, image_encoder: ImageEncoder, n_classes: int, dropout: float = 0.25):
        super().__init__()

        self.encoder: ImageEncoder[Any] = image_encoder

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.encoder.n_features, out_features=n_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x) -> tuple[Tensor, Tensor]:
        x: Tensor = self.encoder(x)

        hazards: Tensor = self.classifier(x)
        surv: Tensor = torch.cumprod(input=1 - hazards, dim=1)

        return hazards, surv
