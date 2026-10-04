"""Shared neural continuous hazard, conditional likelihood and MRI model variants.

All times are continuous months on the baseline time axis. Integration uses
Gauss-Legendre quadrature, unrelated to the visit-bin schedule.
"""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContinuousHazardHead(nn.Module):
    """Positive monthly hazard lambda(u | z)"""

    def __init__(self, features: int, quadrature_points: int = 32):
        super().__init__()
        if quadrature_points < 8:
            raise ValueError("Use at least 8 integration points")
        self.network = nn.Sequential(nn.Linear(features + 1, 64), nn.Tanh(), nn.Linear(64, 1))
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        self.register_buffer("nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def hazard(self, z: Tensor, time: Tensor) -> Tensor:
        if time.ndim == 1:
            time = time[:, None]
        expanded = z[:, None, :].expand(-1, time.shape[1], -1)
        return (F.softplus(self.network(torch.cat((expanded, time[..., None] / 12), -1))).squeeze(-1) + 1e-7) / 12

    def cumulative_hazard(self, z: Tensor, start: Tensor, end: Tensor) -> Tensor:
        if (
            not torch.isfinite(start).all()
            or not torch.isfinite(end).all()
            or (start < 0).any()
            or (end < start).any()
        ):
            raise ValueError("Integration requires finite 0 <= start <= end")
        width = (end - start) / 2
        times = start[:, None] + width[:, None] * (self.nodes[None, :] + 1)
        return width * (self.hazard(z, times) * self.weights).sum(-1)

    def nll(self, z: Tensor, entry: Tensor, outcome: Tensor, event: Tensor) -> Tensor:
        if (outcome <= entry).any() or not ((event == 0) | (event == 1)).all():
            raise ValueError("Likelihood requires positive future follow-up and binary events")
        return self.cumulative_hazard(z, entry, outcome) - event * self.hazard(z, outcome).squeeze(-1).log()

    def survival(self, z: Tensor, entry: Tensor, end: Tensor) -> Tensor:
        return torch.exp(-self.cumulative_hazard(z, entry, end))


class ContinuousMRIModel(nn.Module):
    """A: image only; B: image plus acquisition time; history: matched transformer.

    The hazard query u is the survival time variable in every variant. Baseline
    A receives neither s, nominal visit nor history in its image representation.
    B and history share the same acquisition-time embedding. Padding tokens are
    masked and never encoded as images. Histories contain only the available prefix.
    """

    def __init__(self, encoder, variant="a", features=128, quadrature_points=32, freeze_encoder=False):
        super().__init__()
        if variant not in ("a", "b", "history"):
            raise ValueError("variant must be a, b or history")
        self.variant = variant
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.projection = nn.Linear(encoder.n_features, features)
        # Construct common components in identical order for matched initial weights.
        self.time_embedding = nn.Linear(1, features)
        self.head = ContinuousHazardHead(features, quadrature_points)
        self.transformer = None
        if variant == "history":
            layer = nn.TransformerEncoderLayer(features, 4, features * 2, dropout=0, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def train(self, mode=True):
        """Keep frozen encoder parameters and running statistics fixed."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, images: Tensor, times: Tensor, observed: Tensor) -> Tensor:
        if not observed.any(dim=1).all():
            raise ValueError("Each patient requires at least one observed MRI")
        last = torch.arange(observed.shape[1], device=observed.device)[None, :].expand_as(observed)
        last = last.masked_fill(~observed, -1).max(dim=1).values
        rows = torch.arange(images.shape[0], device=images.device)
        if self.variant != "history":
            z = self.projection(self.encoder(images[rows, last]))
            if self.variant == "b":
                z = z + self.time_embedding(times[rows, last, None] / 12)
            return z
        z = self.projection(self.encoder(images[observed]))
        tokens = z.new_zeros((*observed.shape, z.shape[-1]))
        tokens[observed] = z + self.time_embedding(times[observed][:, None] / 12)
        encoded = self.transformer(tokens, src_key_padding_mask=~observed)
        return encoded[rows, last]

    def forward(self, batch):
        z = self.encode(batch["images"], batch["times"], batch["observed"])
        return self.head.nll(z, batch["entry"], batch["outcome"], batch["event"])
