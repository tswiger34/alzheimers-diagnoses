"""Continuous sinusoidal temporal positional encoding."""

import math

import torch
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):
    """Add sinusoidal encodings for elapsed times, including fractional times."""

    def __init__(self, d_model: int, *, dropout: float, max_time_index: float) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if max_time_index < 0:
            raise ValueError("max_time_index cannot be negative")
        self.max_time_index = float(max_time_index)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "div_term",
            torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)),
        )

    def forward(self, x: Tensor, relative_times: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [batch, visits, features], got {tuple(x.shape)}")
        if relative_times.shape != x.shape[:2]:
            raise ValueError(
                f"relative_times must have shape {tuple(x.shape[:2])}, got {tuple(relative_times.shape)}"
            )
        if not torch.isfinite(relative_times).all():
            raise ValueError("relative_times must contain only finite values")
        if (relative_times < 0).any() or (relative_times > self.max_time_index).any():
            raise ValueError(f"relative_times must be between 0 and {self.max_time_index}")

        angles = relative_times.to(device=x.device, dtype=x.dtype).unsqueeze(-1) * self.get_buffer("div_term").to(
            dtype=x.dtype
        )
        encoding = torch.zeros_like(x)
        encoding[..., 0::2] = torch.sin(angles)
        odd_features = encoding[..., 1::2].shape[-1]
        encoding[..., 1::2] = torch.cos(angles[..., :odd_features])
        return self.dropout(x + encoding)
