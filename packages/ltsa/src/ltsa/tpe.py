"""Continuous sinusoidal temporal positional encoding.

Credit to: https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis/blob/main/src/models.py
"""

import math

import torch
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):
    """Add sinusoidal encodings for elapsed times, including fractional times.

    Transformers typically use positional encoding to inform the model on the order of
    elements in an input sequence. Sequences of visits in longitudinal data are not evenly spaced, and with time
    gaps between visits varying widely in most cases. For this reason, traditional PE is not sufficient for this
    application. *Temporal Positional Encoding* (TPE) is a continuous version of PE that allows for fractional
    time gaps between visits. Rather than using the index of a visit in a sequence, TPE uses the absolute time
    of a visit relative to the final visit in the sequence.


    Attributes:
        d_model: The number of features in the input tensor.
        dropout: The dropout probability to apply after adding the positional encodings.
        max_time_index: The maximum time index to consider for the positional encodings.
    """

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
        """Computes the TPE for the image

        Args:
            x (Tensor):
                The features tensor of shape ``[batch, max_seq_len, d_model]``
            relative_times (Tensor):
                Tensor of absolute times relative to the final image in the sequence. The tensor
                should be of shape ``[batch, max_seq_len]``

        Raises:
            ValueError: If x does not have three dimensions.
            ValueError: If relative_times does not have the same shape as the first two dimensions of x.
            ValueError: If relative_times contains non-finite values.
            ValueError: If relative_times contains values outside the range [0, max_time_index]. Relative times are
              expected to be the *absolute* time relative to the final image in the sequence.

        Returns:
            Tensor: The input tensor x with the temporal positional encodings added.
        """
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
