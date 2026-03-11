"""The temporal positional encoding layer

Credit to: https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis/blob/main/src/models.py
"""

import numpy as np
import torch
from torch import Tensor, long, nn


class TemporalPositionalEncoding(nn.Module):
    """Temporal positional encoding block derived from Holste G et. al.

    Builds positional indice --> PE denominator term --> applies `sin` PE to even positions, `cos` to odd -->
    add PE to model state via a model buffer --> Reorders to batch-first inputs for forward pass

    - If position *i* is even:
        `TE(v) = sin(v/(10000^(2i/d)))`
    - If position *i* is odd:
        `TE(v) = cos(v/(10000^(2i/d)))`

    Attributes:
        - dropout (float): The dropout rate to use
        - pe (Tensor): The positional encoding tensor

    Args:
        d_model (int): Embedding width
        dropout (float, optional): Dropout probability after adding position info. Defaults to 0.25.
        max_len (int, optional): Maximum sequence length to precompute encodings for. Defaults to 5000.
    """

    def __init__(self, d_model: int, dropout: float = 0.25, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position: Tensor = torch.arange(end=max_len).unsqueeze(dim=1)
        div_term: Tensor = torch.exp(
            input=torch.arange(start=0, end=d_model, step=2) * (-np.log(10000.0) / d_model)
        )
        pe: Tensor = torch.zeros(max_len, 1, d_model)
        pe_input: Tensor = position * div_term
        pe[:, 0, ::2] = torch.sin(input=pe_input)
        pe[:, 0, 1::2] = torch.cos(input=pe_input)
        self.register_buffer(name="pe", tensor=pe)
        self.pe: Tensor = torch.permute(input=self.pe, dims=(1, 0, 2))  # shape: batch x seq x feat

    def forward(self, x: Tensor, rel_times: Tensor) -> Tensor:
        """Forward pass for the TPE block

        Args:
            x (Tensor): Input image tensor
            rel_times (Tensor): Tensor of image acquisition times relative to first image

        Returns:
            Tensor: Tensor output after applying TPE and dropout, the tensor should be of shape
              `batch_size x seq_len x d_model`
        """
        rel_times: Tensor = rel_times.to(device=x.device, dtype=long)
        x: Tensor = x + self.pe[0, rel_times, :]
        result: Tensor = self.dropout(x)
        return result
