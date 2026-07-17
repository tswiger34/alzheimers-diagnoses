"""Longitudinal Transformer for Survival Analysis."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ltsa.image_encoder import ImageEncoder
from ltsa.models import LTSAOutputs
from ltsa.tpe import TemporalPositionalEncoding
from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class LTSA(torch.nn.Module):
    """Causal longitudinal image model with discrete survival predictions."""

    def __init__(
        self,
        img_encoder: ImageEncoder,
        *,
        n_heads: int,
        dropout: float,
        n_layers: int,
        max_sequence_length: int,
        max_time_index: float,
        n_time_bins: int,
    ) -> None:
        super().__init__()
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if n_time_bins < 2:
            raise ValueError("n_time_bins must be at least 2")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if n_heads < 1 or img_encoder.n_features % n_heads != 0:
            raise ValueError("n_heads must evenly divide the image encoder feature width")

        self.max_sequence_length = max_sequence_length
        self.img_encoder = img_encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=img_encoder.n_features,
            nhead=n_heads,
            dim_feedforward=img_encoder.n_features,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.tpe = TemporalPositionalEncoding(
            img_encoder.n_features,
            dropout=0.0,
            max_time_index=max_time_index,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(img_encoder.n_features, n_time_bins),
            torch.nn.Sigmoid(),
        )
        self.step_ahead_predictor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(img_encoder.n_features, img_encoder.n_features),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones((max_sequence_length, max_sequence_length), dtype=torch.bool),
                diagonal=1,
            ),
        )

    def _validate_inputs(
        self,
        images: Tensor,
        sequence_lengths: Tensor | Sequence[int],
        relative_times: Tensor,
    ) -> Tensor:
        if images.ndim < 4:
            raise ValueError(f"Expected sequence-shaped images [batch, visits, ...], got {tuple(images.shape)}")
        batch_size, sequence_length = images.shape[:2]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds configured maximum {self.max_sequence_length}"
            )
        lengths = torch.as_tensor(sequence_lengths, device=images.device, dtype=torch.long)
        if lengths.shape != (batch_size,):
            raise ValueError(f"sequence_lengths must have shape ({batch_size},), got {tuple(lengths.shape)}")
        if (lengths < 1).any() or (lengths > sequence_length).any():
            raise ValueError("sequence_lengths must be between 1 and the padded sequence length")
        if relative_times.shape != (batch_size, sequence_length):
            raise ValueError(
                f"relative_times must have shape {(batch_size, sequence_length)}, got {tuple(relative_times.shape)}"
            )
        return lengths

    def forward(
        self,
        images: Tensor,
        *,
        sequence_lengths: Tensor | Sequence[int],
        relative_times: Tensor,
    ) -> LTSAOutputs:
        lengths = self._validate_inputs(images, sequence_lengths, relative_times)
        batch_size, sequence_length = images.shape[:2]
        embeddings = self.img_encoder(images.reshape(batch_size * sequence_length, *images.shape[2:]))
        if embeddings.ndim != 2 or embeddings.shape != (
            batch_size * sequence_length,
            self.img_encoder.n_features,
        ):
            raise ValueError(
                f"Image encoder must return [batch * visits, n_features], got {tuple(embeddings.shape)}"
            )
        embeddings = embeddings.reshape(batch_size, sequence_length, self.img_encoder.n_features)
        temporal_features = self.tpe(embeddings, relative_times)

        visit_indices = torch.arange(sequence_length, device=images.device).unsqueeze(0)
        valid_visit_mask = visit_indices < lengths.unsqueeze(1)
        features, attention_maps = self.transformer(
            temporal_features,
            mask=self.get_buffer("causal_mask")[:sequence_length, :sequence_length],
            src_key_padding_mask=~valid_visit_mask,
            is_causal=True,
            need_weights=True,
        )

        hazards = self.classifier(features)
        survival = torch.cumprod(1 - hazards, dim=-1)
        delta_times = torch.diff(relative_times, dim=1).clamp_min(0)
        delta_times = torch.nn.functional.pad(delta_times, (0, 1), mode="constant", value=0)
        feature_predictions = self.step_ahead_predictor(self.tpe(features, delta_times))
        feature_targets = torch.nn.functional.pad(features[:, 1:, :], (0, 0, 0, 1), value=0)
        next_visit_mask = visit_indices < (lengths - 1).clamp_min(0).unsqueeze(1)

        return LTSAOutputs(
            hazards=hazards,
            surv=survival,
            feat_preds=feature_predictions,
            feat_targets=feature_targets,
            valid_visit_mask=valid_visit_mask,
            next_visit_mask=next_visit_mask,
            attn_map=attention_maps,
        )
