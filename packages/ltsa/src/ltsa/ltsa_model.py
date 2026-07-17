"""Longitudinal Transformer for Survival Analysis."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ltsa.image_encoder import ImageEncoder
from ltsa.models import LTSAOutputs
from ltsa.tpe import TemporalPositionalEncoding
from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class LTSA(torch.nn.Module):
    """Model longitudinal images with causal attention and discrete survival hazards.

    LTSA encodes every visit independently, adds continuous temporal positional
    information, and applies a causal Transformer over each patient's visit sequence.
    The model predicts a discrete hazard curve at every valid visit and auxiliary
    next-visit contextual features.

    Padded visits are excluded as attention keys. The causal mask prevents each visit
    from attending to future visits, allowing predictions at time ``t`` to depend only
    on images acquired on or before ``t``.

    Attributes:
        img_encoder: Image encoder that maps a batch of individual images to
                ``[batch, n_features]`` embeddings.
        n_heads: Number of Transformer attention heads. Must evenly divide the
            encoder feature width.
        dropout: Dropout probability used by the Transformer, hazard classifier,
            and step-ahead feature predictor.
        n_layers: Number of causal Transformer encoder layers.
        max_sequence_length: Maximum number of visits accepted in one sequence.
            The registered causal mask is allocated to this size.
        max_time_index: Largest relative visit time accepted by the temporal
            positional encoding.
        n_time_bins: Number of discrete hazard bins predicted at every visit.
    """

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
        """Initialize the longitudinal survival model.

        Args:
            img_encoder: Image encoder that maps a batch of individual images to
                ``[batch, n_features]`` embeddings.
            n_heads: Number of Transformer attention heads. Must evenly divide the
                encoder feature width.
            dropout: Dropout probability used by the Transformer, hazard classifier,
                and step-ahead feature predictor.
            n_layers: Number of causal Transformer encoder layers.
            max_sequence_length: Maximum number of visits accepted in one sequence.
                The registered causal mask is allocated to this size.
            max_time_index: Largest relative visit time accepted by the temporal
                positional encoding.
            n_time_bins: Number of discrete hazard bins predicted at every visit.

        Raises:
            ValueError: If the maximum sequence length, layer count, or head count is
                invalid; if fewer than two time bins are requested; if ``n_heads`` does
                not divide the encoder feature width; or if temporal encoding
                configuration is invalid.
        """

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
        """Validate sequence metadata and normalize sequence lengths.

        Args:
            images: Padded longitudinal images with shape
                ``[batch, visits, channels, ...]``.
            sequence_lengths: Number of valid visits for each observation. Values must
                be between one and the padded visit dimension.
            relative_times: Relative acquisition times with shape ``[batch, visits]``.

        Returns:
            Sequence lengths as a one-dimensional integer tensor on the image device.

        Raises:
            ValueError: If images are not sequence-shaped, the padded sequence exceeds
                the configured maximum, sequence lengths are malformed or out of range,
                or relative times do not match the batch and visit dimensions.
        """

        if images.ndim < 4:
            raise ValueError(
                f"Expected sequence-shaped images [batch, visits, channels, ...], got {tuple(images.shape)}"
            )
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
        """Predict visit-specific hazards and next-visit contextual features.

        Args:
            images: Padded longitudinal images with shape
                ``[batch, visits, channels, ...]``. The dimensions after ``visits`` must
                match the input contract of ``img_encoder``.
            sequence_lengths: Number of valid visits for each observation. May be a
                one-dimensional tensor or a sequence of integers.
            relative_times: Acquisition times relative to the first eligible time point,
                with shape ``[batch, visits]``. Valid times must fall within the temporal
                positional encoding range.

        Returns:
            An ``LTSAOutputs`` instance containing:

            - ``hazards`` and ``surv`` with shape
              ``[batch, visits, n_time_bins]``.
            - ``feat_preds`` and shifted ``feat_targets`` with shape
              ``[batch, visits, n_features]``.
            - Boolean valid-visit and next-visit masks with shape ``[batch, visits]``.
            - One optional per-head attention tensor per Transformer layer.

        Raises:
            ValueError: If sequence metadata is invalid, relative times are non-finite
                or outside the configured range, or the image encoder does not return
                ``[batch * visits, n_features]`` embeddings.
        """

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
