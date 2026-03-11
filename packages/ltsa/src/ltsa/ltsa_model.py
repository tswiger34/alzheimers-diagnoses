import torch
from torch import Tensor

from ltsa.image_encoder import ImageEncoder
from ltsa.models import LTSAOutputs
from ltsa.tpe import TemporalPositionalEncoding
from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class LTSA(torch.nn.Module):
    """Full LTSA architecture implementation

    The input to :class:``LTSA`` on the forward pass consists of a collection of longitudinal images and key
    metadata for performing survival analysis. Each observation in the pytorch dataset should *at least*
    include the image and the associated metadata. For examples of how to organize your data into valid datasets
    see `docs/COOKBOOK.md`. The minimal metadata is layed out in :class:``SuvivalAnalysisMetadata``.
    """

    def __init__(
        self,
        img_encoder: ImageEncoder,
        n_heads: int,
        dropout: float,
        n_layers: int,
        max_seq_len: int,
        n_classes: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.max_seq_len: int = max_seq_len
        self.device: torch.device | None = device
        self.img_encoder: ImageEncoder = img_encoder

        transformer_encoder = TransformerEncoderLayer(
            d_model=self.img_encoder.n_features,
            nhead=n_heads,
            dim_feedforward=self.img_encoder.n_features,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer: TransformerEncoder = TransformerEncoder(
            encoder_layer=transformer_encoder, num_layers=n_layers
        )

        self.tpe: TemporalPositionalEncoding = TemporalPositionalEncoding(
            d_model=self.img_encoder.n_features, dropout=0, max_len=max_seq_len * 12
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.img_encoder.n_features, out_features=n_classes),
            torch.nn.Sigmoid(),
        )

        self.step_ahead_predictor = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=self.img_encoder.n_features, out_features=self.img_encoder.n_features),
        )

        self.causal_mask: Tensor = torch.triu(
            input=torch.full(size=(max_seq_len, max_seq_len), fill_value=float("-inf"), device=self.device),
            diagonal=1,
        )

    def _create_mask(self, tpe_augmented: Tensor, seq_lengths: tuple[int]) -> Tensor:
        """Create a key padding mask for variable-length longitudinal sequences.

        Masks all tokens beyond last visit, 1 = pad (ignore), 0 = valid (keep)

        Args:
            tpe_augmented: Batch-first tensor of temporally encoded visit features with shape
                ``(batch_size, seq_len, n_features)``.
            seq_lengths: Number of valid visits for each sequence in the batch.

        Returns:
            Tensor: Float mask of shape ``(batch_size, seq_len)`` where padded positions are ``1``
            and valid positions are ``0``.
        """
        src_key_padding_mask: Tensor = (
            torch.ones(size=(tpe_augmented.shape[0], tpe_augmented.shape[1])).float().to(device=self.device)
        )
        for i, seq_length in enumerate(seq_lengths):
            src_key_padding_mask[i, :seq_length] = 0
        return src_key_padding_mask

    def forward(self, x: Tensor, seq_lengths: tuple[int], rel_times: Tensor) -> LTSAOutputs:
        embeddings: Tensor = self.img_encoder(x).reshape(  # shape: batch_size x seq_length x n_features
            len(seq_lengths), self.max_seq_len, self.img_encoder.n_features
        )

        tpe_augmented: Tensor = self.tpe(embeddings, rel_times)  # shape: batch_size x seq_len x n_features

        src_key_padding_mask: Tensor = self._create_mask(tpe_augmented=tpe_augmented, seq_lengths=seq_lengths)

        feats, attn_map = self.transformer(
            tpe_augmented,
            mask=self.causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
            need_weights=True,
        )

        # Using src_key_padding_mask undoes padding so re-pad each sequence in the batch with zeroes
        if feats.shape[1] < self.max_seq_len:
            feats: Tensor = torch.nn.functional.pad(
                input=feats, pad=(0, 0, 0, self.max_seq_len - feats.shape[1], 0, 0), mode="constant"
            )

        hazards: Tensor = self.classifier(feats)
        surv: Tensor = torch.cumprod(input=1 - hazards.view(-1, hazards.shape[-1]), dim=1).view(
            hazards.shape[0], hazards.shape[1], hazards.shape[2]
        )

        # Padding mask used to compute loss later
        padding_mask: Tensor = torch.bitwise_not(input=src_key_padding_mask.bool()).unsqueeze(dim=-1)

        # Get time elapsed (delta) between consecutive visits
        delta_times: Tensor = torch.diff(input=rel_times)  # batch x max_seq_len-1
        delta_times[delta_times < 0] = 0
        delta_times: Tensor = torch.nn.functional.pad(
            input=delta_times, pad=(0, 1), mode="constant", value=0
        )  # batch x max_seq_len

        # Use relative temporal timestep encoding to inform the model of "# months of into the future for which to predict imaging features"
        delta_encoded_feats: Tensor = self.tpe(feats, delta_times)

        # Get actual imaging features of next visit
        feat_preds: Tensor = self.step_ahead_predictor(delta_encoded_feats)
        feat_targets: Tensor = torch.nn.functional.pad(
            input=feats[:, 1:, :], pad=(0, 0, 0, 1), mode="constant", value=0
        )

        results = LTSAOutputs(
            hazards=hazards,
            surv=surv,
            feat_preds=feat_preds,
            feat_targets=feat_targets,
            padding_mask=padding_mask,
            attn_map=attn_map,
        )
        return results
