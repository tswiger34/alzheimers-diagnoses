"""Module for useful types"""

from dataclasses import dataclass
from typing import Hashable, Literal, NamedTuple, SupportsFloat

from torch import Tensor

from ltsa import ImageEncoder
from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


class TranformerLayerOutput(NamedTuple):
    """Output tensors from a single layer in a transformer model

    Attributes:
        feats (Tensor): Output features
        attn_map (Tensor): List of attention mappings from each layer in the transformer
    """

    feats: Tensor
    attn_map: Tensor


class TransformerOutput(NamedTuple):
    """Output tensors from a transformer model

    Attributes:
        feats (Tensor): Output features
        attn_maps (list[Tensor]): List of attention mappings from each layer in the transformer
    """

    feats: Tensor
    attn_maps: list[Tensor]


@dataclass(slots=True)
class TransformerConfig:
    """Configuration for constructing an LTSA transformer encoder.

    Attributes:
        n_heads (int): Number of attention heads in each encoder layer.
        n_layers (int): Number of stacked transformer encoder layers.
        dropout (float): Dropout probability applied within the transformer layers.
        activation (Literal["relu", "gelu"]): Activation function used by each encoder layer feedforward block.
        batch_first (bool): Whether encoder layers expect input tensors in batch-first format. Defaults to True
    """

    n_heads: int
    n_layers: int
    dropout: float
    activation: Literal["relu", "gelu"] = "relu"
    batch_first: bool = True

    def to_encoder_layer(self, img_encoder: ImageEncoder) -> TransformerEncoderLayer:
        """Build a transformer encoder layer for a given image encoder.

        Args:
            img_encoder (ImageEncoder): Image encoder whose feature width defines the transformer model dimension.

        Returns:
            TransformerEncoderLayer: Configured encoder layer matching the image encoder feature width.
        """
        return TransformerEncoderLayer(
            d_model=img_encoder.n_features,
            nhead=self.n_heads,
            dim_feedforward=img_encoder.n_features,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=self.batch_first,
        )

    def to_encoder(self, img_encoder: ImageEncoder) -> TransformerEncoder:
        """Build a transformer encoder for a given image encoder.

        Args:
            img_encoder (ImageEncoder): Image encoder whose feature width defines the transformer model dimension.

        Returns:
            TransformerEncoder: Configured transformer encoder composed of ``n_layers`` encoder layers.
        """
        return TransformerEncoder(
            encoder_layer=self.to_encoder_layer(img_encoder=img_encoder), num_layers=self.n_layers
        )


@dataclass(slots=True)
class DatasetMetadata:
    """Important metadata for your dataset, not required but it can help you organize your experiment

    Attributes:
        n_classes (int): The number of outcome classification labels for your dataset
        max_time_periods (int | float): The max number of time periods a subject can be observed for
        class_label_mapping (dict[Hashable, str], optional): Mapping of class labels to their human-readable names
    """

    n_classes: int
    max_time_periods: int | float
    class_label_mapping: dict[Hashable, str] | None


@dataclass(slots=True)
class SurvivalAnalysisMetadata:
    """Required metadata for single observation in a dataset being used with the LTSA model.

    This is a dataclass with :attr:``slots`` set to `True`, so when subclassing this do not use ``__slots__`` to
    retrieve the field names. Check documentation at https://docs.python.org/3/library/dataclasses.html for
    more info.

    Pytorch's `Tensor` object does not allow for a way type hint what type of values should be in the tensor, so
    the expected types are hinted using a union

    It is recommended that you subclass this object with other attributes useful for your implementation, and
    potentially refine type hints, e.g. exactly what type is used for :attr:``id``, or a clearer definition of
    the time granularity being used in :attr:``event_time`` or :attr:``obs_time``. See examples below

    Attributes:
        id (Tensor | Hashable): Unique identifier for the record
        censorship (Tensor | Literal[1, 0]): The censorship status of the observation
        obs_time (Tensor | SupportsFloat): The time that the record was observed
        event_time (Tensor | SupportsFloat): The time that the event of interest occurred
        label (Tensor | str | int): The classification label of the observation

    Examples:
    ---

    *Simple example where we have a string :attr:``id``, 3 possible classifications, and want to access a single
    image tensor per observation, and a subject ID*

    ```python

    from ltsa import SurvivalAnalysisMetadata

    class ImageMetadata(SurvivalAnalysisMetadata):
        # Refine existing type hints
        id: Tensor | str
        label: Tensor | Literal["class_1", "class_2", "class_3"]

        # Add in new metadata
        img_tensor: Tensor
        subject_id: Tensor | str
    ```
    """

    id: Tensor | Hashable
    censorship: Tensor | Literal[1, 0]
    obs_time: Tensor | SupportsFloat
    event_time: Tensor | SupportsFloat
    label: Tensor | str | int


@dataclass(slots=True)
class LTSAOutputs:
    hazards: Tensor
    surv: Tensor
    feat_preds: Tensor
    feat_targets: Tensor
    padding_mask: Tensor
    ## TODO: Clarify attn_map output type
    attn_map: list[Tensor] | Tensor | None
