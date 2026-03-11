from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, device
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm

from ltsa.transformer.transformer_utils import get_activation_fn, get_clones


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__: list[str] = ["norm"]

    def __init__(self, decoder_layer: torch.nn.Module, num_layers: int, norm: LayerNorm | None = None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers: ModuleList = get_clones(module=decoder_layer, N=num_layers)
        self.num_layers: int = num_layers
        self.norm: LayerNorm | None = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, list, list]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output: Tensor = tgt

        attn_maps_sa: list[Any] = []
        attn_maps_mha: list[Any] = []
        for mod in self.layers:
            output, attn_map_sa, attn_map_mha = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                need_weights=need_weights,
            )
            attn_maps_sa.append(attn_map_sa)
            attn_maps_mha.append(attn_map_mha)

        if self.norm is not None:
            output: Tensor = self.norm(output)

        return output, attn_maps_sa, attn_maps_mha


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__: list[str] = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device: device | None = None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(in_features=d_model, out_features=dim_feedforward, device=device, dtype=dtype)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(in_features=dim_feedforward, out_features=d_model, device=device, dtype=dtype)

        self.norm_first: bool = norm_first
        self.norm1 = LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(normalized_shape=d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation: Callable[[Tensor], Tensor] = get_activation_fn(activation)
        else:
            self.activation: Callable[[Tensor], Tensor] = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x: Tensor = tgt
        if self.norm_first:
            _x, attn_map_sa = self._sa_block(
                x=self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, is_causal=tgt_is_causal
            )
            x: Tensor = x + _x
            _x, attn_map_mha = self._mha_block(
                x=self.norm2(x),
                mem=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_causal,
            )
            x: Tensor = x + _x
            x: Tensor = x + self._ff_block(x=self.norm3(x))
        else:
            _x, attn_map_sa = self._sa_block(
                x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, is_causal=tgt_is_causal
            )
            x: Tensor = self.norm1(x + _x)
            _x, attn_map_mha = self._mha_block(
                x=x,
                mem=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_causal,
            )
            x: Tensor = self.norm2(x + _x)
            x: Tensor = self.norm3(x + self._ff_block(x))

        return x, attn_map_sa, attn_map_mha

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        x, attn_map = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        return self.dropout1(x), attn_map

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        x, attn_map = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        return self.dropout2(x), attn_map

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x: Tensor = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
