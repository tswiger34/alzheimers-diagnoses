import torch

from ltsa.transformer.transformer_decoder import TransformerDecoder, TransformerDecoderLayer


def test_decoder_propagates_per_head_attention() -> None:
    layer = TransformerDecoderLayer(d_model=8, nhead=2, batch_first=True, dropout=0.0)
    decoder = TransformerDecoder(layer, num_layers=1)

    output, self_attention, cross_attention = decoder(
        torch.randn(2, 3, 8),
        torch.randn(2, 4, 8),
        need_weights=True,
    )

    assert output.shape == (2, 3, 8)
    assert self_attention[0].shape == (2, 2, 3, 3)
    assert cross_attention[0].shape == (2, 2, 3, 4)
