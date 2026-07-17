import torch

from ltsa.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderLayer


def test_encoder_returns_per_head_attention_and_honors_padding() -> None:
    layer = TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True, dropout=0.0)
    encoder = TransformerEncoder(layer, num_layers=1)
    source = torch.randn(2, 3, 8)
    causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)

    output, attention = encoder(
        source,
        mask=causal_mask,
        src_key_padding_mask=torch.tensor([[False, False, False], [False, False, True]]),
        is_causal=True,
        need_weights=True,
    )

    assert output.shape == source.shape
    assert attention[0] is not None
    assert attention[0].shape == (2, 2, 3, 3)
