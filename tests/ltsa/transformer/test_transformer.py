import torch

from ltsa.transformer.transformer import Transformer


def test_full_transformer_unpacks_custom_attention_outputs() -> None:
    model = Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        batch_first=True,
    )

    output = model(torch.randn(2, 3, 8), torch.randn(2, 2, 8))

    assert output.shape == (2, 2, 8)
