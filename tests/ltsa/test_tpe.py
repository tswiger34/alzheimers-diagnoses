import pytest
import torch

from ltsa.tpe import TemporalPositionalEncoding


def test_temporal_encoding_supports_fractional_times_and_odd_features() -> None:
    layer = TemporalPositionalEncoding(5, dropout=0.0, max_time_index=24.0)
    values = torch.zeros(2, 3, 5)
    times = torch.tensor([[0.0, 6.5, 12.0], [0.0, 1.25, 24.0]])

    encoded = layer(values, times)

    assert encoded.shape == values.shape
    assert torch.isfinite(encoded).all()
    assert not torch.equal(encoded[0, 1], encoded[0, 2])


def test_temporal_encoding_validates_range() -> None:
    layer = TemporalPositionalEncoding(4, dropout=0.0, max_time_index=12.0)

    with pytest.raises(ValueError, match="between"):
        layer(torch.zeros(1, 1, 4), torch.tensor([[12.1]]))
