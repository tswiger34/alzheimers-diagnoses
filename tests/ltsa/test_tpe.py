"""Tests for tpe module

Expected behaviors:
- for `__init__`:
  - Should expose the `pe` and `dropout` attributes
  - pe should be of shape
"""

import pytest  # noqa


def test_smoke_test():
    """Smoke test that everything can get imported and classes get initialized"""
    from ltsa.tpe import TemporalPositionalEncoding

    tpe: TemporalPositionalEncoding = TemporalPositionalEncoding(d_model=512, dropout=0.5, max_len=5000)
    assert isinstance(tpe, TemporalPositionalEncoding)
