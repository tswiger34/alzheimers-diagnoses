import pytest
import torch

from ltsa.ltsa_model import LTSA


class DummyImageEncoder(torch.nn.Module):
    n_features = 8

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=tuple(range(1, images.ndim))).unsqueeze(1)
        return pooled.repeat(1, self.n_features)


def _model() -> LTSA:
    return LTSA(
        DummyImageEncoder(),  # type: ignore[arg-type]
        n_heads=2,
        dropout=0.0,
        n_layers=1,
        max_sequence_length=3,
        max_time_index=24.0,
        n_time_bins=4,
    )


def test_ltsa_forward_masks_padding_and_backpropagates() -> None:
    model = _model()
    images = torch.randn(2, 3, 1, 8, 8, requires_grad=True)
    outputs = model(
        images,
        sequence_lengths=torch.tensor([3, 2]),
        relative_times=torch.tensor([[0.0, 6.0, 12.0], [0.0, 9.0, 0.0]]),
    )

    assert outputs.hazards.shape == (2, 3, 4)
    assert outputs.surv.shape == (2, 3, 4)
    assert outputs.valid_visit_mask.tolist() == [[True, True, True], [True, True, False]]
    assert outputs.next_visit_mask.tolist() == [[True, True, False], [True, False, False]]
    assert outputs.attn_map[0] is not None
    assert outputs.attn_map[0].shape == (2, 2, 3, 3)

    (outputs.hazards.sum() + outputs.feat_preds.sum()).backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_causal_attention_prevents_future_image_leakage() -> None:
    model = _model().eval()
    images = torch.randn(1, 3, 1, 8, 8)
    changed = images.clone()
    changed[:, 2] += 100
    kwargs = {
        "sequence_lengths": torch.tensor([3]),
        "relative_times": torch.tensor([[0.0, 6.0, 12.0]]),
    }

    with torch.no_grad():
        original = model(images, **kwargs).hazards
        perturbed = model(changed, **kwargs).hazards

    torch.testing.assert_close(original[:, :2], perturbed[:, :2])
    assert not torch.allclose(original[:, 2], perturbed[:, 2])


def test_ltsa_rejects_sequence_longer_than_configuration() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _model()(
            torch.randn(1, 4, 1, 8, 8),
            sequence_lengths=[4],
            relative_times=torch.zeros(1, 4),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ltsa_buffers_and_inputs_transfer_to_cuda() -> None:
    model = _model().to("cuda")
    outputs = model(
        torch.randn(1, 2, 1, 8, 8, device="cuda"),
        sequence_lengths=torch.tensor([2], device="cuda"),
        relative_times=torch.tensor([[0.0, 6.0]], device="cuda"),
    )

    assert outputs.hazards.device.type == "cuda"
    assert model.get_buffer("causal_mask").device.type == "cuda"
