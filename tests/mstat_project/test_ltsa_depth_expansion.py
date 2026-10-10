"""Synthetic acceptance checks for controlled two-layer LTSA expansion."""

import pytest
import torch

from test_ltsa_stabilization import cached  # noqa: F401
from test_ltsa_exploratory import exploratory_case  # noqa: F401
from test_ltsa_b_continuation import completed_history, run_control  # noqa: F401
from mstat_project.ml import ltsa_depth_expansion as depth
from mstat_project.ml.continuous_survival import ContinuousMRIModel
from mstat_project.ml.survival import _seed, write_json
from mstat_project.ml.survival_embeddings import CachedEncoder
from mstat_project.ml.survival_protocol import file_sha256, read_json


@pytest.fixture
def depth_case(completed_history, monkeypatch):
    case, history = completed_history
    continuation = run_control(case)
    report = read_json(continuation / "report.json")
    report["stable"] = True  # Synthetic stability evidence, not a clinical claim.
    write_json(continuation / "report.json", report)
    real_stable = depth.stable_run
    monkeypatch.setattr(depth, "stable_run", lambda path: True if path == continuation / "b-continuation" else real_stable(path))
    return case, history, continuation


def run_depth(case):
    cached, protocol, protocol_path, _, _ = case
    return depth.run_depth(cached.path.parent, cached.records, cached.schedule, protocol, protocol_path,
        "cpu", "debug_test", "exploratory_test", "exploratory_continuation_test", "exploratory_depth_test")


def test_two_layer_expansion_preserves_controls_initialization_and_epoch_zero(depth_case, monkeypatch):
    case, history, continuation = depth_case
    originals = {p: file_sha256(p) for directory in (history, continuation) for p in directory.rglob("*") if p.is_file()}
    original_train = depth.train_variant

    def checked(config, records, schedule, variant, output):
        assert config.n_layers == 2 and config.d_model == 128 and config.dropout == .1
        assert config.freeze_encoder and config.auxiliary_weight == 0 and not config.evaluate_test
        assert config.baseline_init == case[0].path.parent / "seed_42" / "b" / "best.pt"
        return original_train(config, records, schedule, variant, output)

    monkeypatch.setattr(depth, "train_variant", checked)
    output = run_depth(case)
    report = read_json(output / "report.json")
    assert report["exploratory"] and not report["production_promoted"]
    assert report["validation_nll"] <= report["epoch_zero_nll"] + 1e-5
    assert report["matched_sample_files"]
    assert len(report["primary_comparison"]) == 1
    assert report["primary_comparison"][0]["reference"] == "history-1l"
    assert report["primary_comparison"][0]["candidate"] == "history-2l"
    zero = torch.load(output / "epoch_zero.pt", weights_only=False)
    source = torch.load(case[0].path.parent / "seed_42" / "b" / "best.pt", weights_only=False)
    assert any(key.startswith("transformer.layers.1.") for key in zero["model"])
    assert not any(key.startswith("transformer.layers.2.") for key in zero["model"])
    for key, value in source["model"].items():
        torch.testing.assert_close(zero["model"][key], value, rtol=0, atol=0)
    assert all(file_sha256(p) == digest for p, digest in originals.items())
    assert not list(output.rglob("*test_predictions*"))
    assert run_depth(case) == output


def test_two_layers_remain_causal_and_ignore_missing_features(cached):
    _seed(42)
    model = ContinuousMRIModel(CachedEncoder(cached.manifest), "history", n_layers=2, dropout=.1).eval()
    raw = torch.randn(2, 4, cached.manifest["n_features"])
    times = torch.tensor([[0., 6., 12., 24.], [0., 6., 12., 24.]])
    observed = torch.tensor([[True, False, True, True], [True, True, False, False]])
    tokens, states = model.representations(raw, times, observed)
    changed = raw.clone()
    changed[~observed] = float("nan")
    changed_times = times.clone()
    changed_times[~observed] = float("nan")
    masked_tokens, masked_states = model.representations(changed, changed_times, observed)
    torch.testing.assert_close(tokens, masked_tokens)
    torch.testing.assert_close(states, masked_states)
    changed = raw.clone()
    changed[0, 3] += 100
    _, future_states = model.representations(changed, times, observed)
    torch.testing.assert_close(states[0, :3], future_states[0, :3])
    loss = model.head.nll(model.encode_features(raw, times, observed), torch.tensor([24., 24.]),
                          torch.tensor([60., 90.]), torch.tensor([1., 0.])).mean()
    loss.backward()
    for layer in model.transformer.layers:
        gradients = [p.grad for p in layer.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_depth_rejects_changed_control_before_training(depth_case):
    case, _, continuation = depth_case
    (continuation / "b-continuation" / "validation_predictions.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Continuation artifact changed"):
        run_depth(case)
    assert not (case[0].path.parent / "investigations" / "exploratory_depth_test").exists()
