from dataclasses import replace

import polars as pl
import pytest
import torch
from torch import nn

from mstat_project.ml.continuous_survival import ContinuousHazardHead, ContinuousMRIModel
from mstat_project.ml.survival import SurvivalConfig, summarize_predictions, train_variant
from mstat_project.ml.survival_data import (
    SurvivalDataset, at_landmark, build_observations,
    collate_survival, preflight_tensors, sample_epoch,
)


class TinyEncoder(nn.Module):
    n_features = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)

    def forward(self, x):
        return self.linear(x.flatten(1)[:, :1])


def fixture_data(tmp_path):
    schedule = pl.DataFrame({"visit_num": [0, 1, 2, 3, 4, 5, 6], "visit_code": ["BL", "v1", "v2", "v3", "v4", "v5", "v6"],
        "follow_up_months": [0, 6, 12, 24, 36, 48, 60], "follow_up_start": [0, 0, 9, 18, 30, 42, 54],
        "follow_up_end": [0, 9, 18, 30, 42, 54, 66]})
    rows, assignments = [], []
    for i, split in enumerate(("train", "train", "validation", "validation", "test", "test")):
        ptid = f"P{i}"
        assignments.append({"ptid": ptid, "split": split})
        ids = [f"{ptid}_{j}" for j in [0, 1, 2, 3, 6]]
        torch.save({"ptid": ptid, "img_ids": list(reversed(ids)),
                    "images": torch.randn(5, 1, 16, 16, 16)}, tmp_path / f"{ptid}.pt")
        for v, s, image_id in zip([0,1,2,3,6], [0.,6.3,12.8,23.4,60.2], ids, strict=True):
            rows.append({"ptid": ptid, "image_id": image_id, "visit_num": v,
                         "visit_code": "BL" if v == 0 else f"v{v}", "binning_months": s,
                         "acquisition_months": s, "outcome_months": 67.2 if i % 2 == 0 else 91.6,
                         "event_observed": i % 2 == 0, "exclusion_reason": None})
    frame, splits = pl.DataFrame(rows), pl.DataFrame(assignments)
    records, _ = build_observations(frame, schedule, splits, tmp_path)
    return records, frame, schedule, splits


def test_constant_hazard_conditional_loss_and_survival():
    h = ContinuousHazardHead(3)
    with torch.no_grad():
        for p in h.parameters():
            p.zero_()
    z = torch.randn(2, 3, requires_grad=True)
    s, t, event = torch.tensor([12.8, 23.4]), torch.tensor([67.2, 91.6]), torch.tensor([1., 0.])
    rate = (torch.log(torch.tensor(2.)) + 1e-7) / 12
    expected = rate * (t-s) - event * rate.log()
    torch.testing.assert_close(h.nll(z,s,t,event), expected)
    torch.testing.assert_close(h.survival(z,s,t), torch.exp(-rate*(t-s)))
    h.nll(z,s,t,event).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    torch.testing.assert_close(h.survival(z, s, s), torch.ones(2))


def test_quadrature_agrees_with_dense_integration():
    torch.manual_seed(3)
    h = ContinuousHazardHead(3)
    z = torch.randn(2,3)
    s, t = torch.tensor([0.,60.]), torch.tensor([200.,240.])
    u = s[:,None] + (t-s)[:,None] * torch.linspace(0,1,20001)
    reference = torch.trapezoid(h.hazard(z,u), u, dim=1)
    torch.testing.assert_close(h.cumulative_hazard(z,s,t), reference, atol=1e-5, rtol=1e-5)


def test_month60_uses_month24_and_preserves_missing_bins(tmp_path):
    records, *_ = fixture_data(tmp_path)
    rows = at_landmark(records, 60, "test")
    assert len(rows) == 2
    assert rows[0].acquisition_months[-1] == 23.4
    assert rows[0].visit_nums == (0,1,2,3)
    assert rows[0].prediction_months == 60
    # The 60.2-month scan belongs to the 60-month bin but is not available at t=60.
    later = at_landmark(records, 61, "test")
    d = SurvivalDataset(later, history=True, n_slots=7, spatial_size=(16,16,16))
    batch = collate_survival([d[0],d[1]])
    assert batch["observed"][0].tolist() == [True,True,True,True,False,False,True]
    single = SurvivalDataset(rows, history=False, n_slots=7, spatial_size=(16,16,16))
    assert single[0]["images"].shape[0] == 1
    assert single.images.records[0].image_ids == rows[0].image_ids[-1:]


def test_at_risk_and_event_visit_exclusion(tmp_path):
    records, frame, schedule, splits = fixture_data(tmp_path)
    assert len(at_landmark(records,67.2,"test")) == 1
    assert at_landmark(records,100,"test") == []
    invalid = frame.with_columns(pl.lit(60.2).alias("outcome_months"))
    with pytest.raises(ValueError, match="strictly precede"):
        build_observations(invalid, schedule, splits, tmp_path)
    excluded = invalid.with_columns(pl.when(pl.col("acquisition_months") >= 60.2).then(pl.lit("at_or_after_outcome")).otherwise(None).alias("exclusion_reason"))
    usable, _ = build_observations(excluded, schedule, splits, tmp_path)
    assert all(r.acquisition_months[-1] < 60.2 for r in usable)


def test_patient_sampling_reproducible_balanced_and_varies_by_epoch(tmp_path):
    records, *_ = fixture_data(tmp_path)
    sampled = sample_epoch(records,42,1)
    assert len(sampled) == len({r.ptid for r in records})
    assert sampled == sample_epoch(list(reversed(records)),42,1)
    assert sampled != sample_epoch(records,42,2)
    d = SurvivalDataset(records, history=False,n_slots=7,spatial_size=(16,16,16))
    assert sum(1/d.counts[r.ptid] for r in records if r.ptid == records[0].ptid) == pytest.approx(1.)


def test_preflight_rejects_missing_id_and_accepts_reordered_tensor(tmp_path):
    records, *_ = fixture_data(tmp_path)
    assert preflight_tensors(records)["tensor_patients_checked"] == 6
    bad = [replace(r, image_ids=(*r.image_ids, "missing")) for r in records]
    with pytest.raises(ValueError,match="missing 1 selected image"):
        preflight_tensors(bad)


@pytest.mark.parametrize("error",["split", "duplicate_bin", "wrong_bin", "invalid_time"])
def test_cohort_rejects_invalid_source(tmp_path,error):
    _, f, b, s = fixture_data(tmp_path)
    if error == "split": s = pl.concat([s,s.head(1)])
    if error == "duplicate_bin": f = pl.concat([f,f.head(1)])
    if error == "wrong_bin": f = f.with_columns(pl.lit("BL").alias("visit_code"))
    if error == "invalid_time": f = f.with_columns(pl.lit(float("nan")).alias("acquisition_months"))
    with pytest.raises(ValueError): build_observations(f,b,s,tmp_path)


def test_a_ignores_acquisition_time_b_uses_it_and_common_initialization():
    torch.manual_seed(42)
    a = ContinuousMRIModel(TinyEncoder(),"a")
    torch.manual_seed(42)
    b = ContinuousMRIModel(TinyEncoder(),"b")
    for name, weight in a.state_dict().items(): torch.testing.assert_close(weight,b.state_dict()[name])
    images, mask = torch.randn(2,1,1,16,16,16), torch.ones(2,1,dtype=torch.bool)
    t0, t1 = torch.zeros(2,1), torch.ones(2,1)*24
    torch.testing.assert_close(a.encode(images,t0,mask),a.encode(images,t1,mask))
    assert not torch.allclose(b.encode(images,t0,mask),b.encode(images,t1,mask))


def test_history_masks_missing_images_and_times():
    m = ContinuousMRIModel(TinyEncoder(),"history").eval()
    images, times = torch.randn(2,4,1,16,16,16), torch.tensor([[0.,6.,0.,24.]]*2)
    observed = torch.tensor([[True,True,False,True]]*2)
    z = m.encode(images,times,observed)
    images[:,2] = 999
    times[:,2] = -999
    torch.testing.assert_close(z,m.encode(images,times,observed))


@pytest.mark.parametrize("variant",["a","b","history"])
def test_two_epoch_pipeline_selects_and_saves_predictions(tmp_path,variant):
    torch.set_num_threads(2)
    records, _, schedule, _ = fixture_data(tmp_path)
    config = SurvivalConfig(epochs=2,patience=2,device="cpu",pretrained=False,spatial_size=(16,16,16))
    result = train_variant(config,records,schedule,variant,tmp_path/variant,encoder_factory=TinyEncoder)
    assert result["best_epoch"] in (1,2)
    assert result["validation_by_landmark"]["60.0"]["patients"] == 2
    assert (tmp_path/variant/"best.pt").is_file()
    assert (tmp_path/variant/"validation_predictions.json").is_file()
    assert not (tmp_path/variant/"test_predictions.json").exists()


def test_no_comparable_pairs_report_null_not_zero():
    p = [{"outcome_months":100,"landmark_months":60,"event":False,"nll":1,"survival":[.8]}]
    assert summarize_predictions(p,[12])["c_index"] == [None]


def test_full_run_saves_matched_samples_and_comparison(tmp_path, monkeypatch):
    import json
    import mstat_project.ml.survival as pipeline

    records, frame, schedule, splits = fixture_data(tmp_path)
    monkeypatch.setattr(pipeline, "load_source", lambda engine: (frame, schedule, splits))
    stored = []
    monkeypatch.setattr(pipeline, "_persist_run", lambda engine, run_id, output, status, payload: stored.append(status))
    original_train = pipeline.train_variant

    def train_tiny(config, records, schedule, variant, output):
        return original_train(config, records, schedule, variant, output, encoder_factory=TinyEncoder)

    monkeypatch.setattr(pipeline, "train_variant", train_tiny)
    config = SurvivalConfig(
        epochs=2, patience=2, device="cpu", pretrained=False, spatial_size=(16, 16, 16),
        tensor_dir=tmp_path, output_dir=tmp_path / "runs", bootstrap_samples=0,
    )
    config = synthetic_gate(config, records, frame, schedule, splits, tmp_path)
    output = pipeline.run(config, engine=object())
    assert stored == ["running", "running", "running", "completed"]
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert (output / "source.parquet").exists()
    assert (output / "training_landmarks.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["audit"]["tensor_patients_checked"] == 6
    assert manifest["code_sha256"]["survival.py"]
    comparisons = json.loads((output / "comparisons.json").read_text())
    assert comparisons and all(r["candidate"] == "b" and r["reference"] == "a" for r in comparisons)
    for epoch in (1, 2):
        name = f"sampled_epoch_{epoch:03d}.csv"
        assert (output / "a" / name).read_bytes() == (output / "b" / name).read_bytes()


class DiagnosticEncoder(nn.Module):
    """Tiny normalized-image encoder; never constructs ResNet or uses CUDA."""
    n_features = 8

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(12, 8))
        self.batch_sizes = []

    def volume_to_resnet_input(self, x):
        return x[:, :, x.shape[2] // 2].expand(-1, 3, -1, -1)

    def forward(self, x):
        self.batch_sizes.append(len(x))
        return self.model(self.volume_to_resnet_input(x))


def test_diagnostic_panel_is_fixed_balanced_and_excludes_test(tmp_path):
    from mstat_project.ml.survival_diagnostics import select_diagnostic_panel

    records, *_ = fixture_data(tmp_path)
    panel = select_diagnostic_panel(records, 42)
    assert panel == select_diagnostic_panel(list(reversed(records)), 42)
    assert len({r.ptid for r in panel}) == 4
    assert {(r.split, r.event) for r in panel} == {
        ("train", True), ("train", False), ("validation", True), ("validation", False),
    }
    assert all(len(r.image_ids) == 1 for r in panel)
    assert all(r.acquisition_months == (0.,) for r in panel)


@pytest.mark.parametrize("variant", ["a", "b", "history"])
def test_diagnostic_trace_matches_forward_and_preserves_state(tmp_path, variant):
    from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel

    torch.set_num_threads(1)
    records, *_ = fixture_data(tmp_path)
    panel = select_diagnostic_panel(records, 42)
    model = ContinuousMRIModel(DiagnosticEncoder(), variant)
    model.train()
    model.encoder.model.eval()  # Deliberately mixed modes must survive the trace.
    modes = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = trace_diagnostic_panel(model, panel, spatial_size=(16,16,16), device="cpu", horizons=(12.,24.,60.))
    assert model.encoder.batch_sizes == [1] * 4
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    expected_stages = [
        "loaded_volume", "preprocessed_volume", "normalized_input", "encoder_features",
        "projected_features", "head_embedding",
    ]
    if variant == "history":
        expected_stages.append("temporal_features")
    expected_stages.extend(["hidden_preactivation", "hidden_activation", "hazard", "cumulative_hazard", "survival", "risk"])
    assert list(report["panel_variation"]) == expected_stages
    model.eval()
    dataset = SurvivalDataset(panel, history=variant == "history", n_slots=7, spatial_size=(16,16,16))
    with torch.no_grad():
        for i, row in enumerate(report["patients"]):
            batch = collate_survival([dataset[i]])
            z = model.encode(batch["images"], batch["times"], batch["observed"])
            expected = model.head.hazard(z, torch.tensor([report["hazard_query_months"]]))
            torch.testing.assert_close(torch.tensor(row["hazard_per_month"]), expected.flatten())
            survival = torch.stack([model.head.survival(z, torch.tensor([0.]), torch.tensor([h]))
                                    for h in (12.,24.,60.)]).flatten()
            torch.testing.assert_close(torch.tensor(row["survival"]), survival)


def test_diagnostics_identifies_saturated_constant_head_and_cleans_up_on_error(tmp_path, monkeypatch):
    from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel

    records, *_ = fixture_data(tmp_path)
    panel = select_diagnostic_panel(records, 42)
    model = ContinuousMRIModel(DiagnosticEncoder())
    # Retain coverage for custom heads that still use Tanh; production uses GELU.
    model.head.network[1] = nn.Tanh()
    with torch.no_grad():
        model.head.network[0].weight.zero_()
        model.head.network[0].bias.fill_(100.)
    report = trace_diagnostic_panel(model, panel, spatial_size=(16,16,16), device="cpu", horizons=(12.,60.))
    assert report["first_near_constant_stage"] == "hidden_preactivation"
    assert all(r["stages"]["tanh"]["mean_local_derivative"] == 0 for r in report["patients"])
    assert all(r["hazard_near_constant_across_times"] for r in report["patients"])
    assert any("saturated" in w for w in report["warnings"])

    def fail(*args):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(model.encoder, "forward", fail)
    with pytest.raises(RuntimeError, match="probe failure"):
        trace_diagnostic_panel(model, panel, spatial_size=(16,16,16), device="cpu", horizons=(12.,))
    assert model.training
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_diagnostic_artifacts_and_training_are_unchanged(tmp_path):
    import json

    records, _, schedule, _ = fixture_data(tmp_path)
    config = SurvivalConfig(epochs=1, patience=1, device="cpu", pretrained=False, spatial_size=(16,16,16))
    results = []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        results.append(train_variant(replace(config, diagnostics=enabled), records, schedule, "a", output,
                                     encoder_factory=DiagnosticEncoder))
        if enabled:
            before = json.loads((output / "diagnostics_epoch_000.json").read_text())
            after = json.loads((output / "diagnostics_epoch_001.json").read_text())
            assert [p["image_id"] for p in before["patients"]] == [p["image_id"] for p in after["patients"]]
        else:
            assert not list(output.glob("diagnostics*.json"))
    assert results[0] == results[1]


def test_diagnostics_cli_flag(monkeypatch):
    import mstat_project.ml.survival as pipeline

    monkeypatch.setattr(pipeline, "run", lambda config, **kwargs: config)
    assert pipeline.main(["--diagnostics"]).diagnostics is True
    assert pipeline.main([]).diagnostics is False


def test_diagnostics_nonfinite_values_are_json_safe_and_missing_strata_reported(tmp_path):
    import json
    from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel

    records, *_ = fixture_data(tmp_path)
    panel = select_diagnostic_panel([r for r in records if r.split == "train" and r.event], 42)
    model = ContinuousMRIModel(DiagnosticEncoder())
    with torch.no_grad():
        model.projection.weight.fill_(float("nan"))
    report = trace_diagnostic_panel(model, panel, spatial_size=(16,16,16), device="cpu", horizons=(12.,))
    json.dumps(report, allow_nan=False)
    assert report["panel_variation"]["hazard"]["near_constant"] is None
    assert any("Nonfinite" in w for w in report["warnings"])
    assert any("fewer than four" in w for w in report["warnings"])


@pytest.mark.parametrize("variant", ["a", "b", "history"])
def test_frozen_encoder_preserves_weights_buffers_and_features_while_head_learns(variant):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    encoder = DiagnosticEncoder()
    encoder.model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(.5), encoder.model)
    model = ContinuousMRIModel(encoder, variant, freeze_encoder=True)
    model.train()
    assert model.training and model.projection.training and model.head.training
    assert all(not module.training for module in encoder.modules())
    assert all(not p.requires_grad for p in encoder.parameters())
    initial = {k: v.clone() for k, v in encoder.state_dict().items()}
    initial_projection = model.projection.weight.detach().clone()
    initial_head = model.head.network[2].weight.detach().clone()
    images = torch.randn(2, 1, 1, 16, 16, 16)
    features = encoder(images[:, 0]).clone()
    batch = {"images": images, "times": torch.zeros(2, 1), "observed": torch.ones(2, 1, dtype=torch.bool),
             "entry": torch.zeros(2), "outcome": torch.tensor([12., 60.]), "event": torch.tensor([1., 0.])}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model(batch).mean().backward()
        optimizer.step()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(v, initial[k]) for k, v in encoder.state_dict().items())
    assert torch.equal(encoder(images[:, 0]), features)
    assert not torch.equal(model.projection.weight, initial_projection)
    assert not torch.equal(model.head.network[2].weight, initial_head)


def test_frozen_encoder_pipeline_and_cli(tmp_path, monkeypatch):
    import json
    import mstat_project.ml.survival as pipeline

    records, _, schedule, _ = fixture_data(tmp_path)
    config = SurvivalConfig(epochs=1, patience=1, device="cpu", pretrained=False,
                            spatial_size=(16,16,16), diagnostics=True, freeze_encoder=True)
    output = tmp_path / "frozen"
    train_variant(config, records, schedule, "a", output, encoder_factory=DiagnosticEncoder)
    before = json.loads((output / "diagnostics_epoch_000.json").read_text())
    after = json.loads((output / "diagnostics_epoch_001.json").read_text())
    assert before["freeze_encoder"] and after["freeze_encoder"]
    assert [p["stages"]["encoder_features"] for p in before["patients"]] == [
        p["stages"]["encoder_features"] for p in after["patients"]]
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["freeze_encoder"] is True
    # A excludes its unused time embedding and frozen encoder, and includes both normalizations.
    assert len(checkpoint["optimizer"]["param_groups"][0]["params"]) == 10
    monkeypatch.setattr(pipeline, "run", lambda config, **kwargs: config)
    assert pipeline.main(["--freeze-encoder"]).freeze_encoder is True
    assert pipeline.main([]).freeze_encoder is True
    assert pipeline.main(["--no-freeze-encoder"]).freeze_encoder is False


def synthetic_gate(config, records, frame, schedule, splits, tmp_path):
    """Supply deterministic passing evidence for orchestration tests, not research runs."""
    import copy
    from mstat_project.data_prep.data_splitting import frame_fingerprint
    from mstat_project.ml.survival import _seed, model_identity, write_json
    from mstat_project.ml.survival_embeddings import fingerprint
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH, read_json, implementation_hashes, cohort_fingerprint

    protocol = read_json(PROTOCOL_PATH)
    protocol["snapshot_hashes"] = {"source": frame_fingerprint(frame), "schedule": frame_fingerprint(schedule),
                                   "splits": frame_fingerprint(splits)}
    protocol_path = tmp_path / "protocol.json"
    gate_path = tmp_path / "gate.json"
    write_json(protocol_path, protocol)
    _seed(config.seed)
    model = ContinuousMRIModel(TinyEncoder(), freeze_encoder=True)
    identity = model_identity(model, records, config)
    result = {"passed": True, "initial": {"nll": 2.0},
              "final": {"nll": 1.0, "c_index": 1.0, "risk_variance": .1},
              "initialization_check": {"passed": True, "max_survival_difference": .001}}
    write_json(gate_path, {"passed": True, "protocol_sha256": fingerprint(protocol),
        "snapshot_hashes": protocol["snapshot_hashes"], "encoder_sha256": identity["encoder_sha256"],
        "implementation_hashes": implementation_hashes(), "full_cohort_sha256": cohort_fingerprint(records),
        "arms": {arm: copy.deepcopy(result) for arm in ("a", "b", "history")}})
    return replace(config, debug_gate=gate_path, protocol_path=protocol_path)
