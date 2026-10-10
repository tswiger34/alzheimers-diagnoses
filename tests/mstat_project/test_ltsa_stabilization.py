"""CPU-only acceptance tests; synthetic images, no database or pretrained downloads."""

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import polars as pl
import torch
from torch import nn

from test_continuous_survival import DiagnosticEncoder, TinyEncoder, fixture_data, synthetic_gate
from mstat_project.ml.continuous_survival import ContinuousHazardHead, ContinuousMRIModel, bounded_time, initial_hazard
from mstat_project.ml.survival import (
    SurvivalConfig, _loader, model_identity, train_epoch, train_variant, transfer_baseline,
    transfer_prediction_check, write_json,
)
from mstat_project.ml.survival_data import SurvivalDataset, at_landmark, collate_survival
from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel
from mstat_project.ml.survival_embeddings import (
    CachedEncoder, EmbeddingDataset, collate_embeddings, export_embeddings, fingerprint,
    load_embeddings, state_fingerprint, _read_cache,
)
from mstat_project.ml.survival_optimization import configure_final_block, optimizer_groups
from mstat_project.ml.survival_protocol import (
    read_json, require_debug_gate, require_baseline_gate, implementation_hashes, file_sha256,
)


@pytest.fixture
def cached(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    records, frame, schedule, splits = fixture_data(tmp_path)
    encoder = DiagnosticEncoder().eval()
    cache_path = tmp_path / "embeddings.pt"
    provenance = {"source": "synthetic", "schedule": "synthetic", "splits": "synthetic"}
    manifest = export_embeddings(cache_path, records, encoder, spatial_size=(16, 16, 16), provenance=provenance)
    return SimpleNamespace(path=cache_path, records=records, frame=frame, schedule=schedule, splits=splits,
                           encoder=encoder, manifest=manifest, provenance=provenance)


def test_head_initialization_time_scaling_and_positive_gradients():
    head = ContinuousHazardHead(8, baseline_hazard=.003)
    z = (torch.randn(3, 8) * 1000).requires_grad_()
    query = torch.tensor([[0., 180., 720.]]).expand(3, -1)
    values = head.hazard(z, query)
    assert isinstance(head.activation, nn.GELU)
    assert torch.isfinite(values).all() and (values > 0).all()
    assert values.mean().item() == pytest.approx(.003, rel=.03)
    values.sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert (bounded_time(query) < 1).all()
    torch.testing.assert_close(bounded_time(torch.tensor([0., 180.])), torch.tensor([0., .5]))
    with pytest.raises(ValueError, match="nonnegative"):
        bounded_time(torch.tensor([-1.]))
    with torch.no_grad():
        head.output_linear.weight.zero_()
    torch.testing.assert_close(head.hazard(z.detach(), query), torch.full((3, 3), .003))
    start = torch.zeros(3)
    survival = head.survival(z.detach(), start, torch.tensor([12., 24., 60.]))
    assert (survival[:-1] > survival[1:]).all()


def test_initial_hazard_is_patient_weighted_and_training_only():
    a = SimpleNamespace(ptid="a", split="train", event=True, prediction_months=0., outcome_months=20.)
    b = SimpleNamespace(ptid="b", split="train", event=False, prediction_months=0., outcome_months=40.)
    heldout = SimpleNamespace(ptid="x", split="test", event=True, prediction_months=0., outcome_months=.01)
    assert initial_hazard([a, a, b, heldout]) == pytest.approx(1/60)


@pytest.mark.parametrize("variant", ["a", "b", "history"])
def test_cached_path_matches_images_without_encoder_calls(cached, variant):
    records = at_landmark(cached.records, 61, "train")
    images = SurvivalDataset(records, history=variant == "history", n_slots=7, spatial_size=(16, 16, 16))
    embeddings = EmbeddingDataset(records, cache_path=cached.path, history=variant == "history", spatial_size=(16, 16, 16))
    raw_batch = collate_survival([images[i] for i in range(len(images))])
    cached_batch = collate_embeddings([embeddings[i] for i in range(len(embeddings))])
    raw_model = ContinuousMRIModel(copy.deepcopy(cached.encoder), variant, freeze_encoder=True).eval()
    model = ContinuousMRIModel(CachedEncoder(cached.manifest), variant, freeze_encoder=True).eval()
    model.load_state_dict({k: v for k, v in raw_model.state_dict().items() if not k.startswith("encoder.")})
    with torch.no_grad():
        torch.testing.assert_close(model.encode_batch(cached_batch), raw_model.encode_batch(raw_batch), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(model(cached_batch), raw_model(raw_batch), rtol=1e-4, atol=1e-6)
    model(cached_batch).mean().backward()
    assert model.projection.weight.grad.abs().sum() > 0
    assert not list(model.encoder.parameters())
    if variant == "history":
        assert cached_batch["observed"][0].tolist() == [True, True, True, True, False, False, True]
    assert cached_batch["records"] == raw_batch["records"]


def test_cache_rejects_wrong_provenance_missing_images_and_future_inputs(cached):
    with pytest.raises(ValueError, match="fingerprint"):
        load_embeddings(cached.path, provenance={"source": "changed"})
    with pytest.raises(ValueError, match="spatial size"):
        load_embeddings(cached.path, spatial_size=(32, 32, 32))
    r = cached.records[0]
    with pytest.raises(ValueError, match="missing or inconsistent"):
        EmbeddingDataset([replace(r, image_ids=("unknown",))], cache_path=cached.path, history=True, spatial_size=(16, 16, 16))
    with pytest.raises(ValueError, match="prefix"):
        EmbeddingDataset([replace(cached.records[1], prediction_months=0.)], cache_path=cached.path,
                         history=True, spatial_size=(16, 16, 16))
    with pytest.raises(ValueError, match="strictly precede"):
        export_embeddings(cached.path.with_name("invalid.pt"), [replace(r, outcome_months=0.)], cached.encoder,
                          spatial_size=(16, 16, 16), provenance=cached.provenance)


def test_cache_detects_corruption_and_duplicate_metadata(cached):
    package = torch.load(cached.path, weights_only=True)
    package["embeddings"][0, 0] += 1
    bad = cached.path.with_name("corrupt.pt")
    torch.save(package, bad)
    with pytest.raises(ValueError, match="embedding fingerprint"):
        load_embeddings(bad)
    package = torch.load(cached.path, weights_only=True)
    package["metadata"][1] = package["metadata"][0]
    package["manifest"]["metadata_sha256"] = fingerprint(package["metadata"])
    package["manifest"]["cache_id"] = fingerprint({k: v for k, v in package["manifest"].items() if k != "cache_id"})
    duplicate = cached.path.with_name("duplicate.pt")
    torch.save(package, duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        load_embeddings(duplicate)
    _read_cache.cache_clear()


def test_cache_export_preserves_encoder_state_and_modes(cached):
    encoder = cached.encoder.train()
    encoder.model.eval()
    modes = [m.training for m in encoder.modules()]
    before = state_fingerprint(encoder)
    export_embeddings(cached.path.with_name("second.pt"), cached.records[:2], encoder,
                      spatial_size=(16, 16, 16), provenance=cached.provenance)
    assert [m.training for m in encoder.modules()] == modes
    assert state_fingerprint(encoder) == before


def test_temporal_causality_leading_gaps_and_finite_backward():
    torch.manual_seed(42)
    model = ContinuousMRIModel(TinyEncoder(), "history", freeze_encoder=True, dropout=0.).eval()
    raw = torch.randn(2, 5, 8, requires_grad=True)
    observed = torch.tensor([[False, True, False, True, True], [True, False, False, False, True]])
    times = torch.tensor([[999., 6., -999., 24., 60.], [0., -999., -999., -999., 60.]])
    _, before = model.representations(raw, times, observed)
    changed = raw.detach().clone()
    changed[:, -1] += 100
    changed[~observed] = float("nan")
    _, after = model.representations(changed, times, observed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    z = model.encode_features(raw, times, observed)
    assert torch.isfinite(z).all()
    z.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[~observed].abs().sum() == 0
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="at least one"):
        model.encode_features(raw, times, torch.zeros_like(observed))


def test_transfer_copies_all_shared_modules_and_preserves_baseline(cached):
    config = SurvivalConfig(cache_path=cached.path, spatial_size=(16, 16, 16), device="cpu")
    baseline = ContinuousMRIModel(CachedEncoder(cached.manifest), "b", freeze_encoder=True).eval()
    history = ContinuousMRIModel(CachedEncoder(cached.manifest), "history", freeze_encoder=True, dropout=0.).eval()
    identity = model_identity(history, cached.records, config)
    checkpoint = {"variant": "b", "identity": identity, "model": baseline.state_dict()}
    transfer_baseline(history, checkpoint, identity)
    records = at_landmark(cached.records, 61., "train")
    a = next(iter(_loader(records, config, "b", 7)))
    b = next(iter(_loader(records, config, "history", 7)))
    with torch.no_grad():
        history.alpha.zero_()
        torch.testing.assert_close(history(b), baseline(a))
        history.alpha.fill_(.01)
    check = transfer_prediction_check(history, [b], "cpu", config.horizons)
    assert check["passed"] and check["observations"] == 2
    with pytest.raises(ValueError, match="identical"):
        transfer_baseline(history, checkpoint, {**identity, "encoder_sha256": "other"})
    with pytest.raises(ValueError, match="architecture"):
        history.load_state_dict({k: v for k, v in history.state_dict().items() if k != "_architecture_version"})


def test_optimizer_diagnostics_preserve_updates_and_reconstruct_nll():
    torch.manual_seed(3)
    model = ContinuousMRIModel(TinyEncoder(), "history", freeze_encoder=True, dropout=.1)
    batch = {"embeddings": torch.randn(2, 2, 8), "times": torch.tensor([[0., 6.], [0., 12.]]),
             "observed": torch.ones(2, 2, dtype=torch.bool), "entry": torch.tensor([6., 12.]),
             "outcome": torch.tensor([30., 60.]), "event": torch.tensor([1., 0.]), "records": [None, None]}
    models = [copy.deepcopy(model), copy.deepcopy(model)]
    losses, reports = [], []
    for enabled, candidate in zip((False, True), models, strict=True):
        torch.manual_seed(99)
        optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=1e-3)
        losses.append(train_epoch(candidate, [batch], optimizer, "cpu", 5., diagnostics=reports if enabled else None))
    assert losses[0] == losses[1]
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
    report = reports[0]
    assert report["nll_cumulative_mean"] + report["nll_event_log_mean"] == pytest.approx(losses[1])
    assert report["events"] == 1 and report["event_proportion"] == .5
    assert report["gradient_norms"]["attention"] > 0 and report["update_norms"]["survival_head"] > 0
    assert "hidden_preactivation" in report["activations"]
    assert all(not m._forward_hooks for m in models[1].modules())


def test_cached_multivisit_diagnostics_use_common_times_and_restore_state(cached):
    model = ContinuousMRIModel(CachedEncoder(cached.manifest), "history", freeze_encoder=True)
    panel = select_diagnostic_panel(cached.records, 42, latest=True)
    assert all(len(r.image_ids) > 1 for r in panel)
    rng, modes = torch.get_rng_state().clone(), [m.training for m in model.modules()]
    report = trace_diagnostic_panel(model, panel, spatial_size=(16, 16, 16), device="cpu",
                                   horizons=(12., 24., 60.), cache_path=cached.path)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert len(report["risk_variance_by_horizon"]) == 3
    assert "cumulative_hazard" in report["panel_variation"]
    assert "temporal_features" in report["panel_variation"]
    json.dumps(report, allow_nan=False)


def test_step_ahead_pairs_next_observed_visit_and_detaches_target():
    model = ContinuousMRIModel(TinyEncoder(), "history", freeze_encoder=True, dropout=0., step_ahead=True).eval()
    raw = torch.randn(1, 4, 8, requires_grad=True)
    batch = {"embeddings": raw, "times": torch.tensor([[0., -99., 6., 24.]]),
             "observed": torch.tensor([[True, False, True, True]])}
    seen = []
    handle = model.step_predictor.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    loss = model.auxiliary_loss(batch)
    handle.remove()
    torch.testing.assert_close(seen[0][:, -1], bounded_time(torch.tensor([6., 18.])))
    loss.backward()
    # Last embedding is only an auxiliary target; causality blocks influence on earlier source states.
    assert raw.grad[0, -1].abs().sum() == 0
    assert raw.grad[0, 0].abs().sum() > 0
    assert raw.grad[0, 1].abs().sum() == 0
    single = {"embeddings": raw.detach()[:, :1], "times": torch.zeros(1, 1), "observed": torch.ones(1, 1, dtype=torch.bool)}
    zero = model.auxiliary_loss(single)
    assert zero.item() == 0 and zero.requires_grad
    zero.backward()


def test_final_block_only_changes_and_cached_training_is_rejected():
    class Encoder(nn.Module):
        n_features = 8

        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({"stem": nn.Linear(8, 8),
                                        "layer4": nn.Sequential(nn.BatchNorm1d(8), nn.Linear(8, 8))})

        def forward(self, x):
            return self.model["layer4"](self.model["stem"](x))

    # ModuleDict exposes its registered layer4 as an attribute, like ResNet.
    model = ContinuousMRIModel(Encoder(), "history", freeze_encoder=True, dropout=0.)
    configure_final_block(model)
    model.train()
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    groups = optimizer_groups(model, 1e-4, 1e-5)
    assert [g["lr"] for g in groups] == [1e-4, 1e-5]
    optimizer = torch.optim.AdamW(groups)
    model.encoder(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    assert not model.encoder.model.layer4[0].training
    assert any(not torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items() if "layer4" in k)
    assert all(torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items() if "stem" in k or "running_" in k)
    with pytest.raises(ValueError, match="Cached inputs"):
        SurvivalConfig(variants=("history",), cache_path="cache.pt", baseline_init="history.pt",
                       fine_tune_final_block=True).validate()


def test_debug_gate_rejects_stale_code_wrong_cohort_and_false_success(tmp_path):
    records, frame, schedule, splits = fixture_data(tmp_path)
    config = synthetic_gate(SurvivalConfig(spatial_size=(16, 16, 16)), records, frame, schedule, splits, tmp_path)
    protocol, gate = read_json(config.protocol_path), read_json(config.debug_gate)
    identity = {"encoder_sha256": gate["encoder_sha256"]}
    assert require_debug_gate(config.debug_gate, protocol, identity, records=records)["passed"]
    for modification in ("code", "metric", "cohort", "initialization"):
        broken = copy.deepcopy(gate)
        if modification == "code": broken["implementation_hashes"] = {}
        if modification == "metric": broken["arms"]["a"]["final"]["c_index"] = .5
        if modification == "cohort": broken["full_cohort_sha256"] = "different"
        if modification == "initialization": broken["arms"]["history"]["initialization_check"]["passed"] = False
        write_json(config.debug_gate, broken)
        with pytest.raises(ValueError):
            require_debug_gate(config.debug_gate, protocol, identity, records=records)


def test_baseline_gate_binds_promoted_checkpoint_files(tmp_path):
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH
    protocol = read_json(PROTOCOL_PATH)
    identity = {"encoder_sha256": "encoder", "cohort_sha256": "cohort"}
    arms = {}
    for arm in ("a", "b"):
        path = tmp_path / f"{arm}.pt"
        path.write_bytes(arm.encode())
        arms[arm] = {"passed": True, "stable": True, "validation_nll": 1.,
                     "checkpoint": str(path), "checkpoint_sha256": file_sha256(path)}
    report = {"passed": True, "protocol_sha256": fingerprint(protocol), "identity": identity,
              "implementation_hashes": implementation_hashes(), "arms": arms}
    path = tmp_path / "baseline_gate.json"
    write_json(path, report)
    require_baseline_gate(path, protocol, identity, tmp_path / "b.pt")
    (tmp_path / "b.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        require_baseline_gate(path, protocol, identity, tmp_path / "b.pt")


def test_existing_training_entry_point_requires_gate_before_writing(tmp_path, monkeypatch):
    import mstat_project.ml.survival as pipeline
    _, frame, schedule, splits = fixture_data(tmp_path)
    monkeypatch.setattr(pipeline, "load_source", lambda engine: (frame, schedule, splits))
    output = tmp_path / "runs"
    with pytest.raises(ValueError, match="--debug-gate"):
        pipeline.run(SurvivalConfig(tensor_dir=tmp_path, output_dir=output), engine=object())
    assert not output.exists()
    with pytest.raises(ValueError, match="validation-only"):
        SurvivalConfig(evaluate_test=True).validate()


def test_named_experiment_cli_writes_named_validation_artifacts(tmp_path, monkeypatch):
    import mstat_project.ml.survival as pipeline
    records, frame, schedule, splits = fixture_data(tmp_path)
    config = SurvivalConfig(variants=("a",), experiment_id="baseline-stable", epochs=1, patience=1,
        pretrained=False, device="cpu", spatial_size=(16, 16, 16), tensor_dir=tmp_path, output_dir=tmp_path / "runs")
    config = synthetic_gate(config, records, frame, schedule, splits, tmp_path)
    monkeypatch.setattr(pipeline, "load_source", lambda engine: (frame, schedule, splits))
    monkeypatch.setattr(pipeline, "_persist_run", lambda *args: None)
    original = pipeline.train_variant
    monkeypatch.setattr(pipeline, "train_variant", lambda c, r, s, v, o: original(c, r, s, v, o, encoder_factory=TinyEncoder))
    path = pipeline.run(config, engine=object())
    assert (path / "baseline-stable" / "validation_predictions.json").exists()
    assert not (path / "a").exists()
    assert not list(path.rglob("test_predictions.json"))


def test_epoch_zero_checkpoint_remains_selectable(cached, tmp_path, monkeypatch):
    import mstat_project.ml.survival as pipeline
    config = SurvivalConfig(variants=("b",), cache_path=cached.path, epochs=1, patience=1, device="cpu",
                            spatial_size=(16, 16, 16))
    train_variant(config, cached.records, cached.schedule, "b", tmp_path / "b")
    config = replace(config, variants=("history",), baseline_init=tmp_path / "b" / "best.pt")
    values = iter([1., 2.])
    monkeypatch.setattr(pipeline, "validation_loss", lambda *args: next(values))
    result = train_variant(config, cached.records, cached.schedule, "history", tmp_path / "history")
    checkpoint = torch.load(tmp_path / "history" / "best.pt", weights_only=False)
    assert result["best_epoch"] == checkpoint["epoch"] == 0
    assert result["validation_nll"] == 1.


def test_debug_subset_is_balanced_deterministic_and_training_only(tmp_path):
    from mstat_project.ml.ltsa_experiments import select_debug_patients
    from mstat_project.ml.survival_data import Observation
    records = []
    for i in range(24):
        records.append(Observation(ptid=f"p{i:02d}", split="train" if i < 20 else "test",
            image_ids=(f"{i}-0", f"{i}-24"), acquisition_months=(0., 24.), visit_nums=(0, 3),
            prediction_months=24., outcome_months=60.+i, event=i % 2 == 0, tensor_path=tmp_path / f"{i}.pt"))
    chosen = select_debug_patients(records)
    assert chosen == select_debug_patients(list(reversed(records)))
    assert len(chosen) == len({r.ptid for r in chosen}) == 16
    assert sum(r.event for r in chosen) == 8
    assert all(r.split == "train" and r.visit_nums == (0, 3) for r in chosen)


def test_debug_failure_is_saved_and_never_trains_later_arms(cached, tmp_path, monkeypatch):
    import mstat_project.ml.ltsa_experiments as experiments
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH
    subset = [r for r in cached.records if r.split == "train" and len(r.image_ids) == 5]
    monkeypatch.setattr(experiments, "select_debug_patients", lambda records, seed: subset)
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate nonfinite loss")
    monkeypatch.setattr(experiments, "train_epoch", fail)
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    result = experiments.run_debug(config, cached.records, tmp_path / "debug", read_json(PROTOCOL_PATH), max_updates=1)
    assert not result["passed"]
    assert set(result["arms"]) == {"a"}
    assert "deliberate nonfinite" in result["arms"]["a"]["error"]
    assert read_json(tmp_path / "debug" / "gate.json") == json.loads(json.dumps(result, default=str))
    assert not (tmp_path / "debug" / "b.pt").exists()


def test_debug_checkpoint_io_failure_is_recorded_without_passing_gate(cached, tmp_path, monkeypatch):
    from mstat_project.ml import ltsa_experiments as experiments
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH

    subset = at_landmark(cached.records, 61., "train")
    monkeypatch.setattr(experiments, "select_debug_patients", lambda *args: subset)

    def fail(*args, **kwargs):
        raise PermissionError("persistent checkpoint lock; temporary retained")

    monkeypatch.setattr(experiments, "_save_checkpoint", fail)
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    output = tmp_path / "debug_io_failure"
    report = experiments.run_debug(config, cached.records, output, read_json(PROTOCOL_PATH), max_updates=1)
    assert not report["passed"] and set(report["arms"]) == {"a"}
    assert "PermissionError" in report["arms"]["a"]["error"]
    assert read_json(output / "gate.json")["arms"]["a"]["error"] == report["arms"]["a"]["error"]
    assert (output / "a_trace.json").is_file()


@pytest.mark.parametrize("variant", ["a", "b", "history"])
def test_reusing_experiment_rejects_changed_training_config(cached, tmp_path, variant):
    from mstat_project.ml.ltsa_experiments import train_if_missing
    # Production starts with the default A/B configuration and dispatches one
    # named experiment at a time. Each saved arm must receive a valid config.
    config = SurvivalConfig(cache_path=cached.path, epochs=1, patience=1,
                            device="cpu", spatial_size=(16, 16, 16))
    path = tmp_path / f"named-{variant}"
    first = train_if_missing(config, cached.records, cached.schedule, variant, path)
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["variants"] == (variant,)
    assert checkpoint["config"]["experiment_id"] == path.name
    assert checkpoint["variant"] == variant
    assert config.variants == ("a", "b") and config.experiment_id is None
    assert train_if_missing(config, cached.records, cached.schedule, variant, path) == json.loads(json.dumps(first))
    with pytest.raises(ValueError, match="weight_decay"):
        train_if_missing(replace(config, weight_decay=.5), cached.records, cached.schedule, variant, path)


def test_explicit_report_reads_test_only_after_artifacts_are_locked(cached, tmp_path, monkeypatch):
    import mstat_project.ml.ltsa_experiments as experiments
    config = SurvivalConfig(cache_path=cached.path, epochs=1, patience=1, device="cpu", spatial_size=(16, 16, 16))
    root = tmp_path / "workspace"
    seed_root = root / "seed_42"
    for name, variant in (("a", "a"), ("b", "b"), ("history-1l", "history")):
        train_variant(config, cached.records, cached.schedule, variant, seed_root / name)
    assert not list(root.rglob("*_test_predictions.json"))
    monkeypatch.setattr(experiments, "stable_run", lambda path: True)
    original_loader = experiments._loader
    observed_splits = []
    def report_loader(records, *args):
        assert (root / "final_report" / "locked_experiments.json").is_file()
        observed_splits.extend(r.split for r in records)
        return original_loader(records, *args)
    monkeypatch.setattr(experiments, "_loader", report_loader)
    output = experiments.final_report(root, seed_root, config, cached.records, cached.schedule,
                                       {"snapshot_hashes": cached.provenance})
    assert set(observed_splits) == {"test"}
    assert (output / "report.md").is_file()
    assert len(list(output.glob("*_test_predictions.json"))) == 3
    assert any(row["primary"] for row in read_json(output / "comparisons.json"))
    manifest = read_json(output / "locked_experiments.json")
    assert all(item["checkpoint_sha256"] for item in manifest["experiments"].values())


@pytest.fixture
def reference_snapshot(tmp_path):
    from mstat_project.data_prep.data_splitting import frame_fingerprint

    records, frame, schedule, splits = fixture_data(tmp_path)
    schedule = schedule.with_columns(
        pl.col(name).cast(pl.Decimal(precision=10, scale=2)) for name in ("follow_up_start", "follow_up_end")
    )
    frame.write_parquet(tmp_path / "source.parquet")
    schedule.write_csv(tmp_path / "schedule.csv")
    splits.write_csv(tmp_path / "patient_splits.csv")
    checkpoints = {}
    for arm in ("a", "b"):
        path = tmp_path / f"reference_{arm}.pt"
        path.write_bytes(f"synthetic reference {arm}".encode())
        checkpoints[arm] = {"checkpoint": str(path), "checkpoint_sha256": file_sha256(path)}
    protocol = {"reference_directory": str(tmp_path), "reference_baselines": checkpoints,
                "reference_config": {"tensor_dir": str(tmp_path)},
                "snapshot_hashes": {"source": frame_fingerprint(frame), "schedule": frame_fingerprint(schedule),
                                    "splits": frame_fingerprint(splits)}}
    return SimpleNamespace(protocol=protocol, records=records, frame=frame, schedule=schedule, splits=splits,
                           directory=tmp_path)


def test_reference_snapshot_preserves_decimal_schedule_fingerprint(reference_snapshot):
    from mstat_project.data_prep.data_splitting import frame_fingerprint
    from mstat_project.ml.ltsa_experiments import load_snapshot

    snapshot = reference_snapshot
    # Reproduce the old reader's loss of formatting before checking the fixed path.
    inferred = pl.read_csv(snapshot.directory / "schedule.csv")
    assert inferred["follow_up_start"].dtype == pl.Float64
    assert frame_fingerprint(inferred) != snapshot.protocol["snapshot_hashes"]["schedule"]
    records, schedule, audit = load_snapshot(snapshot.protocol)
    assert frame_fingerprint(schedule) == snapshot.protocol["snapshot_hashes"]["schedule"]
    assert schedule["follow_up_start"].dtype == pl.Decimal(precision=10, scale=2)
    assert records == snapshot.records
    assert audit["eligible_patients"] == 6


@pytest.mark.parametrize("component", ["source", "schedule", "splits"])
def test_reference_snapshot_still_rejects_changed_data(reference_snapshot, component):
    from mstat_project.ml.ltsa_experiments import load_snapshot

    snapshot = reference_snapshot
    if component == "source":
        snapshot.frame.with_columns((pl.col("outcome_months") + 1).alias("outcome_months")).write_parquet(
            snapshot.directory / "source.parquet")
    elif component == "schedule":
        path = snapshot.directory / "schedule.csv"
        path.write_text(path.read_text().replace("9.00", "8.00"), encoding="utf-8")
    else:
        path = snapshot.directory / "patient_splits.csv"
        path.write_text(path.read_text().replace("train", "test", 1), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"fingerprints changed: {component}:"):
        load_snapshot(snapshot.protocol)


def test_reference_snapshot_rejects_precision_loss_instead_of_rounding(reference_snapshot):
    from mstat_project.ml.ltsa_experiments import load_snapshot

    path = reference_snapshot.directory / "schedule.csv"
    path.write_text(path.read_text().replace("9.00", "9.001"), encoding="utf-8")
    with pytest.raises(ValueError, match="two-decimal boundaries"):
        load_snapshot(reference_snapshot.protocol)


def test_reference_snapshot_still_rejects_changed_checkpoint(reference_snapshot):
    from mstat_project.ml.ltsa_experiments import load_snapshot

    (reference_snapshot.directory / "reference_a.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Reference a checkpoint fingerprint changed"):
        load_snapshot(reference_snapshot.protocol)


def test_debug_risk_inspection_detects_probability_rounding():
    from mstat_project.ml.ltsa_debug_inspection import risk_diagnostics

    result = risk_diagnostics(torch.tensor([30., 25., 20.]), torch.tensor([1., 2., 3.]), torch.ones(3))
    assert result["probabilities_equal_one"] == 3
    scores = result["full_followup"]
    assert scores["probability_c_index"] == .5
    assert scores["probability_float64_c_index"] == scores["cumulative_hazard_c_index"] == 1.
    assert scores["pairs_tied_only_after_probability_conversion"] == scores["comparable_pairs"] == 3
    assert scores["cumulative_hazard_tied_pairs"] == 0
    small = risk_diagnostics(torch.tensor([1e-9, 2e-9]), torch.tensor([1., 2.]), torch.ones(2))
    assert small["probabilities_equal_zero"] == 2
    assert small["full_followup"]["probability_c_index"] == .5
    assert small["full_followup"]["probability_float64_c_index"] == 0.


def test_debug_risk_inspection_separates_horizon_and_full_followup():
    from mstat_project.ml.ltsa_debug_inspection import risk_diagnostics

    result = risk_diagnostics(torch.tensor([1., 3., 2.]), torch.tensor([10., 70., 80.]), torch.ones(3))
    assert result["full_followup"]["comparable_pairs"] == 3
    assert result["full_followup"]["cumulative_hazard_c_index"] == pytest.approx(1/3)
    assert result["horizon_truncated"]["comparable_pairs"] == 2
    assert result["horizon_truncated"]["cumulative_hazard_c_index"] == 0.
    empty_pairs = risk_diagnostics(torch.ones(2), torch.tensor([1., 2.]), torch.zeros(2))
    assert empty_pairs["full_followup"]["cumulative_hazard_c_index"] is None
    json.dumps(empty_pairs, allow_nan=False)
    with pytest.raises(ValueError, match="finite"):
        risk_diagnostics(torch.tensor([float("nan")]), torch.ones(1), torch.ones(1))


def test_debug_prediction_inspection_matches_gate_and_restores_state(cached):
    from mstat_project.ml.ltsa_debug_inspection import inspect_predictions
    from mstat_project.ml.ltsa_experiments import debug_metrics

    records = at_landmark(cached.records, 61., "train")
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    model = ContinuousMRIModel(CachedEncoder(cached.manifest), "history", freeze_encoder=True)
    loader = _loader(records, config, "history", 7)
    expected = debug_metrics(model, loader, "cpu")
    model.train()
    flags = [module.training for module in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    result = inspect_predictions(model, loader, "cpu")
    assert result["nll"] == pytest.approx(expected["nll"])
    assert result["full_followup"]["cumulative_hazard_c_index"] == expected["c_index"]
    assert result["full_followup"]["probability_c_index"] == expected["probability_c_index"]
    assert result["nll"] == pytest.approx(result["nll_cumulative_mean"] + result["nll_event_log_mean"])
    assert result["values"]["patients"] == [r.ptid for r in records]
    assert [module.training for module in model.modules()] == flags
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name])


def test_debug_inspection_preserves_failed_gate_and_checks_identity(cached, monkeypatch):
    import shutil
    from mstat_project.ml import ltsa_debug_inspection as inspection
    from mstat_project.ml.survival_protocol import cohort_fingerprint

    workspace = cached.path.parent / "inspection_workspace"
    directory = workspace / "debug"
    directory.mkdir(parents=True)
    cache_path = workspace / "debug_embeddings.pt"
    shutil.copyfile(cached.path, cache_path)
    config = SurvivalConfig(cache_path=cache_path, device="cpu", spatial_size=(16, 16, 16), dropout=0.)
    subset = at_landmark(cached.records, 61., "train")
    monkeypatch.setattr(inspection, "select_debug_patients", lambda records, seed: subset)
    protocol = {"snapshot_hashes": cached.provenance, "debug": {"minimum_nll_drop": .5, "minimum_c_index": .95}}
    model = ContinuousMRIModel(CachedEncoder(cached.manifest), "history", freeze_encoder=True, dropout=0.)
    checkpoint = {"model": model.state_dict(), "variant": "history", "config": asdict(config), "update": 2000,
                  "identity": model_identity(model, subset, config)}
    torch.save(checkpoint, directory / "history.pt")
    gate = {"passed": False, "protocol_sha256": fingerprint(protocol), "snapshot_hashes": cached.provenance,
            "full_cohort_sha256": cohort_fingerprint(cached.records), "config": asdict(config),
            "encoder_sha256": cached.manifest["encoder_sha256"], "implementation_hashes": implementation_hashes(),
            "patients": [r.ptid for r in subset], "image_ids": [r.image_ids[-1] for r in subset],
            "arms": {"history": {"passed": False, "initial": {"nll": 1.5}}}}
    write_json(directory / "gate.json", gate)
    before = {p.name: file_sha256(p) for p in directory.iterdir()}
    result = inspection.inspect_debug_run(workspace, cached.records, protocol, "cpu")
    assert result["diagnostic_only"] and result["implementation_matches_training"]
    assert not result["saved_gate_passed"]
    assert result["arms"]["history"]["required_nll"] == 1.
    assert {p.name: file_sha256(p) for p in directory.iterdir()} == before
    checkpoint["identity"]["encoder_sha256"] = "wrong encoder"
    torch.save(checkpoint, directory / "history.pt")
    with pytest.raises(ValueError, match="identity"):
        inspection.inspect_debug_run(workspace, cached.records, protocol, "cpu")


def test_debug_gate_ranks_hazards_without_probability_ties():
    from mstat_project.ml.ltsa_experiments import debug_metrics

    head = SimpleNamespace(nll=lambda *args: torch.ones(3),
                           cumulative_hazard=lambda *args: torch.tensor([30., 25., 20.]))
    model = SimpleNamespace(head=head, eval=lambda: None, encode_batch=lambda batch: torch.zeros(3, 1))
    batch = {"entry": torch.zeros(3), "outcome": torch.tensor([1., 2., 3.]), "event": torch.ones(3)}
    result = debug_metrics(model, [batch], "cpu")
    assert result["c_index"] == 1.
    assert result["probability_c_index"] == .5
    assert result["probabilities_equal_one"] == 3
    # Ranking correction does not silently relax the separate variation gate.
    assert result["risk_variance"] == 0.


@pytest.mark.parametrize("stage", ["debug", "cache-full", "baselines"])
def test_debug_named_attempt_routes_gate_and_preserves_original(cached, monkeypatch, stage):
    from mstat_project.ml import ltsa_experiments as experiments

    workspace = cached.path.parent / "named_attempt"
    original = workspace / "debug"
    original.mkdir(parents=True)
    (original / "gate.json").write_text('{"passed": false}', encoding="utf-8")
    before = file_sha256(original / "gate.json")
    selected = workspace / "debug_risk_v2"
    monkeypatch.setattr(experiments, "load_snapshot", lambda *args: (cached.records, cached.schedule, {}))
    monkeypatch.setattr(experiments, "validation_support", lambda records: {"confirmatory_support": True})
    monkeypatch.setattr(experiments, "load_embeddings", lambda *args, **kwargs: ({"manifest": cached.manifest}, {}))

    def run(config, records, output, protocol):
        assert config.cache_path == workspace / "debug_embeddings.pt"
        assert output == selected and config.debug_gate == selected / "gate.json"
        return {"passed": False}

    def require(path, *args, **kwargs):
        assert path == selected / "gate.json"
        raise ValueError("selected gate reached")

    monkeypatch.setattr(experiments, "run_debug", run)
    monkeypatch.setattr(experiments, "require_debug_gate", require)
    error, match = (RuntimeError, "debug_risk_v2") if stage == "debug" else (ValueError, "selected gate reached")
    with pytest.raises(error, match=match):
        experiments.main([stage, "--workspace", str(workspace), "--debug-run", "debug_risk_v2", "--device", "cpu"])
    assert file_sha256(original / "gate.json") == before


@pytest.mark.parametrize("name", ["../debug", "debug/other", "seed_42", "debug_..", "debug_foo/bar"])
def test_debug_attempt_rejects_unsafe_or_conflicting_names(name):
    import argparse
    from mstat_project.ml.ltsa_experiments import debug_run_name

    with pytest.raises(argparse.ArgumentTypeError):
        debug_run_name(name)


def test_debug_failure_keeps_checkpoint_with_passing_ranking(cached, monkeypatch):
    from mstat_project.ml import ltsa_experiments as experiments
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH

    subset = at_landmark(cached.records, 61., "train")
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    monkeypatch.setattr(experiments, "select_debug_patients", lambda *args: subset)
    monkeypatch.setattr(experiments, "train_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiments, "transfer_prediction_check", lambda *args: {"passed": True})
    # A/B pass immediately; history later improves NLL but loses ranking.
    values = iter([(2., 1.), (1.4, 1.), (2., 1.), (1.4, 1.),
                   (1.4, .98)] + [(1.2, .98)] * 24 + [(.5, .8)])

    def metrics(*args):
        nll, c_index = next(values)
        return {"nll": nll, "c_index": c_index, "probability_c_index": c_index, "risk_variance": .1}

    monkeypatch.setattr(experiments, "debug_metrics", metrics)
    output = cached.path.parent / "debug_preserve_rank"
    report = experiments.run_debug(config, cached.records, output, read_json(PROTOCOL_PATH), max_updates=25)
    assert not report["passed"] and not report["arms"]["history"]["passed"]
    assert report["c_index_risk_score"] == "cumulative_hazard_60_months"
    best = torch.load(output / "history_best_rank.pt", weights_only=False)
    last = torch.load(output / "history.pt", weights_only=False)
    assert best["update"] == 1 and last["update"] == 25
    assert best["metrics"]["nll"] == 1.2
    assert report["arms"]["history"]["best_rank"]["checkpoint"] == "history_best_rank.pt"
    # A retry cannot overwrite an existing attempt.
    with pytest.raises(FileExistsError):
        experiments.run_debug(config, cached.records, output, read_json(PROTOCOL_PATH), max_updates=25)


def test_debug_catches_history_pass_between_diagnostics_and_preserves_baseline_schedule(cached, monkeypatch):
    from mstat_project.ml import ltsa_experiments as experiments
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH

    subset = at_landmark(cached.records, 61., "train")
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    monkeypatch.setattr(experiments, "select_debug_patients", lambda *args: subset)
    monkeypatch.setattr(experiments, "transfer_prediction_check", lambda *args: {"passed": True})
    counts = {arm: 0 for arm in ("a", "b", "history")}

    def train(model, *args, **kwargs):
        counts[model.variant] += 1
        if model.variant == "history" and counts["history"] == 2:
            assert kwargs["diagnostics"] is None

    def metrics(model, *args):
        update = counts[model.variant]
        if model.variant == "history":
            nll = {0: 1.4, 1: 1.2, 2: .8}[update]
        else:
            # A/B retain their original checks and selected initialization.
            assert update in (0, 1, 25)
            nll = {0: 2., 1: 1.8, 25: 1.4}[update]
        return {"nll": nll, "c_index": .98, "probability_c_index": .98, "risk_variance": .1}

    monkeypatch.setattr(experiments, "train_epoch", train)
    monkeypatch.setattr(experiments, "debug_metrics", metrics)
    output = cached.path.parent / "debug_dense_checks"
    report = experiments.run_debug(config, cached.records, output, read_json(PROTOCOL_PATH), max_updates=25)
    assert report["passed"] and counts == {"a": 25, "b": 25, "history": 2}
    assert report["gate_evaluation"]["history"] == "every update"
    trace = read_json(output / "history_trace.json")
    assert [row["update"] for row in trace] == [1, 2]
    assert trace[-1]["optimization"] == []
    checkpoint = torch.load(output / "history.pt", weights_only=False)
    assert checkpoint["update"] == 2


def test_extra_history_gate_checks_restore_loader_rng(cached, monkeypatch):
    from mstat_project.ml import ltsa_experiments as experiments
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH

    subset = at_landmark(cached.records, 61., "train")
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16))
    monkeypatch.setattr(experiments, "select_debug_patients", lambda *args: subset)
    monkeypatch.setattr(experiments, "transfer_prediction_check", lambda *args: {"passed": True})
    counts = {arm: 0 for arm in ("a", "b", "history")}
    before_extra = []

    def train(model, *args, **kwargs):
        counts[model.variant] += 1
        if model.variant == "history" and counts["history"] == 2:
            before_extra.append(torch.get_rng_state().clone())
        if model.variant == "history" and counts["history"] == 3:
            torch.testing.assert_close(torch.get_rng_state(), before_extra[0])

    def metrics(model, *args):
        torch.rand(1)  # Simulate the DataLoader iterator's CPU seed draw.
        update = counts[model.variant]
        nll = (2. if update == 0 else 1.4) if model.variant != "history" else 1.4
        return {"nll": nll, "c_index": .98, "probability_c_index": .98, "risk_variance": .1}

    monkeypatch.setattr(experiments, "train_epoch", train)
    monkeypatch.setattr(experiments, "debug_metrics", metrics)
    report = experiments.run_debug(config, cached.records, cached.path.parent / "debug_rng",
                                   read_json(PROTOCOL_PATH), max_updates=3)
    assert not report["passed"] and counts["history"] == 3 and len(before_extra) == 1


@pytest.fixture
def continuation_source(cached, monkeypatch):
    """Synthetic saved run for continuation mechanics, not empirical gate evidence."""
    import shutil
    from mstat_project.ml import ltsa_debug_control as control, ltsa_debug_inspection as inspection
    from mstat_project.ml.ltsa_experiments import debug_metrics
    from mstat_project.ml.survival_protocol import cohort_fingerprint

    workspace = cached.path.parent / "control_workspace"
    source = workspace / "debug_control_source"
    source.mkdir(parents=True)
    cache_path = workspace / "debug_embeddings.pt"
    shutil.copyfile(cached.path, cache_path)
    subset = at_landmark(cached.records, 61., "train")
    monkeypatch.setattr(control, "select_debug_patients", lambda *args: subset)
    monkeypatch.setattr(inspection, "select_debug_patients", lambda *args: subset)
    config = SurvivalConfig(cache_path=cache_path, device="cpu", spatial_size=(16, 16, 16),
                            batch_size=16, learning_rate=.001, weight_decay=0., dropout=0.)
    protocol = {"snapshot_hashes": cached.provenance,
                "debug": {"max_updates": 2000, "minimum_nll_drop": .5, "minimum_c_index": .95}}
    arms = {}
    for variant in ("b", "history"):
        model = ContinuousMRIModel(CachedEncoder(cached.manifest), variant, freeze_encoder=True, dropout=0.)
        metrics = debug_metrics(model, _loader(subset, config, variant, 1), "cpu")
        torch.save({"model": model.state_dict(), "config": asdict(config), "variant": variant,
                    "identity": model_identity(model, subset, config), "update": 25}, source / f"{variant}.pt")
        arms[variant] = {"passed": variant == "b", "initial": metrics, "final": metrics}
    write_json(source / "history_trace.json", [{"update": i, **arms["history"]["final"]} for i in (1, 2)])
    gate = {"passed": False, "protocol_sha256": fingerprint(protocol), "snapshot_hashes": cached.provenance,
            "full_cohort_sha256": cohort_fingerprint(cached.records), "config": asdict(config),
            "encoder_sha256": cached.manifest["encoder_sha256"], "implementation_hashes": implementation_hashes(),
            "patients": [r.ptid for r in subset], "image_ids": [r.image_ids[-1] for r in subset],
            "c_index_risk_score": "cumulative_hazard_60_months", "arms": arms}
    write_json(source / "gate.json", gate)
    return SimpleNamespace(workspace=workspace, source=source, protocol=protocol,
                           records=cached.records, subset=subset, gate=gate)


def test_b_continuation_uses_saved_weights_fresh_optimizer_and_preserves_source(continuation_source, monkeypatch):
    from mstat_project.ml import ltsa_debug_control as control

    case = continuation_source
    before = {p.name: file_sha256(p) for p in case.source.iterdir() if p.is_file()}
    checkpoint = torch.load(case.source / "b.pt", weights_only=False)
    original_train = control.train_epoch
    calls = []

    def train(model, loader, optimizer, *args, **kwargs):
        if not calls:
            assert not optimizer.state
            assert optimizer.param_groups[0]["lr"] == .001 and optimizer.param_groups[0]["weight_decay"] == 0.
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), checkpoint["model"][name])
        assert model.variant == "b" and model.transformer is None
        assert loader.dataset.records == case.subset
        calls.append(True)
        return original_train(model, loader, optimizer, *args, **kwargs)

    monkeypatch.setattr(control, "train_epoch", train)
    output = control.run_b_continuation(case.workspace, case.records, case.protocol, "cpu",
                                       debug_run=case.source.name, max_updates=2)
    report = read_json(output / "report.json")
    assert report["status"] == "completed" and report["diagnostic_only"] and "passed" not in report
    assert report["initial"]["nll"] == pytest.approx(case.gate["arms"]["b"]["final"]["nll"])
    assert len(calls) == 2 and report["final"]["update"] == 2
    trace = read_json(output / "trace.json")
    assert trace[0]["history"]["nll"] == case.gate["arms"]["history"]["final"]["nll"]
    last = torch.load(output / "last.pt", weights_only=False)
    assert last["diagnostic_only"] and last["initialization_sha256"] == before["b.pt"]
    assert not torch.equal(last["model"]["projection.weight"], checkpoint["model"]["projection.weight"])
    assert {p.name: file_sha256(p) for p in case.source.iterdir() if p.is_file()} == before


def test_b_continuation_rejects_changed_training_code(continuation_source):
    from mstat_project.ml.ltsa_debug_control import run_b_continuation

    case = continuation_source
    case.gate["implementation_hashes"] = {"changed": "code"}
    write_json(case.source / "gate.json", case.gate)
    with pytest.raises(ValueError, match="same training implementation"):
        run_b_continuation(case.workspace, case.records, case.protocol, "cpu",
                           debug_run=case.source.name, max_updates=2)
    assert not list(case.source.glob("b_continuation_*"))


def test_b_continuation_records_joint_criteria_without_stopping_or_promoting(continuation_source, monkeypatch):
    from mstat_project.ml import ltsa_debug_control as control

    case = continuation_source
    initial = case.gate["arms"]["b"]["final"]
    improved = {**initial, "nll": initial["nll"] - 1., "c_index": 1., "risk_variance": .1}
    values = iter([initial, improved, improved])
    monkeypatch.setattr(control, "debug_metrics", lambda *args: next(values))
    monkeypatch.setattr(control, "train_epoch", lambda *args, **kwargs: None)
    output = control.run_b_continuation(case.workspace, case.records, case.protocol, "cpu",
                                       debug_run=case.source.name, max_updates=2)
    report = read_json(output / "report.json")
    assert report["criteria_met_at"] == 1 and report["final"]["update"] == 2
    assert (output / "first_joint_criteria.pt").is_file()
    assert not (output / "gate.json").exists()
    assert not read_json(case.source / "gate.json")["passed"]


def test_patience_investigation_isolated_matched_and_gated(cached, monkeypatch):
    from mstat_project.ml import ltsa_baseline_investigation as investigation
    from mstat_project.ml.survival_protocol import PROTOCOL_PATH, cohort_fingerprint

    workspace = cached.path.parent
    source = workspace / "seed_42"
    protocol = read_json(PROTOCOL_PATH)
    protocol["snapshot_hashes"] = cached.provenance
    protocol["reference_config"]["spatial_size"] = [16, 16, 16]
    identity = {"cohort_sha256": cohort_fingerprint(cached.records),
                "encoder_sha256": cached.manifest["encoder_sha256"]}
    configs, arms = {}, {}
    for arm in ("a", "b"):
        path = source / arm / "best.pt"
        path.parent.mkdir(parents=True)
        config = SurvivalConfig(variants=(arm,), experiment_id=arm, cache_path=cached.path,
                                spatial_size=(16, 16, 16), device="cpu")
        configs[arm] = config
        torch.save({"variant": arm, "config": asdict(config), "identity": identity,
                    "implementation_hashes": implementation_hashes()}, path)
        arms[arm] = {"checkpoint_sha256": file_sha256(path), "validation_nll": 1.075}
    gate = {"passed": False, "identity": identity, "implementation_hashes": implementation_hashes(),
            "protocol_sha256": fingerprint(protocol), "arms": arms}
    write_json(source / "baseline_gate.json", gate)
    debug = workspace / "debug_source" / "gate.json"
    debug.parent.mkdir()
    write_json(debug, {"passed": True})
    originals = {p: file_sha256(p) for p in source.rglob("*") if p.is_file()}
    candidate = workspace / "investigations" / "patience_8_seed_42"

    def reject(*args, **kwargs):
        raise ValueError("debug gate rejected")

    monkeypatch.setattr(investigation, "require_debug_gate", reject)
    with pytest.raises(ValueError, match="debug gate rejected"):
        investigation.run_investigation(workspace, cached.records, cached.schedule, protocol,
                                        PROTOCOL_PATH, "cpu", "debug_source")
    assert not candidate.exists()
    checked, trained = [], []

    def require(path, passed_protocol, manifest, *, records):
        assert path == debug and passed_protocol == protocol and records == cached.records
        assert manifest["cache_id"] == cached.manifest["cache_id"]
        checked.append(True)

    def train(config, records, schedule, arm, path):
        config.validate()
        assert config.patience == 8 and config.baseline_init is None
        permitted_changes = {"patience", "debug_gate", "protocol_path", "output_dir", "cache_path", "device"}
        for key, value in asdict(configs[arm]).items():
            if key not in permitted_changes:
                assert asdict(config)[key] == value
        assert records == cached.records and path == candidate / arm
        path.mkdir()
        (path / "best.pt").write_bytes(b"synthetic candidate checkpoint")
        trained.append(arm)
        return {"validation_nll": 1., "best_epoch": 10}

    monkeypatch.setattr(investigation, "require_debug_gate", require)
    monkeypatch.setattr(investigation, "train_if_missing", train)
    monkeypatch.setattr(investigation, "stable_run", lambda path: True)
    result = investigation.run_investigation(workspace, cached.records, cached.schedule, protocol,
                                            PROTOCOL_PATH, "cpu", "debug_source")
    assert checked == [True] and trained == ["a", "b"] and result == candidate
    report = read_json(result / "report.json")
    assert not report["production_promoted"] and report["selection_split"] == "validation"
    assert all(row["meets_reference_nll"] for row in report["arms"].values())
    assert not (result / "baseline_gate.json").exists()
    assert all(file_sha256(path) == digest for path, digest in originals.items())
