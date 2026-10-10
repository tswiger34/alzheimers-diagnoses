"""Synthetic CPU checks for architecture isolation and patient-balanced attribution."""

import copy
import json
import math
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_ltsa_stabilization import cached  # noqa: F401 -- shared synthetic cache fixture
from mstat_project.ml import ltsa_architecture_ablation as ablation
from mstat_project.ml.continuous_survival import ContinuousMRIModel, initial_hazard
from mstat_project.ml.survival import SurvivalConfig, _loader, _seed, train_variant, validation_loss, write_json
from mstat_project.ml.survival_embeddings import CachedEncoder, state_fingerprint
from mstat_project.ml.survival_protocol import file_sha256, read_json


def config_for(cached, **kwargs):
    return SurvivalConfig(cache_path=cached.path, spatial_size=(16, 16, 16), device="cpu", **kwargs)


def test_ablation_catalog_separates_single_factor_reversals_from_joint_controls():
    control = asdict(ablation.STABILIZED)
    for name, architecture in ablation.EXPERIMENTS.items():
        differences = sum(value != control[key] for key, value in asdict(architecture).items())
        assert differences == {"stabilized": 0, "legacy_initialization": 2, "legacy_all": 7}.get(name, 1)


@pytest.mark.parametrize("arm", ["a", "b"])
def test_ablation_stabilized_matches_production_initial_weights_and_predictions(cached, arm):
    config = config_for(cached)
    rate = initial_hazard(cached.records)
    _seed(42)
    production = ContinuousMRIModel(CachedEncoder(cached.manifest), arm, freeze_encoder=True, baseline_hazard=rate)
    _seed(42)
    candidate = ablation.AblationModel(cached.manifest, arm, ablation.STABILIZED, rate)
    for name, value in production.state_dict().items():
        if name != "_architecture_version":
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
    batch = next(iter(_loader([r for r in cached.records if r.split == "validation"], config, arm, 1)))
    torch.testing.assert_close(production.encode_batch(batch), candidate.encode_batch(batch), rtol=0, atol=0)
    torch.testing.assert_close(production(batch), candidate(batch), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        production.load_state_dict(candidate.state_dict())
    ablation.load_saved_weights(candidate, {"variant": arm, "model": production.state_dict(),
        "identity": {"encoder_sha256": cached.manifest["encoder_sha256"]}}, "stabilized", cached.manifest)


@pytest.mark.parametrize("name", list(ablation.EXPERIMENTS))
def test_ablation_matches_shared_weights_and_changes_only_declared_initialization(cached, name):
    _seed(42)
    control = ablation.AblationModel(cached.manifest, "b", ablation.STABILIZED, .003)
    _seed(42)
    candidate = ablation.AblationModel(cached.manifest, "b", ablation.EXPERIMENTS[name], .003)
    spec = ablation.EXPERIMENTS[name]
    for key, value in candidate.state_dict().items():
        if key == "head.network.2.weight" and not spec.small_output_weights:
            assert not torch.equal(value, control.state_dict()[key])
        elif key == "head.network.2.bias" and not spec.calibrated_output_bias:
            assert not torch.equal(value, control.state_dict()[key])
        else:
            torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    assert isinstance(candidate.projection_norm, nn.LayerNorm) == spec.projection_norm
    assert isinstance(candidate.head.input_norm, nn.LayerNorm) == spec.head_norm
    assert isinstance(candidate.head.activation, nn.Tanh) == (spec.activation == "tanh")


def test_reference_adapter_matches_original_formula_and_rejects_encoder_mismatch(cached):
    _seed(17)
    model = ablation.AblationModel(cached.manifest, "b", ablation.EXPERIMENTS["legacy_all"], .003)
    # Independently reconstruct the original constructor's parameter order.
    _seed(17)
    projection = nn.Linear(cached.manifest["n_features"], 128)
    time_embedding = nn.Linear(1, 128)
    network = nn.Sequential(nn.Linear(129, 64), nn.Tanh(), nn.Linear(64, 1))
    for expected, actual in zip(network.parameters(), model.head.network.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state = {key: value for key, value in model.state_dict().items() if key != "_ablation_format_version"}
    state.update({f"encoder.{key}": value for key, value in cached.encoder.state_dict().items()})
    checkpoint = {"variant": "b", "model": state}
    assert ablation.encoder_state_hash(cached.encoder.state_dict()) == state_fingerprint(cached.encoder)
    ablation.load_saved_weights(model, checkpoint, "reference", cached.manifest)
    batch = next(iter(_loader([r for r in cached.records if r.split == "validation"], config_for(cached), "b", 1)))
    z = projection(batch["embeddings"][:, 0]) + time_embedding(batch["times"][:, :1] / 12)
    torch.testing.assert_close(model.encode_batch(batch), z)
    inputs = torch.cat((z, batch["outcome"][:, None] / 12), -1)
    expected_hazard = (F.softplus(network(inputs)).squeeze(-1) + 1e-7) / 12
    torch.testing.assert_close(model.head.hazard(z, batch["outcome"]).squeeze(-1), expected_hazard)
    with pytest.raises(ValueError, match="encoder weights"):
        ablation.load_saved_weights(model, checkpoint, "reference", {**cached.manifest, "encoder_sha256": "wrong"})
    with pytest.raises(ValueError, match="version 2"):
        ablation.load_saved_weights(model, checkpoint, "stabilized", cached.manifest)


@pytest.mark.parametrize("name", ["stabilized", "legacy_all", "legacy_hazard_time"])
def test_ablation_preserves_monthly_constant_hazard_integration(cached, name):
    model = ablation.AblationModel(cached.manifest, "b", ablation.EXPERIMENTS[name], .003)
    rate = .003
    transformed = 12 * rate - 1e-7
    with torch.no_grad():
        model.head.output_linear.weight.zero_()
        model.head.output_linear.bias.fill_(transformed + math.log(-math.expm1(-transformed)))
    z = torch.randn(3, 128)
    start, end = torch.tensor([0., 24., 240.]), torch.tensor([12., 84., 960.])
    cumulative, event_log = model.head.nll_components(z, start, end, torch.tensor([1., 0., 1.]))
    torch.testing.assert_close(cumulative, rate * (end - start))
    torch.testing.assert_close(event_log, -torch.tensor([1., 0., 1.]) * math.log(rate))


def component_rows():
    return [
        {"ptid": "event", "entry": 0., "outcome": 80., "event": True, "image_ids": ["e0"],
         "cumulative_hazard": 2., "event_log_hazard": 3., "nll": 5.},
        {"ptid": "event", "entry": 48., "outcome": 80., "event": True, "image_ids": ["e0", "e1"],
         "cumulative_hazard": 4., "event_log_hazard": 1., "nll": 5.},
        {"ptid": "censored", "entry": 72., "outcome": 90., "event": False, "image_ids": ["c0"],
         "cumulative_hazard": 10., "event_log_hazard": 0., "nll": 10.},
    ]


def test_components_weight_patients_and_reconstruct_all_stratum_contributions():
    rows = component_rows()
    result = ablation.aggregate_components(rows)
    assert result["overall"]["nll"] == 7.5  # Not the observation mean 20/3.
    assert result["overall"]["cumulative_hazard"] == 6.5
    assert result["overall"]["event_log_hazard"] == 1.
    for key in ("by_event", "by_entry_months"):
        for term in ablation.TERMS:
            assert sum(group["contribution_to_overall"][term] for group in result[key].values()) == result["overall"][term]
    with pytest.raises(ValueError, match="Duplicate"):
        ablation.aggregate_components(rows + [rows[0]])
    broken = copy.deepcopy(rows)
    broken[0]["nll"] += 1
    with pytest.raises(ValueError, match="reconstruct"):
        ablation.aggregate_components(broken)


def test_component_differences_are_paired_and_signed():
    rows = component_rows()
    modified = copy.deepcopy(rows)
    for row in modified:
        row["cumulative_hazard"] += .25
        row["event_log_hazard"] -= .1
        row["nll"] += .15
    left = {**ablation.aggregate_components(rows), "rows": rows}
    right = {**ablation.aggregate_components(modified), "rows": modified}
    result = ablation.paired_components(left, right)
    assert result["mean_difference"] == pytest.approx({"cumulative_hazard": .25, "event_log_hazard": -.1, "nll": .15})
    right["rows"][0]["image_ids"] = ["changed"]
    with pytest.raises(ValueError, match="identical validation histories"):
        ablation.paired_components(left, right)


def test_validation_component_evaluation_matches_loss_restores_state_and_rejects_test(cached):
    config = config_for(cached, batch_size=3)
    rows = [r for r in cached.records if r.split == "validation"][:-1]  # Unequal history counts.
    loader = _loader(rows, config, "b", 1)
    model = ablation.AblationModel(cached.manifest, "b", ablation.STABILIZED, initial_hazard(cached.records))
    expected = validation_loss(model, loader, "cpu")
    model.train()
    flags = [module.training for module in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    result = ablation.validation_components(model, loader, "cpu")
    assert result["overall"]["nll"] == pytest.approx(expected, abs=1e-6)
    assert [module.training for module in model.modules()] == flags
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    heldout = _loader([r for r in cached.records if r.split == "test"], config, "b", 1)
    with pytest.raises(ValueError, match="validation observations only"):
        ablation.validation_components(model, heldout, "cpu")
    assert [module.training for module in model.modules()] == flags


@pytest.mark.parametrize("arm", ["a", "b"])
def test_ablation_training_reproduces_production_control_and_preserves_artifacts(cached, arm):
    config = config_for(cached, variants=(arm,), epochs=2, patience=1, batch_size=2)
    production_path = cached.path.parent / f"production_{arm}"
    expected = train_variant(config, cached.records, cached.schedule, arm, production_path)
    checkpoint = torch.load(production_path / "best.pt", map_location="cpu", weights_only=False)
    output = cached.path.parent / f"architecture_test_{arm}"
    inspection = output / "saved_checkpoint_inspection"
    inspection.mkdir(parents=True)
    model = ablation.AblationModel(cached.manifest, arm, ablation.STABILIZED, initial_hazard(cached.records))
    ablation.load_saved_weights(model, checkpoint, "stabilized", cached.manifest)
    loader = _loader([r for r in cached.records if r.split == "validation"], config, arm, 1)
    components = ablation.validation_components(model, loader, "cpu")
    # Both synthetic saved controls use the same predictions; reference formula
    # and legacy checkpoint conversion are exercised independently above.
    for kind in ("reference", "stabilized"):
        write_json(inspection / f"{arm}_{kind}.json", components)
    study = {"output": output, "records": cached.records, "manifest": cached.manifest,
        "configs": {arm: config}, "specification": {"synthetic": True},
        "sources": {arm: {"stabilized_nll": expected["validation_nll"]}}}
    original_hashes = {p: file_sha256(p) for p in production_path.iterdir() if p.is_file()}
    summary = ablation.train_candidate(study, arm, "stabilized")
    assert summary["reproduced_saved_control"] and not summary["production_promoted"]
    assert summary["best_epoch"] == expected["best_epoch"]
    assert summary["validation_nll"] == pytest.approx(expected["validation_nll"], abs=1e-6)
    saved = torch.load(output / arm / "stabilized" / "best.pt", weights_only=False)
    for key, value in checkpoint["model"].items():
        if key != "_architecture_version":
            torch.testing.assert_close(saved["model"][key], value, rtol=1e-5, atol=1e-7)
    assert all(file_sha256(path) == digest for path, digest in original_hashes.items())
    assert not list(output.rglob("*test_predictions*"))
    assert not list(output.rglob("*gate.json"))
    assert ablation.train_candidate(study, arm, "stabilized") == json.loads(json.dumps(summary))
    (output / arm / "stabilized" / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint or identity"):
        ablation.train_candidate(study, arm, "stabilized")


def test_ablation_cli_stops_before_candidates_when_fresh_control_does_not_reproduce(tmp_path, monkeypatch):
    study = {"output": tmp_path}
    monkeypatch.setattr(ablation, "prepare_study", lambda *args: study)
    monkeypatch.setattr(ablation, "require_inspection", lambda *args: None)
    trained = []

    def fail(study, arm, experiment):
        trained.append((arm, experiment))
        return {"reproduced_saved_control": False}

    monkeypatch.setattr(ablation, "train_candidate", fail)
    with pytest.raises(ValueError, match="Fresh stabilized control"):
        ablation.main(["train", "--debug-run", "debug_test", "--arms", "b", "--experiments", "no_head_norm"])
    assert trained == [("b", "stabilized")]


def test_ablation_rejects_failed_saved_checkpoint_reproduction(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "inspect_saved", lambda study: {"reproduced": False})
    with pytest.raises(ValueError, match="Saved validation NLL reproduction failed"):
        ablation.require_inspection({"output": tmp_path})


def test_saved_checkpoint_inspection_reproduces_both_formats_and_never_evaluates_test(cached, monkeypatch):
    output = cached.path.parent / "architecture_saved_inspection"
    output.mkdir()
    sources, configs = {}, {}
    for arm in ("a", "b"):
        configs[arm] = config_for(cached, variants=(arm,))
        sources[arm] = {}
        for kind, experiment in (("reference", "legacy_all"), ("stabilized", "stabilized")):
            model = ablation.AblationModel(cached.manifest, arm, ablation.EXPERIMENTS[experiment], .003)
            state = {key: value for key, value in model.state_dict().items() if key != "_ablation_format_version"}
            identity = {"encoder_sha256": cached.manifest["encoder_sha256"]}
            if kind == "reference":
                state.update({f"encoder.{key}": value for key, value in cached.encoder.state_dict().items()})
            else:
                state["_architecture_version"] = torch.tensor(2)
            path = output / f"{arm}_{kind}.pt"
            torch.save({"model": state, "variant": arm, "identity": identity}, path)
            value = validation_loss(model, _loader([r for r in cached.records if r.split == "validation"],
                                                    configs[arm], arm, 1), "cpu")
            sources[arm].update({f"{kind}_checkpoint": str(path), f"{kind}_sha256": file_sha256(path),
                                 f"{kind}_nll": value})
    study = {"output": output, "records": cached.records, "manifest": cached.manifest,
             "configs": configs, "specification": {"synthetic": True}, "sources": sources}
    original_loader, observed = ablation._loader, []

    def validation_only(records, *args):
        assert records and all(r.split == "validation" for r in records)
        observed.append(True)
        return original_loader(records, *args)

    monkeypatch.setattr(ablation, "_loader", validation_only)
    result = ablation.inspect_saved(study)
    assert result["reproduced"] and len(observed) == 6
    assert set(result["arms"]) == {"a", "b"}
    for arm in ("a", "b"):
        delta = result["arms"][arm]["stabilized_minus_reference"]
        assert delta["nll"] == pytest.approx(delta["cumulative_hazard"] + delta["event_log_hazard"])
    sources["a"]["reference_nll"] += .1
    assert not ablation.inspect_saved(study)["reproduced"]
    sources["a"]["reference_sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed after"):
        ablation.inspect_saved(study)


def test_reference_replay_requires_verified_inputs_and_preserves_historical_targets(tmp_path):
    original = {"sources": {arm: {"reference_nll": 1.05} for arm in ("a", "b")},
                "cache_id": "original-cache", "ablation_code_sha256": "previous-code"}
    write_json(tmp_path / "specification.json", original)
    folder = tmp_path / "reference_replay_test"
    folder.mkdir()
    path = folder / "report.json"
    report = {"study_fingerprint": ablation.fingerprint(original),
        "replay_code_sha256": file_sha256(ablation.Path(ablation.__file__).with_name("ltsa_reference_replay.py")),
        "batch_size": 6, "device": "cpu", "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "feature_differences": {"singleton_minus_cache": {"max_abs": 0., "mean_image_rms": 0.}},
        "arms": {arm: {"expected_nll": 1.05,
            "raw_batch": {"validation_nll_original_reduction": 1.05},
            "raw_singleton": {"validation_nll_original_reduction": 1.04997, "overall": {"nll": 1.04997}},
            "cached": {"validation_nll_original_reduction": 1.04997, "overall": {"nll": 1.04997}}}
            for arm in ("a", "b")}}
    write_json(path, report)
    current = {**original, "ablation_code_sha256": "new-code"}
    accepted = ablation.verified_reference_replay(path, current, 6, "cpu")
    assert accepted["cached_reference_nll"] == {"a": 1.04997, "b": 1.04997}
    assert current["sources"]["a"]["reference_nll"] == 1.05
    assert accepted["sha256"] == file_sha256(path)
    with pytest.raises(ValueError, match="provenance"):
        ablation.verified_reference_replay(path, {**current, "cache_id": "changed"}, 6, "cpu")
    with pytest.raises(ValueError, match="provenance"):
        ablation.verified_reference_replay(path, current, 1, "cpu")
    for field, value, message in (
        ("raw", 1.051, "raw reproduction"),
        ("singleton", 1.04996, "likelihood agreement"),
        ("features", .0001, "exact singleton/cache"),
    ):
        changed = copy.deepcopy(report)
        if field == "features":
            changed["feature_differences"]["singleton_minus_cache"]["max_abs"] = value
        else:
            key = "raw_batch" if field == "raw" else "raw_singleton"
            changed["arms"]["a"][key]["validation_nll_original_reduction"] = value
        write_json(path, changed)
        with pytest.raises(ValueError, match=message):
            ablation.verified_reference_replay(path, current, 6, "cpu")
