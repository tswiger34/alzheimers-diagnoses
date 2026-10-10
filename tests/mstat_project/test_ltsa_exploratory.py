"""Synthetic integration checks for the explicit exploratory LTSA amendment."""

from dataclasses import asdict

import pytest
import torch

from test_ltsa_stabilization import cached  # noqa: F401
from mstat_project.ml import ltsa_exploratory as exploratory
from mstat_project.ml import survival
from mstat_project.ml.ltsa_architecture_ablation import AblationModel, EXPERIMENTS
from mstat_project.ml.survival import SurvivalConfig, train_variant, write_json
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, cohort_fingerprint, file_sha256, fingerprint, implementation_hashes, read_json,
    require_baseline_gate,
)


@pytest.fixture
def exploratory_case(cached, monkeypatch):
    root = cached.path.parent
    protocol = read_json(PROTOCOL_PATH)
    protocol["snapshot_hashes"] = cached.provenance
    protocol["bootstrap_samples"] = 0  # Tiny synthetic strata have no inferential intervals.
    config = SurvivalConfig(cache_path=cached.path, device="cpu", spatial_size=(16, 16, 16),
                            epochs=2, patience=1, batch_size=2)
    for key in ("epochs", "patience", "batch_size"):
        protocol["reference_config"][key] = asdict(config)[key]
    # Protocols use JSON arrays; a tuple would differ from the saved list when
    # the runner compares the in-memory protocol with protocol.json.
    protocol["reference_config"]["spatial_size"] = list(config.spatial_size)
    arms, summaries = {}, {}
    for arm in ("a", "b"):
        directory = root / "seed_42" / arm
        summary = train_variant(config, cached.records, cached.schedule, arm, directory)
        summaries[arm] = summary
        path = directory / "best.pt"
        value = summary["validation_nll"]
        protocol["reference_baselines"][arm]["validation_nll"] = value - .1
        arms[arm] = {"passed": False, "stable": True, "validation_nll": value,
                     "checkpoint": str(path.resolve()), "checkpoint_sha256": file_sha256(path)}
    reference = AblationModel(cached.manifest, "b", EXPERIMENTS["legacy_all"], .003)
    state = {k: v for k, v in reference.state_dict().items() if k != "_ablation_format_version"}
    state.update({f"encoder.{k}": v for k, v in cached.encoder.state_dict().items()})
    reference_path = root / "historical_b.pt"
    torch.save({"variant": "b", "model": state}, reference_path)
    protocol["reference_baselines"]["b"].update(checkpoint=str(reference_path),
                                                checkpoint_sha256=file_sha256(reference_path))
    protocol_path = root / "protocol.json"
    write_json(protocol_path, protocol)
    gate = {"passed": False, "arms": arms, "identity": summaries["b"]["identity"],
            "protocol_sha256": fingerprint(protocol), "implementation_hashes": implementation_hashes()}
    gate_path = root / "seed_42" / "baseline_gate.json"
    write_json(gate_path, gate)
    debug_path = root / "debug_test" / "gate.json"
    debug_path.parent.mkdir()
    write_json(debug_path, {"passed": True, "protocol_sha256": fingerprint(protocol),
        "snapshot_hashes": protocol["snapshot_hashes"], "encoder_sha256": cached.manifest["encoder_sha256"],
        "implementation_hashes": implementation_hashes(), "full_cohort_sha256": cohort_fingerprint(cached.records),
        "arms": {arm: {"passed": True, "initial": {"nll": 2.},
                       "final": {"nll": 1., "c_index": 1., "risk_variance": .1},
                       "initialization_check": {"passed": True, "max_survival_difference": .001}}
                 for arm in ("a", "b", "history")}})
    # Source stability is synthetic evidence here; production stable_run is also
    # checked explicitly by the runner and its rejection is tested below.
    real_stable = exploratory.stable_run
    monkeypatch.setattr(exploratory, "stable_run", lambda path: True if path.parent.name == "seed_42" else real_stable(path))
    return cached, protocol, protocol_path, gate, debug_path


def run_case(case, name="exploratory_test"):
    cached, protocol, path, _, _ = case
    return exploratory.run_exploratory(cached.path.parent, cached.records, cached.schedule,
                                       protocol, path, "cpu", "debug_test", name)


def test_exploratory_runs_with_failed_nll_only_keeps_epoch_zero_and_production_gate(exploratory_case, monkeypatch):
    cached, protocol, protocol_path, gate, debug_path = exploratory_case
    root = cached.path.parent
    originals = {p: file_sha256(p) for p in (root / "seed_42").rglob("*") if p.is_file()}
    originals[debug_path] = file_sha256(debug_path)
    original_loader = survival._loader

    def no_test(records, *args, **kwargs):
        assert records and all(r.split in ("train", "validation") for r in records)
        return original_loader(records, *args, **kwargs)

    monkeypatch.setattr(survival, "_loader", no_test)
    monkeypatch.setattr(exploratory, "_loader", no_test)
    output = run_case(exploratory_case)
    report = read_json(output / "report.json")
    spec = read_json(output / "specification.json")
    assert report["exploratory"] and not report["production_promoted"]
    assert not report["source_baseline_gate_passed"]
    assert "thresholds only" in spec["waived_requirement"]
    assert spec["debug_gate_sha256"] == file_sha256(debug_path)
    assert report["validation_nll"] <= report["epoch_zero_nll"] + 1e-5
    assert len(report["primary_comparison"]) == 1
    assert report["primary_comparison"][0]["reference"] == "b"
    assert not list(output.rglob("*test_predictions*"))
    assert not list(output.rglob("baseline_gate.json"))
    for relative in ("epoch_zero.pt", "history-1l/best.pt", "history-1l/last.pt"):
        saved = torch.load(output / relative, weights_only=False)
        assert saved["exploratory"] and not saved["production_promoted"]
        assert saved["specification_sha256"] == file_sha256(output / "specification.json")
    assert all(file_sha256(path) == digest for path, digest in originals.items())
    with pytest.raises(ValueError, match="Production baseline gate failed"):
        require_baseline_gate(root / "seed_42" / "baseline_gate.json", protocol, gate["identity"],
                              root / "seed_42" / "b" / "best.pt")
    assert run_case(exploratory_case) == output
    (output / "epoch_zero_predictions.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        run_case(exploratory_case)


@pytest.mark.parametrize("failure", ["debug", "stability", "checkpoint", "protocol"])
def test_exploratory_rejects_non_nll_failures_before_training(exploratory_case, monkeypatch, failure):
    cached, protocol, path, gate, debug = exploratory_case
    if failure == "debug":
        value = read_json(debug)
        value["passed"] = False
        write_json(debug, value)
    elif failure == "stability":
        monkeypatch.setattr(exploratory, "stable_run", lambda path: False)
    elif failure == "checkpoint":
        gate["arms"]["b"]["checkpoint_sha256"] = "changed"
        write_json(cached.path.parent / "seed_42" / "baseline_gate.json", gate)
    else:
        protocol["reference_config"]["learning_rate"] *= 2
    with pytest.raises(ValueError):
        run_case(exploratory_case)
    assert not (cached.path.parent / "investigations").exists()


def test_exploratory_rejects_incomplete_output(exploratory_case, monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(exploratory, "train_variant", stop)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_case(exploratory_case)
    with pytest.raises(ValueError, match="Incomplete exploratory"):
        run_case(exploratory_case)
