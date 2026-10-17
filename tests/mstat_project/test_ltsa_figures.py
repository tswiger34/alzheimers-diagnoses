"""Scientific-method and reporting checks on synthetic patients only."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstat_project.ml import ltsa_figures as figures
from mstat_project.ml.continuous_survival import ContinuousHazardHead, ContinuousMRIModel
from mstat_project.ml.survival_comparison import compare_experiments


def rows(n=40, shift=0.):
    result = []
    for i in range(n):
        risk = .8-.6*i/max(n-1, 1)+shift
        result.append({"ptid": f"P{i:03d}", "split": "validation", "landmark_months": 24.,
            "outcome_months": 29.+i*2 if i < n//4 else 124.+i,
            "event": i < n//4, "acquisition_months": 20., "visit_nums": list(range(i % 3+1)),
            "available_history_ids": [f"P{i}_{k}" for k in range(i % 3+1)],
            "survival": [1-risk/5, 1-risk/2, 1-risk], "nll": 999.})
    return result


@pytest.fixture
def case(tmp_path):
    protocol = tmp_path / "protocol.json"
    figures.write_json(protocol, {"snapshot_hashes": {"source": "synthetic"}})
    manifest = {"format_version": 1, "root": ".", "protocol": "protocol.json",
        "primary": {"landmark_months": 24, "horizon_months": 60},
        "bootstrap_samples": 20, "bootstrap_seed": 42, "experiments": []}
    identity = {"encoder_sha256": "encoder", "preprocessing_sha256": "prep", "cohort_sha256": "cohort",
                "quadrature_points": 32}
    for role, shift in (("b", 0), ("b-continuation", .01), ("history-1l", -.01), ("history-2l", -.02)):
        directory = tmp_path / role
        directory.mkdir()
        checkpoint = directory / "best.pt"
        checkpoint.write_bytes(role.encode())  # audit/render must never deserialize it.
        summary = directory / "summary.json"
        predictions = directory / "validation_predictions.json"
        figures.write_json(summary, {"identity": identity, "variant": "b" if role in ("b", "b-continuation") else "history",
                                    "horizons_months": [12, 24, 60],
                                    "best_epoch": 1, "validation_nll": 1.})
        figures.write_json(predictions, rows(40, shift))
        figures.write_json(directory / "epochs.json", [{"epoch": 0, "validation_nll": 1.1},
                                                       {"epoch": 1, "validation_nll": 1.}])
        manifest["experiments"].append({"role": role, "seed": 42, "checkpoint": str(checkpoint),
            "checkpoint_sha256": figures.sha256(checkpoint), "summary": str(summary),
            "summary_sha256": figures.sha256(summary), "predictions": {"validation": str(predictions)},
            "prediction_sha256": {"validation": figures.sha256(predictions)}})
    manifest["experiments"].append({"role": "latest-only-1l", "seed": 42, "status": "pending"})
    path = tmp_path / "manifest.json"
    figures.write_json(path, manifest)
    return path, manifest


def test_paired_statistics_reproduces_existing_intervals_and_keeps_saved_nll_distinct():
    a, b = rows(), rows(shift=.01)
    b[0]["survival"] = [1., 1., 1.]
    result = figures.paired_statistics(a, b, 2, 60, 100, 42)
    existing = compare_experiments({"b": a, "history-1l": b}, [12, 24, 60],
                                  bootstrap_samples=100, seed=42, expected_split="validation")
    expected = next(r for r in existing if r["mri_count_stratum"] == "all" and r["horizon_months"] == 60)
    assert result["difference"] == pytest.approx(expected["difference"])
    assert result["confidence_interval"] == pytest.approx([expected["confidence_interval_low"], expected["confidence_interval_high"]])
    assert result["candidate_interval"] == pytest.approx(expected["candidate_confidence_interval"])
    assert result["bootstrap_samples_used"] == expected["bootstrap_samples_used"]
    assert result["nll_difference"] is None  # 999 is full-follow-up, never primary NLL.
    assert result == figures.paired_statistics(a, b, 2, 60, 100, 42)


def test_sparse_no_pairs_tied_risk_and_nll_are_handled_separately():
    a, b = rows(40), rows(40)
    for r in a+b:
        r["survival"] = [.7, .7, .7]
    result = figures.paired_statistics(a, b, 2, 60, 20)
    assert result["candidate_c_index"] == .5
    assert result["difference"] == 0
    sparse = figures.paired_statistics(a[:4], b[:4], 2, 60, 20)
    assert sparse["status"] == "descriptive_sparse"
    assert sparse["confidence_interval"] is None
    for r in a+b:
        r["event"] = False
    for r in a:
        r["horizon_nll"] = 2.
    for r in b:
        r["horizon_nll"] = 1.
    result = figures.paired_statistics(a, b, 2, 60, 20)
    assert result["status"] == "no_comparable_pairs"
    assert result["candidate_c_index"] is None
    assert result["nll_difference"] == -1.
    assert result["nll_interval"] is None  # no events: descriptive only
    assert figures.paired_statistics(a, b, 0, 12, 20)["nll_difference"] is None


@pytest.mark.parametrize("field,value", [("ptid", "wrong"), ("outcome_months", 200.),
    ("event", False), ("available_history_ids", ["other"]), ("acquisition_months", 19.)])
def test_match_rejects_different_patients_outcomes_and_histories(field, value):
    a, b = rows(), rows()
    b[0][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        figures.match_rows(a, b)


def test_validation_rejects_duplicates_nonfinite_and_nonmonotone():
    a = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        figures.validate_rows(a+[a[0]], [12, 24, 60], "validation")
    a[0]["survival"] = [.8, .9, .7]
    with pytest.raises(ValueError, match="nonmonotone"):
        figures.validate_rows(a, [12, 24, 60], "validation")
    a[0]["survival"] = [.8, float("nan"), .7]
    with pytest.raises(ValueError, match="nonmonotone"):
        figures.validate_rows(a, [12, 24, 60], "validation")


def test_km_retains_early_censoring_handles_ties_endpoint_and_unsupported_followup():
    # One censor at month 5 leaves three at risk at the month-10 event.
    assert figures.km_risk([5, 10, 60, 80], [False, True, False, False]) == pytest.approx(1/3)
    assert figures.km_risk([60, 60], [True, False]) == .5
    assert figures.km_risk([10, 20], [False, False]) is None
    assert figures.km_risk([10], [True]) == 1.


def test_calibration_keeps_equal_risks_together_and_is_reproducible():
    a = rows(120)
    for r in a:
        r["survival"] = [.9, .8, .7]
    result = figures.calibration(a, 2, 50, 42)
    assert len(result) == 1
    assert result[0]["patients"] == 120
    assert result[0]["confidence_interval"] is not None
    assert result == figures.calibration(a, 2, 50, 42)
    for r in a:
        r["event"] = False
        r["outcome_months"] = 30.
    result = figures.calibration(a, 2, 20)
    assert result[0]["status"] == "unsupported_followup"
    assert result[0]["observed_risk"] is None


def test_loss_components_average_within_patients_then_across_patients():
    a = [{"ptid": "a", "entry": 0, "image_ids": ["x"], "cumulative_hazard": 1., "event_log_hazard": 2., "nll": 3.},
         {"ptid": "a", "entry": 1, "image_ids": ["y"], "cumulative_hazard": 3., "event_log_hazard": 4., "nll": 7.},
         {"ptid": "b", "entry": 0, "image_ids": ["z"], "cumulative_hazard": 10., "event_log_hazard": 0., "nll": 10.}]
    result = figures.aggregate_components(a)
    assert result["nll"] == 7.5
    assert result["cumulative_hazard"] + result["event_log_hazard"] == result["nll"]
    with pytest.raises(ValueError, match="Duplicate"):
        figures.aggregate_components(a+[a[0]])
    a[0]["nll"] = 20.
    with pytest.raises(ValueError, match="reconstruct"):
        figures.aggregate_components(a)


def test_audit_never_deserializes_models_and_rejects_changed_artifacts(case, monkeypatch):
    path, manifest = case
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("audit invoked checkpoint loading"))
    result = figures.audit(path)
    assert result["pending"] == [{"role": "latest-only-1l", "seed": 42}]
    source = Path(manifest["experiments"][0]["predictions"]["validation"])
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction artifact changed"):
        figures.audit(path)


def continuation_horizon_metadata(case):
    """Match the original continued-B writer: horizons live only in its spec."""
    path, manifest = case
    item = next(e for e in manifest["experiments"] if e["role"] == "b-continuation")
    summary = figures.read_json(item["summary"])
    summary.pop("horizons_months")
    figures.write_json(item["summary"], summary)
    item["summary_sha256"] = figures.sha256(item["summary"])
    specification = Path(item["summary"]).parent.parent / "specification.json"
    figures.write_json(specification, {"config": {"horizons": [12, 24, 60]}})
    item["specification"] = str(specification)
    item["specification_sha256"] = figures.sha256(specification)
    figures.write_json(path, manifest)
    return path, manifest, item, specification


def test_audit_resolves_missing_continuation_horizons_from_pinned_specification(case, monkeypatch):
    path, _, _, specification = continuation_horizon_metadata(case)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("horizon resolution loaded a checkpoint"))
    _, _, _, experiments, _, files = figures.load_manifest(path, "validation")
    continued = next(e for e in experiments if e["role"] == "b-continuation")
    assert "horizons_months" not in continued["summary_data"]
    assert continued["horizons"] == [12, 24, 60]
    assert files[str(specification)] == figures.sha256(specification)
    assert len(figures.audit(path)["completed"]) == 4


def test_audit_rejects_changed_horizon_specification(case):
    path, _, _, specification = continuation_horizon_metadata(case)
    figures.write_json(specification, {"config": {"horizons": [12, 60]}})
    with pytest.raises(ValueError, match="Horizon specification changed"):
        figures.audit(path)


def test_audit_rejects_conflicting_horizons(case):
    path, manifest, item, _ = continuation_horizon_metadata(case)
    summary = figures.read_json(item["summary"])
    summary["horizons_months"] = [12, 60]
    figures.write_json(item["summary"], summary)
    item["summary_sha256"] = figures.sha256(item["summary"])
    figures.write_json(path, manifest)
    with pytest.raises(ValueError, match="Summary and specification prediction horizons differ"):
        figures.audit(path)


def test_audit_does_not_guess_missing_horizons(case):
    path, manifest, item, _ = continuation_horizon_metadata(case)
    item.pop("specification")
    item.pop("specification_sha256")
    figures.write_json(path, manifest)
    with pytest.raises(ValueError, match="Missing prediction horizons for b-continuation seed 42"):
        figures.audit(path)


@pytest.mark.parametrize("values", [None, [], [12, 24], [60, 12, 24], [12, 24, 24, 60],
                                  [True, 24, 60], [12, "24", 60], [12, float("inf"), 60]])
def test_prediction_horizons_rejects_invalid_metadata(tmp_path, values):
    with pytest.raises(ValueError, match="Invalid summary.horizons_months"):
        figures.prediction_horizons({"horizons_months": values}, {"role": "b", "seed": 42}, tmp_path, {})


def test_test_outcomes_are_not_opened_without_final_lock(case, monkeypatch):
    path, _ = case
    real_read = figures.read_json
    def guarded_read(source):
        assert "test_predictions" not in str(source)
        return real_read(source)
    monkeypatch.setattr(figures, "read_json", guarded_read)
    with pytest.raises(ValueError, match="final analysis lock"):
        figures.audit(path, "test")


def test_latest_only_results_require_matched_training_provenance(case):
    path, manifest = case
    latest = copy.deepcopy(manifest["experiments"][2])
    latest["role"] = "latest-only-1l"
    latest["control_specification"] = "missing.json"
    manifest["experiments"][-1] = latest
    figures.write_json(path, manifest)
    with pytest.raises(ValueError, match="control specifications"):
        figures.audit(path)


def test_failed_experiments_are_recorded_without_performance_estimates(case):
    path, manifest = case
    report = path.parent / "failed.json"
    figures.write_json(report, {"status": "failed", "diagnostic_only": True})
    manifest["experiments"].append({"role": "history-1l", "seed": 43, "status": "failed",
        "reason": "Nonfinite optimization", "report": str(report), "report_sha256": figures.sha256(report)})
    figures.write_json(path, manifest)
    result = figures.audit(path)
    assert any(r.get("status") == "failed" for r in result["pending"])
    assert not any(r["role"] == "history-1l" and r["seed"] == 43 for r in result["completed"])


def test_evaluation_rejects_changed_sources_and_mislabeled_history(case):
    path, _ = case
    _, _, _, experiments, _, sources = figures.load_manifest(path, "validation")
    primary = figures.selected(experiments[0]["rows"])
    evaluation = {"format_version": 1, "split": "validation", "analysis_sha256": figures.sha256(figures.__file__),
                  "sources": sources, "experiments": {}}
    for e in experiments:
        evaluated = [{"ptid": r["ptid"], "entry": 24, "image_ids": r["available_history_ids"],
                      "cumulative_hazard": .1, "event_log_hazard": 1., "nll": 1.1} for r in figures.selected(e["rows"])]
        evaluation["experiments"][f"{e['role']}:{e['seed']}"] = {"checkpoint_sha256": e["checkpoint_sha256"],
                                                                  "primary_rows": evaluated}
    target = path.parent / "evaluation.json"
    figures.write_json(target, evaluation)
    figures.attach_evaluation(experiments, target, "validation", {})
    assert all(r["horizon_nll"] == 1.1 for r in primary)
    evaluation["experiments"]["b:42"]["primary_rows"][0]["entry"] = 0
    figures.write_json(target, evaluation)
    with pytest.raises(ValueError, match="history or entry differs"):
        figures.attach_evaluation(experiments, target, "validation", {})


def test_joint_seed_summary_resamples_patients_not_seeds(case):
    path, _ = case
    manifest, _, _, experiments, _, _ = figures.load_manifest(path, "validation")
    all_seeds = []
    for e in experiments:
        for seed in (42, 43, 44):
            clone = copy.deepcopy(e)
            clone["seed"] = seed
            if clone["role"] == "history-1l":
                clone["rows"][0]["survival"] = [1., 1., 1.]
            all_seeds.append(clone)
    tables = figures.analysis_tables(manifest, all_seeds)
    original = next(r for r in tables["comparisons"] if r["comparison"] == "Training-controlled" and r["seed"] == 42)
    mean = next(r for r in tables["cross_seed"] if r["comparison"] == "Training-controlled")
    assert mean["difference"] == pytest.approx(original["difference"])
    assert mean["confidence_interval"] == pytest.approx(original["confidence_interval"])


def test_render_exports_eight_figures_tables_and_preserves_inputs(case, monkeypatch):
    path, manifest = case
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("render invoked checkpoint loading"))
    before = {str(p): figures.sha256(p) for p in path.parent.rglob("*") if p.is_file()}
    output = path.parent / "figures"
    figures.render(path, "validation", output)
    for name in figures.CAPTIONS:
        for suffix in ("png", "pdf", "svg"):
            assert (output / f"{name}.{suffix}").stat().st_size > 100
    provenance = figures.read_json(output / "provenance.json")
    assert not provenance["likelihood_evaluated"]
    assert len(provenance["outputs"]) == 24
    assert "Pending" in (output / "figure_3.svg").read_text(encoding="utf-8")
    for source, digest in before.items():
        assert figures.sha256(source) == digest
    for source in output.glob("source_*.csv"):
        content = source.read_text(encoding="utf-8")
        assert "ptid" not in content and "P000" not in content
    with pytest.raises(FileExistsError):
        figures.render(path, "validation", output)


def test_primary_horizon_likelihood_uses_original_months_and_event_boundary():
    head = ContinuousHazardHead(4, baseline_hazard=.01)
    with torch.no_grad():
        head.output_linear.weight.zero_()
    entry = torch.full((3,), 24.)
    # The final source event rounds to 84 in float32 but is after the horizon.
    source_times = [54., 84., 84.000001]
    outcome = torch.tensor(source_times)
    batch = {"entry": entry, "outcome": outcome,
             "records": [SimpleNamespace(prediction_months=24., outcome_months=t, event=True) for t in source_times]}
    h, event_log = figures.primary_components(head, torch.zeros(3, 4), batch)
    torch.testing.assert_close(h, torch.tensor([.3, .6, .6]))
    torch.testing.assert_close(event_log, torch.tensor([-np.log(.01), -np.log(.01), 0.], dtype=torch.float32))


def test_evaluate_replays_synthetic_frozen_checkpoint_without_training(tmp_path, monkeypatch):
    # Integration fixture includes cache, full validation weighting and primary predictions.
    from test_continuous_survival import DiagnosticEncoder, fixture_data
    from mstat_project.ml import ltsa_experiments
    from mstat_project.ml.survival import SurvivalConfig, _loader, model_identity, predict, validation_loss
    from mstat_project.ml.survival_data import at_landmark
    from mstat_project.ml.survival_embeddings import CachedEncoder, export_embeddings
    from mstat_project.ml.survival_protocol import implementation_hashes
    torch.set_num_threads(2)
    records, _, schedule, _ = fixture_data(tmp_path)
    cache = tmp_path / "embeddings.pt"
    provenance = {"source": "synthetic", "schedule": "synthetic", "splits": "synthetic"}
    cache_manifest = export_embeddings(cache, records, DiagnosticEncoder().eval(),
                                      spatial_size=(16, 16, 16), provenance=provenance)
    config = SurvivalConfig(cache_path=cache, spatial_size=(16, 16, 16), device="cpu", batch_size=2)
    model = ContinuousMRIModel(CachedEncoder(cache_manifest), "b", freeze_encoder=True).eval()
    identity = model_identity(model, records, config)
    directory = tmp_path / "b"
    directory.mkdir()
    checkpoint = directory / "best.pt"
    val = _loader([r for r in records if r.split == "validation"], config, "b", schedule.height)
    nll = validation_loss(model, val, "cpu")
    torch.save({"variant": "b", "model": model.state_dict(), "identity": identity,
                "config": asdict(config), "implementation_hashes": implementation_hashes()}, checkpoint)
    primary = at_landmark(records, 24, "validation")
    predictions = predict(model, _loader(primary, config, "b", schedule.height), "cpu", config.horizons)
    summary = directory / "summary.json"
    prediction_path = directory / "validation_predictions.json"
    figures.write_json(summary, {"identity": identity, "variant": "b", "validation_nll": nll, "best_epoch": 0,
                                "horizons_months": list(config.horizons)})
    figures.write_json(prediction_path, predictions)
    figures.write_json(tmp_path / "protocol.json", {"snapshot_hashes": provenance,
                       "reference_config": {"spatial_size": [16, 16, 16]}})
    manifest = {"format_version": 1, "root": ".", "protocol": "protocol.json", "cache": "embeddings.pt",
        "primary": {"landmark_months": 24, "horizon_months": 60}, "bootstrap_samples": 10,
        "experiments": [{"role": "b", "seed": 42, "checkpoint": str(checkpoint),
            "checkpoint_sha256": figures.sha256(checkpoint), "summary": str(summary),
            "summary_sha256": figures.sha256(summary), "predictions": {"validation": str(prediction_path)},
            "prediction_sha256": {"validation": figures.sha256(prediction_path)}}]}
    path = tmp_path / "manifest.json"
    figures.write_json(path, manifest)
    monkeypatch.setattr(ltsa_experiments, "load_snapshot", lambda protocol: (records, schedule, {}))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: pytest.fail("evaluation created optimizer"))
    output = tmp_path / "evaluation"
    figures.evaluate(path, "validation", "cpu", output)
    result = figures.read_json(output / "evaluation.json")
    assert result["experiments"]["b:42"]["full"]["nll"] == pytest.approx(nll, abs=1e-5)
    _, _, _, experiments, _, sources = figures.load_manifest(path, "validation")
    figures.attach_evaluation(experiments, output / "evaluation.json", "validation", sources)
    assert all("horizon_nll" in r for r in experiments[0]["rows"])
