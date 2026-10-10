"""Synthetic full-cohort continuation control, with no image encoder training."""

import pytest
import torch
import polars as pl

from test_ltsa_stabilization import cached  # noqa: F401
from test_ltsa_exploratory import exploratory_case, run_case  # noqa: F401
from mstat_project.ml import ltsa_b_continuation as control
from mstat_project.ml.survival import write_json
from mstat_project.ml.survival_protocol import file_sha256, read_json


def test_sampling_csv_preserves_numeric_identifiers_and_rejects_real_changes(tmp_path):
    frame = pl.DataFrame({"ptid": ["001", "002"], "image_id": ["001119575", "761013"],
                          "acquisition_months": [0., 48.657534246575345]})
    path = tmp_path / "sampled.csv"
    frame.write_csv(path)
    assert control.matching_samples(frame, path)
    assert not control.matching_samples(frame.reverse(), path)
    changed = frame.with_columns(pl.lit("1119575").alias("image_id"))
    assert not control.matching_samples(changed, path)
    changed = frame.with_columns((pl.col("acquisition_months") + 1).alias("acquisition_months"))
    assert not control.matching_samples(changed, path)


@pytest.fixture
def completed_history(exploratory_case, monkeypatch):
    source = run_case(exploratory_case)
    # Sparse synthetic panels are not evidence of clinical stability. Supply
    # positive gate evidence here; explicit rejection is exercised separately.
    report = read_json(source / "report.json")
    report["stable"] = True
    write_json(source / "report.json", report)
    original = control.stable_run
    monkeypatch.setattr(control, "stable_run", lambda p: True if p == source / "history-1l" else original(p))
    return exploratory_case, source


def run_control(case, name="exploratory_continuation_test"):
    cached, protocol, path, _, _ = case
    return control.run_control(cached.path.parent, cached.records, cached.schedule, protocol, path,
                               "cpu", "debug_test", "exploratory_test", name)


def test_continuation_initializes_exact_b_fresh_optimizer_paired_samples_and_preserves_history(completed_history, monkeypatch):
    case, source = completed_history
    cached = case[0]
    original_hashes = {p: file_sha256(p) for p in source.rglob("*") if p.is_file()}
    baseline = cached.path.parent / "seed_42" / "b" / "best.pt"
    original_hashes[baseline] = file_sha256(baseline)
    loader = control._loader

    def no_test(records, *args, **kwargs):
        assert records and all(r.split in ("train", "validation") for r in records)
        return loader(records, *args, **kwargs)

    monkeypatch.setattr(control, "_loader", no_test)
    output = run_control(case)
    report = read_json(output / "report.json")
    assert report["exploratory"] and not report["production_promoted"]
    assert report["continuation_nll"] <= report["epoch_zero_nll"]
    assert len(report["primary_comparison"]) == 1
    assert report["primary_comparison"][0]["reference"] == "b-continuation"
    assert report["primary_comparison"][0]["candidate"] == "history-1l"
    assert report["history_minus_continuation_nll"] == pytest.approx(report["history_nll"] - report["continuation_nll"])
    zero = torch.load(output / "b-continuation" / "epoch_zero.pt", weights_only=False)
    start = torch.load(baseline, weights_only=False)
    assert zero["optimizer"]["state"] == {}
    assert zero["epoch"] == 0 and zero["initialization_sha256"] == file_sha256(baseline)
    assert not any(k.startswith("transformer.") or k == "alpha" for k in zero["model"])
    for key, value in start["model"].items():
        torch.testing.assert_close(value, zero["model"][key], rtol=0, atol=0)
    epochs = read_json(output / "b-continuation" / "epochs.json")
    assert all(row["matched_history_samples"] for row in epochs if row["epoch"] > 0)
    assert all(file_sha256(p) == digest for p, digest in original_hashes.items())
    assert not list(output.rglob("*test_predictions*"))
    assert run_control(case) == output


def test_continuation_keeps_epoch_zero_when_updates_worsen_validation(completed_history, monkeypatch):
    case, _ = completed_history
    original = control.validation_loss
    calls = []

    def worsen(model, loader, device):
        if not calls:
            calls.append(original(model, loader, device))
            return calls[0]
        return calls[0] + 1.

    monkeypatch.setattr(control, "validation_loss", worsen)
    output = run_control(case, "exploratory_epoch_zero_test")
    report = read_json(output / "report.json")
    assert report["best_epoch"] == 0
    assert report["continuation_nll"] == report["epoch_zero_nll"]
    best = torch.load(output / "b-continuation" / "best.pt", weights_only=False)
    zero = torch.load(output / "b-continuation" / "epoch_zero.pt", weights_only=False)
    for key, value in zero["model"].items():
        torch.testing.assert_close(best["model"][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["history_artifact", "history_stability", "debug"])
def test_continuation_rejects_stale_or_failed_history_and_debug(completed_history, failure):
    case, source = completed_history
    if failure == "history_artifact":
        (source / "history-1l" / "validation_predictions.json").write_text("[]", encoding="utf-8")
    elif failure == "history_stability":
        report = read_json(source / "report.json")
        report["stable"] = False
        write_json(source / "report.json", report)
    else:
        debug = case[-1]
        report = read_json(debug)
        report["passed"] = False
        write_json(debug, report)
    with pytest.raises(ValueError):
        run_control(case)
    assert not (case[0].path.parent / "investigations" / "exploratory_continuation_test").exists()
