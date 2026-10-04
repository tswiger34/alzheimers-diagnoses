import copy

import pytest

from mstat_project.ml.survival_comparison import compare_predictions


def predictions():
    rows = [
        {"ptid": f"p{i}", "split": "test", "landmark_months": 60.0,
         "outcome_months": t, "event": event, "acquisition_months": 24.0,
         "available_history_ids": [f"{i}-0", f"{i}-24"], "visit_nums": [0, 3],
         "survival": [probability]}
        for i, (t, event, probability) in enumerate([(66., True, .9), (70., True, .5), (80., False, .1)])
    ]
    history = copy.deepcopy(rows)
    for row, probability in zip(history, [.1, .5, .9], strict=True):
        row["survival"] = [probability]
    return {"a": rows, "b": copy.deepcopy(rows), "history": history}


def test_comparison_is_paired_and_reports_candidate_minus_reference():
    data = predictions()
    data["history"].reverse()
    results = compare_predictions(data, (12.,), bootstrap_samples=100, seed=42)
    assert len(results) == 3
    assert results[0]["difference"] == 0
    assert results[2]["reference"] == "b"
    assert results[2]["candidate"] == "history"
    assert results[2]["difference"] == 1
    assert results[2]["confidence_interval_low"] == 1
    assert results[2]["confidence_interval_high"] == 1
    assert results == compare_predictions(data, (12.,), bootstrap_samples=100, seed=42)


@pytest.mark.parametrize("field,value", [
    ("ptid", "other"), ("outcome_months", 90.), ("event", False),
    ("acquisition_months", 23.), ("available_history_ids", ["wrong"]),
    ("visit_nums", [0, 1]), ("split", "train"),
])
def test_comparison_rejects_unmatched_cohorts(field, value):
    data = predictions()
    data["history"][0][field] = value
    with pytest.raises(ValueError):
        compare_predictions(data, (12.,), bootstrap_samples=0)


def test_no_events_within_horizon_is_not_a_zero_c_index():
    results = compare_predictions(predictions(), (1.,), bootstrap_samples=0)
    assert all(r["status"] == "no_comparable_pairs" and r["difference"] is None for r in results)


def test_duplicate_patient_landmark_is_rejected():
    data = predictions()
    data["a"].append(copy.deepcopy(data["a"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        compare_predictions(data, (12.,), bootstrap_samples=0)


@pytest.mark.parametrize("module_name,variants", [
    ("baseline", ("a", "b")), ("ltsa", ("history",)), ("compare", ("a", "b", "history")),
])
def test_existing_commands_select_the_continuous_pipeline(monkeypatch, module_name, variants):
    import importlib
    import mstat_project.ml.survival as pipeline

    module = importlib.import_module(f"mstat_project.ml.{module_name}")
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda config, **kwargs: calls.append((config, kwargs)))
    module.main(["--epochs", "2", "--preflight-only"])
    assert calls[0][0].variants == variants
    assert calls[0][0].epochs == 2
    assert calls[0][1]["preflight_only"] is True


@pytest.mark.parametrize("module_name", ["baseline", "ltsa", "compare"])
def test_legacy_commands_require_explicit_opt_in(monkeypatch, module_name):
    import importlib

    module = importlib.import_module(f"mstat_project.ml.{module_name}")
    monkeypatch.setattr(module, "legacy_main", lambda argv: argv)
    assert module.main(["--legacy", "--epochs", "2"]) == ["--epochs", "2"]
