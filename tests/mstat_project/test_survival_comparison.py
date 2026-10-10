import copy

import pytest

from mstat_project.ml.survival_comparison import compare_predictions, compare_experiments


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


def test_named_comparisons_stratify_history_and_suppress_sparse_intervals():
    data = predictions()
    named = {"b": data["b"], "history-1l": data["history"]}
    results = compare_experiments(named, (12.,), bootstrap_samples=100)
    assert {r["mri_count_stratum"] for r in results} == {"all", "2"}
    assert all(r["status"] == "descriptive_sparse" for r in results)
    assert all(r["bootstrap_samples_used"] == 0 and r["confidence_interval_low"] is None for r in results)
    bad = copy.deepcopy(named)
    bad["history-1l"][0]["available_history_ids"] = ["other"]
    with pytest.raises(ValueError, match="mismatch"):
        compare_experiments(bad, (12.,))


def test_individual_and_paired_intervals_are_reproducible_with_whole_patients():
    rows = [{"ptid": f"p{i:02d}", "split": "test", "landmark_months": 24.,
             "outcome_months": 25.+i, "event": i < 30, "acquisition_months": 20.,
             "available_history_ids": [f"{i}-0", f"{i}-20"], "visit_nums": [0, 3],
             "nll": 1., "survival": [.5]} for i in range(40)]
    history = copy.deepcopy(rows)
    for i, row in enumerate(history):
        row["survival"] = [.05 + i/50]
        row["nll"] = .8
    data = {"b": rows, "history-1l": history}
    results = compare_experiments(data, (60.,), bootstrap_samples=40, seed=7)
    primary = next(r for r in results if r["primary"])
    assert primary["reference_c_index"] == .5
    assert primary["candidate_c_index"] == 1.
    assert primary["difference"] == .5
    assert primary["reference_confidence_interval"] == [.5, .5]
    assert primary["candidate_confidence_interval"] == [1., 1.]
    assert primary["bootstrap_samples_used"] == 40
    assert primary["reference_mean_nll"] == 1.
    assert primary["candidate_mean_nll"] == pytest.approx(.8)
    assert results == compare_experiments(data, (60.,), bootstrap_samples=40, seed=7)
    repeated = copy.deepcopy(data)
    for name in repeated:
        second = copy.deepcopy(repeated[name])
        for row in second:
            row["landmark_months"] = 0.
        repeated[name].extend(second)
    expanded = compare_experiments(repeated, (60.,), bootstrap_samples=40, seed=7)
    assert next(r for r in expanded if r["primary"]) == primary
