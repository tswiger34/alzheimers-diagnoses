"""Source-backed scientific figures; never trains or selects a checkpoint.

Audit/render read JSON only. Torch/model imports are confined to explicit evaluate.
All paths in a manifest are relative to its declared root, itself relative to the
manifest. Evaluation and figure directories must be new to preserve past editions.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = 1
ROLES = ("a", "historical-b", "b", "b-continuation", "history-1l", "history-2l", "latest-only-1l")
LABELS = {
    "a": "Baseline A",
    "historical-b": "Historical B",
    "b": "Baseline B",
    "b-continuation": "Continued B",
    "history-1l": "LTSA-1L",
    "history-2l": "LTSA-2L",
    "latest-only-1l": "Latest-only LTSA-1L",
}
COLORS = dict(zip(ROLES, ("#777777", "#555555", "#52758c", "#b87b26", "#263d77", "#89618c", "#59704c")))
MARKERS = dict(zip(ROLES, ("v", "x", "s", "D", "o", "^", "P")))
PAIRS = (
    ("b", "history-1l", "Original primary"),
    ("b", "b-continuation", "Additional training"),
    ("b-continuation", "history-1l", "Training-controlled"),
    ("history-1l", "history-2l", "Secondary depth"),
    ("latest-only-1l", "history-1l", "Direct history"),
)
CAPTIONS = {
    "figure_1": "Study design and eligible MRI information. All scans precede the prediction landmark and clinical outcome. Counts describe patients at risk at month 24; early censoring is retained.",
    "figure_2": "Primary endpoint: prediction at month 24 over the following 60 months. C-index uses horizon-truncated outcomes. Differences are candidate minus reference. Positive C-index differences and negative NLL differences favor the candidate. Intervals are paired patient-bootstrap 95% percentile intervals.",
    "figure_3": "Controls for additional training and access to earlier scans. Learning curves show the full-validation, full-follow-up patient-balanced selection objective. Seed variation is distinct from patient-sampling uncertainty. Latest-only results require a separately trained matched LTSA control.",
    "figure_4": "Secondary MRI-count analyses at the primary endpoint. Counts include the latest scan. Groups overlap for All and ≥2, so rows are not independent. Open markers denote descriptive sparse estimates; subgroup interval overlap is not a test of effect modification.",
    "figure_5": "Coarse grouped calibration at month 24 / next 60 months. Model-specific predicted-risk tertiles retain tied risks together. Observed risk is 1 − Kaplan–Meier survival at 60 months, retaining early censoring. Bootstrap boundaries are fixed. Independent censoring within groups is assumed; sparse or unsupported groups receive no inferential interval.",
    "supplement_1": "Secondary landmark/horizon grid. Each cell is a separate paired patient comparison. No repeated landmark observations are treated as independent patients. Sparse cells are descriptive; × denotes a nonestimable cell. This grid does not redefine the primary endpoint.",
    "supplement_2": "Likelihood components and shared-architecture investigation. Components use equal patient weights after averaging eligible observations within each patient. Full-follow-up selection NLL and primary-window NLL are distinct. Single-factor reversals are conditional effects, not a unique causal decomposition.",
    "supplement_3": "Optimization and prediction diagnostics. Actual optimizer update norms and fixed-panel risk variance are diagnostic quantities. Nonzero updates or changed predictions alone do not demonstrate a benefit from earlier MRI scans. Patient points carry no exported identifying labels.",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def path_for(root, value):
    # Native and Windows-style relative artifact names remain portable.
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def prediction_horizons(summary, item, root, files):
    """Resolve prediction order from saved, pinned metadata, never from defaults.

    The original B-continuation summary omits horizons. Its immutable experiment
    specification records the same config.horizons used to write predictions.
    """
    candidates = []
    if "horizons_months" in summary:
        candidates.append(("summary.horizons_months", summary["horizons_months"]))
    if "specification" in item or "specification_sha256" in item:
        if not item.get("specification") or not item.get("specification_sha256"):
            raise ValueError("Horizon specification requires both a path and specification_sha256")
        source = path_for(root, item["specification"])
        digest = sha256(source)
        if digest != item["specification_sha256"]:
            raise ValueError(f"Horizon specification changed: {source}")
        specification = read_json(source)
        config = specification.get("config") if isinstance(specification, dict) else None
        candidates.append(("specification.config.horizons", config.get("horizons") if isinstance(config, dict) else None))
        files[str(source)] = digest
    if not candidates:
        raise ValueError(
            f"Missing prediction horizons for {item['role']} seed {item['seed']}: "
            "summary lacks horizons_months; provide the saved specification and specification_sha256"
        )
    for label, values in candidates:
        if (not isinstance(values, list) or not values
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
            or values != sorted(set(values)) or 60 not in values):
            raise ValueError(f"Invalid {label}: expected increasing positive horizons including 60 months")
    horizons = candidates[0][1]
    if any(values != horizons for _, values in candidates[1:]):
        raise ValueError("Summary and specification prediction horizons differ")
    return list(horizons)


def load_manifest(path, split):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("format_version") != VERSION or split not in ("validation", "test"):
        raise ValueError("Unsupported manifest version or split")
    if manifest.get("primary") != {"landmark_months": 24, "horizon_months": 60}:
        raise ValueError("This figure package fixes the primary endpoint at month 24 / next 60")
    if manifest.get("bootstrap_samples", 1000) < 1:
        raise ValueError("At least one bootstrap resample is required")
    root = path_for(path.parent, manifest.get("root", "."))
    protocol_path = path_for(root, manifest["protocol"])
    protocol = read_json(protocol_path)
    files = {str(path): sha256(path), str(protocol_path): sha256(protocol_path)}
    directory = Path(__file__).parent
    dependencies = {p for p in directory.glob("*survival*.py")}
    dependencies.update(
        directory / name
        for name in (
            "ltsa_experiments.py",
            "models.py",
            "landmarks.py",
            "metrics.py",
            "utils.py",
            "ltsa_architecture_ablation.py",
        )
    )
    for dependency in sorted(dependencies):
        files[str(dependency)] = sha256(dependency)
    if split == "test":
        # Authorization is checked before even opening test prediction outcomes.
        validate_test_lock(
            root,
            manifest,
            protocol,
            [e for e in manifest["experiments"] if e.get("status") not in ("pending", "failed")],
            files,
        )
    loaded, pending, seen, common_identity = [], [], set(), None
    for item in manifest["experiments"]:
        key = (item["role"], item["seed"])
        if item["role"] not in ROLES or key in seen:
            raise ValueError("Unknown role or duplicate role/seed")
        seen.add(key)
        if item.get("status") == "pending":
            pending.append({"role": item["role"], "seed": item["seed"]})
            continue
        if item.get("status") == "failed":
            source = path_for(root, item["report"])
            if not item.get("reason") or sha256(source) != item["report_sha256"]:
                raise ValueError("Failed experiment requires a reason and pinned diagnostic report")
            files[str(source)] = sha256(source)
            pending.append(
                {"role": item["role"], "seed": item["seed"], "status": "failed", "reason": item["reason"]}
            )
            continue
        required = {
            "checkpoint",
            "checkpoint_sha256",
            "summary",
            "summary_sha256",
            "predictions",
            "prediction_sha256",
        }
        if (
            required - item.keys()
            or split not in item.get("predictions", {})
            or split not in item.get("prediction_sha256", {})
        ):
            raise ValueError(f"Incomplete {split} experiment: {item['role']} seed {item['seed']}")
        checkpoint = path_for(root, item["checkpoint"])
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
        summary_path = path_for(root, item["summary"])
        if sha256(summary_path) != item["summary_sha256"]:
            raise ValueError("Selected summary changed")
        summary = read_json(summary_path)
        if not math.isfinite(summary["validation_nll"]):
            raise ValueError("Nonfinite selected validation NLL")
        expected_variant = (
            "a"
            if item["role"] == "a"
            else "history"
            if item["role"] in ("history-1l", "history-2l", "latest-only-1l")
            else "b"
        )
        if summary.get("variant") != expected_variant:
            raise ValueError("Manifest role does not match selected architecture variant")
        identity = summary.get("identity")
        if item["role"] != "historical-b":
            if not identity or not all(
                k in identity
                for k in ("encoder_sha256", "preprocessing_sha256", "cohort_sha256", "quadrature_points")
            ):
                raise ValueError("Completed experiment lacks cohort/encoder/preprocessing identity")
            shared = {
                k: identity[k]
                for k in ("encoder_sha256", "preprocessing_sha256", "cohort_sha256", "quadrature_points")
            }
            if common_identity is not None and shared != common_identity:
                raise ValueError("Experiments have incompatible data/encoder identities")
            common_identity = shared
        prediction_path = path_for(root, item["predictions"][split])
        if sha256(prediction_path) != item["prediction_sha256"][split]:
            raise ValueError("Selected prediction artifact changed")
        rows = read_json(prediction_path)
        horizons = prediction_horizons(summary, item, root, files)
        validate_rows(rows, horizons, split)
        for source in (checkpoint, summary_path, prediction_path):
            files[str(source)] = sha256(source)
        directory = summary_path.parent
        # Pin JSON diagnostics/epochs/counts and any completed investigation report.
        for source in sorted(directory.glob("*.json")):
            if source.name != "test_predictions.json" or split == "test":
                files[str(source)] = sha256(source)
        if item.get("report"):
            report_path = path_for(root, item["report"])
            report = read_json(report_path)
            if report.get("stable") is False:
                raise ValueError("Unstable experiment cannot be presented as a retained result")
            for relative, digest in report.get("artifacts", {}).items():
                source = path_for(report_path.parent, relative)
                if not source.is_relative_to(report_path.parent) or sha256(source) != digest:
                    raise ValueError("Changed or escaping investigation artifact")
                files[str(source)] = digest
            files[str(report_path)] = sha256(report_path)
        loaded.append(
            {
                **item,
                "checkpoint_path": checkpoint,
                "directory": directory,
                "summary_data": summary,
                "rows": rows,
                "horizons": horizons,
            }
        )
    if not loaded:
        raise ValueError("Manifest has no completed experiments")
    baseline = next((e for e in loaded if e["role"] == "b" and e["seed"] == 42), loaded[0])
    for experiment in loaded:
        if experiment["horizons"] != baseline["horizons"]:
            raise ValueError("Prediction horizons differ")
        match_rows(baseline["rows"], experiment["rows"])
    for experiment in loaded:
        if experiment["role"] == "latest-only-1l":
            validate_history_control(root, experiment, loaded, files)
    if manifest.get("architecture_report"):
        source = path_for(root, manifest["architecture_report"])
        files[str(source)] = sha256(source)
    if manifest.get("cohort_audit"):
        source = path_for(root, manifest["cohort_audit"])
        cohort = read_json(source)["cohort"]
        if (
            cohort["source_images"] - sum(r["len"] for r in cohort["exclusions"]) - cohort["unassigned_images"]
            != cohort["eligible_images"]
        ):
            raise ValueError("Cohort audit image flow does not reconcile")
        files[str(source)] = sha256(source)
    return manifest, root, protocol, loaded, pending, files


def validate_history_control(root, candidate, experiments, files):
    control = next((e for e in experiments if e["role"] == "history-1l" and e["seed"] == candidate["seed"]), None)
    if control is None:
        raise ValueError("Latest-only control requires matching full-history seed")
    if any(not e.get("control_specification") for e in (control, candidate)):
        raise ValueError("Direct history comparison requires both matched control specifications")
    left, right = [path_for(root, e["control_specification"]) for e in (control, candidate)]
    specs = [read_json(p) for p in (left, right)]
    required = (
        "shared_initial_weights_sha256",
        "baseline_checkpoint_sha256",
        "sample_files_sha256",
        "optimizer",
        "scheduler",
        "selection",
        "architecture",
        "training_settings",
    )
    if any(not specs[0].get(k) or specs[0][k] != specs[1].get(k) for k in required):
        raise ValueError("Latest-only and full-history controls lack matching training provenance")
    if [s.get("input_mode") for s in specs] != ["full_history", "latest_only"]:
        raise ValueError("Direct control must differ only in input availability")
    baseline = next((e for e in experiments if e["role"] == "b" and e["seed"] == candidate["seed"]), None)
    if baseline is None or specs[0]["baseline_checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ValueError("Direct control requires corresponding selected B initialization")
    for spec in specs:
        initial = path_for(root, spec["shared_initial_weights"])
        if sha256(initial) != spec["shared_initial_weights_sha256"]:
            raise ValueError("Shared initial weights changed")
        files[str(initial)] = sha256(initial)
        for value in (spec["shared_initial_weights_sha256"], spec["baseline_checkpoint_sha256"]):
            if len(value) != 64:
                raise ValueError("Control requires SHA256 fingerprints")
        for sample, digest in spec["sample_files_sha256"].items():
            source = path_for(root, sample)
            if sha256(source) != digest:
                raise ValueError("Matched sample artifact changed")
            files[str(source)] = digest
    for source in (left, right):
        files[str(source)] = sha256(source)


def validate_test_lock(root, manifest, protocol, experiments, files):
    if not manifest.get("final_lock"):
        raise ValueError("Test figures require an explicit final analysis lock and passed gates")
    path = path_for(root, manifest["final_lock"])
    lock = read_json(path)
    # Analysis changes and checkpoint changes require a new, reviewed lock.
    from mstat_project.ml.survival_embeddings import fingerprint
    from mstat_project.ml.survival_protocol import implementation_hashes

    expected = {f"{e['role']}:{e['seed']}": e["checkpoint_sha256"] for e in experiments}
    if (
        lock.get("protocol_sha256") != fingerprint(protocol)
        or lock.get("checkpoints") != expected
        or lock.get("analysis_sha256") != sha256(__file__)
        or not lock.get("selection_locked")
        or not lock.get("gates")
    ):
        raise ValueError("Final test lock does not match this protocol, analysis and selected checkpoints")
    kinds = set()
    for gate in lock["gates"]:
        source = path_for(root, gate["path"])
        value = read_json(source)
        if (
            sha256(source) != gate["sha256"]
            or value.get("passed") is not True
            or value.get("implementation_hashes") != implementation_hashes()
            or value.get("protocol_sha256") != fingerprint(protocol)
        ):
            raise ValueError("Final figures require unchanged, passed experiment gates")
        kinds.add(gate["kind"])
        files[str(source)] = sha256(source)
    if not {"debug", "baseline"}.issubset(kinds):
        raise ValueError("Final lock requires debug and baseline promotion gates")
    files[str(path)] = sha256(path)


def validate_rows(rows, horizons, split):
    seen = set()
    if 60 not in horizons or len(horizons) != len(set(horizons)):
        raise ValueError("Primary horizon missing or duplicate horizons")
    for r in rows:
        key = (r["landmark_months"], r["ptid"])
        history = r["available_history_ids"]
        if key in seen or r["split"] != split:
            raise ValueError("Duplicate patient/landmark or wrong prediction split")
        seen.add(key)
        if (
            not history
            or len(history) != len(set(history))
            or len(history) != len(r["visit_nums"])
            or type(r["event"]) is not bool
            or not isinstance(r["ptid"], str)
            or not all(math.isfinite(r[k]) for k in ("landmark_months", "outcome_months", "acquisition_months"))
            or not 0 <= r["acquisition_months"] <= r["landmark_months"] < r["outcome_months"]
        ):
            raise ValueError("Invalid eligible patient/history/outcome")
        survival = r["survival"]
        if (
            len(survival) != len(horizons)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in survival)
            or any(a < b - 1e-7 for a, b in zip(survival, survival[1:]))
        ):
            raise ValueError("Invalid or nonmonotone survival predictions")
    if not rows or not any(r["landmark_months"] == 24 for r in rows):
        raise ValueError("Completed experiment lacks primary-landmark predictions")


def match_rows(left, right):
    fields = ("split", "outcome_months", "event", "acquisition_months", "available_history_ids", "visit_nums")
    a = {(r["landmark_months"], r["ptid"]): r for r in left}
    b = {(r["landmark_months"], r["ptid"]): r for r in right}
    if a.keys() != b.keys() or any(any(a[k][f] != b[k][f] for f in fields) for k in a):
        raise ValueError("Patient, outcome or eligible-history mismatch")


def selected(rows, landmark=24, stratum="all"):
    return sorted(
        (
            r
            for r in rows
            if r["landmark_months"] == landmark
            and (
                stratum == "all"
                or (stratum == "2+" and len(r["available_history_ids"]) >= 2)
                or (stratum == "3+" and len(r["available_history_ids"]) >= 3)
                or str(len(r["available_history_ids"])) == stratum
            )
        ),
        key=lambda r: r["ptid"],
    )


def resample_counts(n, repeats, seed):
    rng = random.Random(seed)
    return (
        np.array(
            [np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(repeats)], dtype=float
        )
        if n
        else np.empty((repeats, 0))
    )


def interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def paired_statistics(left, right, horizon_index, horizon, repeats=1000, seed=42):
    """Exact existing Harrell rule; multiplicities implement whole-patient bootstrap.

    Compute each original comparable pair once. Duplicated patient draws give
    multiplicity products, and strict time ordering excludes self/tied-time pairs.
    """
    match_rows(left, right)
    left = sorted(left, key=lambda r: r["ptid"])
    right = sorted(right, key=lambda r: r["ptid"])
    n = len(left)
    times = np.array([r["outcome_months"] - r["landmark_months"] for r in left])
    events = np.array([r["event"] for r in left], dtype=bool) & (times <= horizon)
    observed = np.minimum(times, horizon)
    first, second = np.where(events[:, None] & (observed[:, None] < observed[None, :]))
    risks = [1 - np.array([r["survival"][horizon_index] for r in rows]) for rows in (left, right)]
    sparse = n < 30 or int(events.sum()) < 5
    result = {
        "patients": n,
        "events_within_horizon": int(events.sum()),
        "comparable_pairs": len(first),
        "status": "descriptive_sparse" if sparse else "estimated",
        "bootstrap_samples_requested": repeats,
        "bootstrap_samples_used": 0,
        "reference_c_index": None,
        "candidate_c_index": None,
        "difference": None,
        "confidence_interval": None,
        "reference_interval": None,
        "candidate_interval": None,
        "nll_difference": None,
        "nll_interval": None,
        "nll_bootstrap_samples_used": 0,
    }
    counts = resample_counts(n, repeats, seed) if not sparse else np.empty((0, n))
    if len(first):
        credits = [(r[first] > r[second]).astype(float) + 0.5 * (r[first] == r[second]) for r in risks]
        scores = [float(c.mean()) for c in credits]
        result.update(reference_c_index=scores[0], candidate_c_index=scores[1], difference=scores[1] - scores[0])
        if len(counts):
            samples = []
            for count in counts:
                weights = count[first] * count[second]
                if weights.sum():
                    samples.append([float(np.dot(weights, credit) / weights.sum()) for credit in credits])
            if samples:
                samples = np.array(samples)
                result.update(
                    reference_interval=interval(samples[:, 0]),
                    candidate_interval=interval(samples[:, 1]),
                    confidence_interval=interval(samples[:, 1] - samples[:, 0]),
                    bootstrap_samples_used=len(samples),
                )
            else:
                result["status"] = "no_estimable_bootstrap_samples"
    else:
        result["status"] = "no_comparable_pairs"
    # Saved full-follow-up `nll` is deliberately never used for this endpoint.
    if (
        horizon == 60
        and n
        and all(
            r["landmark_months"] == 24
            and isinstance(r.get("horizon_nll"), (float, int))
            and math.isfinite(r["horizon_nll"])
            for rows in (left, right)
            for r in rows
        )
    ):
        delta = np.array([b["horizon_nll"] - a["horizon_nll"] for a, b in zip(left, right, strict=True)])
        result["nll_difference"] = float(delta.mean())
        if len(counts):
            result["nll_interval"] = interval(counts @ delta / n)
            result["nll_bootstrap_samples_used"] = len(counts)
    return result


def km_risk(times, events, horizon=60):
    """Return null when follow-up ends early with nonzero survival remaining."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    if not len(times):
        return None
    survival = 1.0
    for t in sorted(set(times[(times <= horizon) & events])):
        survival *= 1 - np.sum(events & (times == t)) / np.sum(times >= t)
    if np.max(times) < horizon and survival > 0:
        return None
    return float(1 - survival)


def calibration(rows, index, repeats=1000, seed=42):
    risks = 1 - np.array([r["survival"][index] for r in rows])
    if not len(risks):
        return []
    boundaries = np.unique(np.quantile(risks, [1 / 3, 2 / 3]))
    groups = np.searchsorted(boundaries, risks, side="left")
    results, rng = [], random.Random(seed)
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        times = np.array([rows[i]["outcome_months"] - 24 for i in indices])
        events = np.array([rows[i]["event"] for i in indices], bool)
        observed = km_risk(times, events)
        event_count = int(np.sum(events & (times <= 60)))
        sparse = len(indices) < 30 or event_count < 5
        samples = []
        if observed is not None and not sparse:
            for _ in range(repeats):
                draws = [rng.randrange(len(indices)) for _ in indices]
                value = km_risk(times[draws], events[draws])
                if value is not None:
                    samples.append(value)
        results.append(
            {
                "group": int(group + 1),
                "patients": len(indices),
                "events_within_horizon": event_count,
                "at_risk_at_horizon": int(np.sum(times >= 60)),
                "predicted_risk": float(risks[indices].mean()),
                "observed_risk": observed,
                "confidence_interval": interval(samples),
                "bootstrap_samples_used": len(samples),
                "boundaries": boundaries.tolist(),
                "status": "unsupported_followup"
                if observed is None
                else "descriptive_sparse"
                if sparse
                else "estimated"
                if samples
                else "no_estimable_bootstrap_samples",
            }
        )
    return results


def aggregate_components(rows):
    terms = ("cumulative_hazard", "event_log_hazard", "nll")
    patients = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r["ptid"], r["entry"], tuple(r["image_ids"]))
        if key in seen or not all(math.isfinite(r[t]) for t in terms):
            raise ValueError("Duplicate or nonfinite likelihood component")
        seen.add(key)
        if abs(r["nll"] - r["cumulative_hazard"] - r["event_log_hazard"]) > 1e-5:
            raise ValueError("Components do not reconstruct NLL")
        patients[r["ptid"]].append(r)
    return {
        "patients": len(patients),
        "observations": len(rows),
        **{
            t: float(np.mean([np.mean([r[t] for r in group]) for group in patients.values()]))
            if patients
            else None
            for t in terms
        },
    }


def audit(path, split="validation"):
    manifest, _, _, experiments, pending, files = load_manifest(path, split)
    return {
        "format_version": VERSION,
        "split": split,
        "primary": manifest["primary"],
        "completed": [
            {
                "role": e["role"],
                "seed": e["seed"],
                "checkpoint_sha256": e["checkpoint_sha256"],
                "primary_patients": len(selected(e["rows"])),
            }
            for e in experiments
        ],
        "pending": pending,
        "sources": files,
        "analysis_sha256": sha256(__file__),
    }


def primary_components(head, z, batch):
    """Use source precision for the inclusive event boundary, original months for H."""
    import torch

    if any(r.prediction_months != 24 for r in batch["records"]):
        raise ValueError("Primary likelihood requires month-24 observations")
    end = torch.minimum(batch["outcome"], batch["entry"] + 60)
    # Collators store float32 event indicators. Do not apply bitwise & to them,
    # or classify source times just beyond 84 after float32 rounding to 84.
    events = torch.tensor(
        [r.event and r.outcome_months <= 84 for r in batch["records"]],
        dtype=batch["outcome"].dtype,
        device=batch["outcome"].device,
    )
    return head.nll_components(z, batch["entry"], end, events)


def evaluate(path, split, device, output):
    """Only explicit inference entry point. Test uses existing locked predictions.

    Does not select/promote checkpoints, create model optimizers, or write into runs.
    """
    from dataclasses import replace

    import torch

    from mstat_project.ml.continuous_survival import ContinuousMRIModel
    from mstat_project.ml.ltsa_experiments import load_snapshot
    from mstat_project.ml.survival import SurvivalConfig, _device_batch, _loader, model_identity
    from mstat_project.ml.survival_data import at_landmark
    from mstat_project.ml.survival_embeddings import CachedEncoder, load_embeddings
    from mstat_project.ml.survival_protocol import implementation_hashes

    manifest, root, protocol, experiments, _, files = load_manifest(path, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, schedule, _ = load_snapshot(protocol)
    cache = path_for(root, manifest["cache"])
    payload, _ = load_embeddings(
        cache, provenance=protocol["snapshot_hashes"], spatial_size=protocol["reference_config"]["spatial_size"]
    )
    files[str(cache)] = sha256(cache)
    evaluated = {}
    for experiment in experiments:
        print(f"Evaluating {experiment['role']} seed {experiment['seed']}: {split}", flush=True)
        checkpoint = torch.load(experiment["checkpoint_path"], map_location="cpu", weights_only=False)
        config_values = dict(checkpoint["config"])
        input_mode = config_values.pop("input_mode", "full_history")
        config = replace(SurvivalConfig(**config_values), device=device, cache_path=cache, evaluate_test=False)
        if config.seed != experiment["seed"] or checkpoint["variant"] != experiment["summary_data"]["variant"]:
            raise ValueError("Manifest seed/variant differs from checkpoint configuration")
        if not config.freeze_encoder or config.fine_tune_final_block:
            raise ValueError(
                "This cached figure evaluator requires frozen encoders; raw fine-tuned models need a separate evaluator"
            )
        depth = 2 if experiment["role"] == "history-2l" else 1
        if experiment["role"] in ("history-1l", "history-2l", "latest-only-1l") and config.n_layers != depth:
            raise ValueError("Manifest role does not match checkpoint temporal depth")
        variant = checkpoint["variant"]
        if experiment["role"] == "historical-b":
            from mstat_project.ml.ltsa_architecture_ablation import EXPERIMENTS, AblationModel, load_saved_weights

            model = AblationModel(
                payload["manifest"],
                variant,
                EXPERIMENTS["legacy_all"],
                0.002,
                quadrature_points=config.quadrature_points,
            )
            load_saved_weights(model, checkpoint, "reference", payload["manifest"])
        else:
            if checkpoint.get("implementation_hashes") != implementation_hashes():
                raise ValueError("Checkpoint inference implementation changed")
            model = ContinuousMRIModel(
                CachedEncoder(payload["manifest"]),
                variant,
                freeze_encoder=True,
                quadrature_points=config.quadrature_points,
                n_layers=config.n_layers,
                d_model=config.d_model,
                dropout=config.dropout,
                step_ahead=config.auxiliary_weight > 0,
            )
            if checkpoint.get("identity") != model_identity(model, records, config):
                raise ValueError("Checkpoint/cache/snapshot identity differs")
            # Raw-encoder weights have already been verified by the shared identity.
            state = {k: v for k, v in checkpoint["model"].items() if not k.startswith("encoder.")}
            model.load_state_dict(state, strict=True)
        if experiment["role"] == "latest-only-1l" and input_mode != "latest_only":
            raise ValueError("Latest-only evaluation requires a checkpoint trained with that input mode")
        model.to(device).eval()
        full_rows, horizon_rows = [], []
        sets = [("full", [r for r in records if r.split == split]), ("horizon", at_landmark(records, 24, split))]
        with torch.inference_mode():
            for scope, observations in sets:
                for batch in _loader(observations, config, variant, schedule.height):
                    batch = _device_batch(batch, device)
                    if input_mode == "latest_only" and variant == "history":
                        slots = torch.arange(batch["observed"].shape[1], device=device).expand_as(
                            batch["observed"]
                        )
                        last = slots.masked_fill(~batch["observed"], -1).max(1).values
                        batch["observed"] = slots == last[:, None]
                    z = model.encode_batch(batch)
                    end = batch["outcome"]
                    events = batch["event"]
                    if scope == "horizon":
                        cumulative, event_log = primary_components(model.head, z, batch)
                    else:
                        cumulative, event_log = model.head.nll_components(z, batch["entry"], end, events)
                    for record, h, e in zip(
                        batch["records"], cumulative.cpu().tolist(), event_log.cpu().tolist(), strict=True
                    ):
                        (full_rows if scope == "full" else horizon_rows).append(
                            {
                                "ptid": record.ptid,
                                "entry": record.prediction_months,
                                "image_ids": list(record.image_ids),
                                "cumulative_hazard": h,
                                "event_log_hazard": e,
                                "nll": h + e,
                            }
                        )
                    if scope == "horizon":
                        survival = (
                            torch.stack(
                                [
                                    model.head.survival(z, batch["entry"], batch["entry"] + t)
                                    for t in experiment["horizons"]
                                ],
                                1,
                            )
                            .cpu()
                            .tolist()
                        )
                        saved = {r["ptid"]: r for r in selected(experiment["rows"])}
                        for record, values in zip(batch["records"], survival, strict=True):
                            saved_row = saved.get(record.ptid)
                            if (
                                saved_row is None
                                or record.outcome_months != saved_row["outcome_months"]
                                or record.event != saved_row["event"]
                                or list(record.image_ids) != saved_row["available_history_ids"]
                                or list(record.visit_nums) != saved_row["visit_nums"]
                                or record.acquisition_months[-1] != saved_row["acquisition_months"]
                                or not np.allclose(values, saved_row["survival"], atol=1e-6, rtol=1e-6)
                            ):
                                raise ValueError("Re-evaluated predictions differ from selected saved predictions")
        overall = aggregate_components(full_rows)
        expected = experiment["summary_data"]["validation_nll"]
        if experiment["role"] == "historical-b":
            # Explicit, verified replay accounts for batch-versus-singleton encoder numerics.
            replay_path = path_for(root, experiment["reference_replay"])
            replay = read_json(replay_path)
            if (
                not replay["arms"]["b"]["raw_batch"]["reproduced"]
                or abs(replay["arms"]["b"]["expected_nll"] - expected) > 1e-8
            ):
                raise ValueError("Historical cached reference requires the reproduced original replay")
            expected = replay["arms"]["b"]["cached"]["validation_nll_original_reduction"]
            files[str(replay_path)] = sha256(replay_path)
        if split == "validation" and abs(overall["nll"] - expected) > 1e-5:
            raise ValueError("Full patient-weighted validation NLL does not reproduce selected checkpoint")
        key = f"{experiment['role']}:{experiment['seed']}"
        evaluated[key] = {
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "full": overall,
            "horizon": aggregate_components(horizon_rows),
            "primary_rows": horizon_rows,
        }
    write_json(
        output / "evaluation.json",
        {
            "format_version": VERSION,
            "split": split,
            "sources": files,
            "analysis_sha256": sha256(__file__),
            "experiments": evaluated,
            "runtime": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "device": device,
            },
        },
    )
    print(f"Saved {output / 'evaluation.json'}", flush=True)


def attach_evaluation(experiments, evaluation_path, split, sources):
    if evaluation_path is None:
        return None
    evaluation = read_json(evaluation_path)
    if evaluation["split"] != split or evaluation["analysis_sha256"] != sha256(__file__):
        raise ValueError("Evaluation split or analysis implementation differs")
    for path, digest in evaluation["sources"].items():
        if sha256(path) != digest:
            raise ValueError("An evaluated source artifact changed")
    sources[str(Path(evaluation_path).resolve())] = sha256(evaluation_path)
    for experiment in experiments:
        key = f"{experiment['role']}:{experiment['seed']}"
        value = evaluation["experiments"].get(key)
        if value is None or value["checkpoint_sha256"] != experiment["checkpoint_sha256"]:
            raise ValueError("Evaluation lacks the selected experiment checkpoint")
        horizon = {r["ptid"]: r for r in value["primary_rows"]}
        if len(horizon) != len(value["primary_rows"]) or set(horizon) != {
            r["ptid"] for r in selected(experiment["rows"])
        }:
            raise ValueError("Evaluated primary cohort differs")
        aggregate_components(value["primary_rows"])
        for row in selected(experiment["rows"]):
            if (
                horizon[row["ptid"]]["entry"] != 24
                or horizon[row["ptid"]]["image_ids"] != row["available_history_ids"]
            ):
                raise ValueError("Evaluated primary history or entry differs")
            row["horizon_nll"] = horizon[row["ptid"]]["nll"]
        # selected returns the original dictionaries; the new field is local only.
    return evaluation


def analysis_tables(manifest, experiments):
    repeats, seed = manifest.get("bootstrap_samples", 1000), manifest.get("bootstrap_seed", 42)
    lookup = {(e["role"], e["seed"]): e for e in experiments}
    absolute, comparisons, strata, grid, calibration_rows, support = [], [], [], [], [], []
    for e in experiments:
        rows = selected(e["rows"])
        j = e["horizons"].index(60)
        score = paired_statistics(rows, rows, j, 60, repeats, seed)
        absolute.append({"role": e["role"], "seed": e["seed"], **score})
        for row in calibration(rows, j, repeats, seed):
            calibration_rows.append({"role": e["role"], "seed": e["seed"], **row})
    for training_seed in sorted({e["seed"] for e in experiments}):
        for reference, candidate, comparison in PAIRS:
            if (reference, training_seed) not in lookup or (candidate, training_seed) not in lookup:
                continue
            a, b = lookup[reference, training_seed], lookup[candidate, training_seed]
            prefix = {
                "reference": reference,
                "candidate": candidate,
                "comparison": comparison,
                "seed": training_seed,
            }
            for stratum in ("all", "1", "2", "3+", "2+"):
                row = {
                    **prefix,
                    "stratum": stratum,
                    **paired_statistics(
                        selected(a["rows"], stratum=stratum),
                        selected(b["rows"], stratum=stratum),
                        a["horizons"].index(60),
                        60,
                        repeats,
                        seed,
                    ),
                }
                strata.append(row)
                if stratum == "all":
                    comparisons.append(row)
        for reference in ("b", "b-continuation"):
            for candidate in ("history-1l", "history-2l"):
                if (reference, training_seed) not in lookup or (candidate, training_seed) not in lookup:
                    continue
                a, b = lookup[reference, training_seed], lookup[candidate, training_seed]
                for landmark in sorted({r["landmark_months"] for r in a["rows"]}):
                    for j, horizon in enumerate(a["horizons"]):
                        grid.append(
                            {
                                "reference": reference,
                                "candidate": candidate,
                                "seed": training_seed,
                                "landmark": landmark,
                                "horizon": horizon,
                                **paired_statistics(
                                    selected(a["rows"], landmark),
                                    selected(b["rows"], landmark),
                                    j,
                                    horizon,
                                    repeats,
                                    seed,
                                ),
                            }
                        )
    primary = next((e for e in experiments if e["role"] == "b" and e["seed"] == 42), experiments[0])
    for stratum in ("1", "2", "3+", "all"):
        rows = selected(primary["rows"], stratum=stratum)
        times = [r["outcome_months"] - 24 for r in rows]
        support.append(
            {
                "stratum": stratum,
                "patients": len(rows),
                "events_within_horizon": sum(r["event"] and t <= 60 for r, t in zip(rows, times)),
                "censored_before_horizon": sum(not r["event"] and t < 60 for r, t in zip(rows, times)),
                "event_free_through_horizon": sum(
                    t > 60 or (not r["event"] and t == 60) for r, t in zip(rows, times)
                ),
            }
        )
    cross_seed = []
    # Require the planned matched seeds. Same patients are drawn jointly across seeds.
    for reference, candidate, name in PAIRS:
        if not all((role, s) in lookup for role in (reference, candidate) for s in (42, 43, 44)):
            continue
        panels = [
            (selected(lookup[reference, s]["rows"]), selected(lookup[candidate, s]["rows"])) for s in (42, 43, 44)
        ]
        n = len(panels[0][0])
        counts = resample_counts(n, repeats, seed)
        samples, nll_samples = [], []
        point = []
        nll_deltas = []
        for a, b in panels:
            point.append(
                paired_statistics(a, b, lookup[reference, 42]["horizons"].index(60), 60, 1, seed)["difference"]
            )
            if all(r.get("horizon_nll") is not None for rows in (a, b) for r in rows):
                nll_deltas.append(
                    np.array([y["horizon_nll"] - x["horizon_nll"] for x, y in zip(a, b, strict=True)])
                )
        for count in counts:
            indices = np.repeat(np.arange(n), count.astype(int))
            differences = []
            for a, b in panels:
                # Synthetic draw keys preserve duplicates as independent draws of whole patients.
                aa = [{**a[i], "ptid": str(k)} for k, i in enumerate(indices)]
                bb = [{**b[i], "ptid": str(k)} for k, i in enumerate(indices)]
                difference = paired_statistics(aa, bb, lookup[reference, 42]["horizons"].index(60), 60, 0, seed)[
                    "difference"
                ]
                if difference is not None:
                    differences.append(difference)
            if len(differences) == 3:
                samples.append(float(np.mean(differences)))
            if len(nll_deltas) == 3:
                nll_samples.append(float(np.mean([np.dot(count, delta) / n for delta in nll_deltas])))
        cross_seed.append(
            {
                "reference": reference,
                "candidate": candidate,
                "comparison": name,
                "difference": float(np.mean(point)) if all(x is not None for x in point) else None,
                "confidence_interval": interval(samples),
                "bootstrap_samples_used": len(samples),
                "nll_difference": float(np.mean(nll_deltas)) if len(nll_deltas) == 3 else None,
                "nll_interval": interval(nll_samples),
                "nll_bootstrap_samples_used": len(nll_samples),
            }
        )
    return {
        "absolute": absolute,
        "comparisons": comparisons,
        "strata": strata,
        "grid": grid,
        "calibration": calibration_rows,
        "support": support,
        "cross_seed": cross_seed,
    }


def export_csv(path, rows):
    if not rows:
        path.write_text("status\npending\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows
        )


def pending_panel(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def forest(ax, rows, *, value="difference", bounds="confidence_interval", labels=None, zero=True):
    if not rows:
        pending_panel(ax, "No completed matched experiment results")
        return
    names = labels or [r.get("comparison", r.get("role", "")) for r in rows]
    for i, (r, label) in enumerate(zip(rows, names, strict=True)):
        v = r.get(value)
        if v is None:
            message = (
                "Failed experiment (see captions)"
                if r.get("status") == "failed"
                else "Pending matched experiments"
                if r.get("status") == "pending"
                else "Not estimable"
                if value != "nll_difference"
                else "Pending checkpoint evaluation"
            )
            ax.text(0.02, i, message, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
            continue
        color = COLORS.get(r.get("candidate", r.get("role")), "#263d77")
        limits = r.get(bounds)
        if limits is not None:
            ax.plot(limits, [i, i], color=color, lw=1.5)
        ax.plot(
            v,
            i,
            marker=MARKERS.get(r.get("candidate", r.get("role")), "o"),
            color=color,
            markerfacecolor="white" if r.get("status") == "descriptive_sparse" else color,
            ms=6,
        )
        ax.annotate(
            f"{v:+.5f}" if zero else f"{v:.4f}", (v, i), xytext=(4, 6), textcoords="offset points", fontsize=7
        )
    if zero:
        ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.set_yticks(range(len(rows)), names)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.margins(x=0.2)


def render(path, split, output, evaluation_path=None):
    # No model construction or torch.load is permitted in this function.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    manifest, root, _, experiments, pending, sources = load_manifest(path, split)
    evaluation = attach_evaluation(experiments, evaluation_path, split, sources)
    tables = analysis_tables(manifest, experiments)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    edition = "Exploratory validation" if split == "validation" else "Locked held-out test"
    lookup = {(e["role"], e["seed"]): e for e in experiments}
    output_files = {}
    learning_rows = []

    def save(fig, name, title):
        fig.suptitle(f"{title}\n{edition} · primary question: month 24 / next 60 months", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93), pad=2)
        for extension in ("pdf", "svg", "png"):
            target = output / f"{name}.{extension}"
            fig.savefig(target, bbox_inches="tight", dpi=300)
            output_files[target.name] = sha256(target)
        plt.close(fig)

    def curves(ax):
        for role in ("b-continuation", "history-1l", "history-2l"):
            e = lookup.get((role, 42))
            if e is None or not (e["directory"] / "epochs.json").exists():
                continue
            rows = read_json(e["directory"] / "epochs.json")
            learning_rows.extend({"role": role, "seed": 42, **r} for r in rows)
            ax.plot(
                [r["epoch"] for r in rows],
                [r["validation_nll"] for r in rows],
                color=COLORS[role],
                marker=MARKERS[role],
                label=LABELS[role],
                ms=4,
            )
            best = e["summary_data"]["best_epoch"]
            selected_row = next(r for r in rows if r["epoch"] == best)
            ax.plot(best, selected_row["validation_nll"], marker="o", ms=10, mfc="none", mec=COLORS[role])
        if ("b", 42) in lookup:
            ax.axhline(
                lookup["b", 42]["summary_data"]["validation_nll"], ls="--", color=COLORS["b"], label="Original B"
            )
        ax.set(xlabel="Continuation epoch", ylabel="Full-follow-up validation NLL (nats)")
        ax.legend(fontsize=7)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 1.5, 0.8, 1.2]})
    ax = axes[0]
    ax.axvspan(0, 24, color="#e8edf5")
    ax.axvspan(24, 84, color="#f6ecd8")
    ax.axvline(24, color="#333333", ls="--")
    ax.text(12, 0.65, "Eligible MRI prefix\nthrough month 24", ha="center")
    ax.text(54, 0.65, "Predict events over the next 60 months", ha="center")
    ax.text(24, 0.12, "Prediction landmark", ha="center")
    ax.set(xlim=(0, 84), ylim=(0, 1), xticks=[0, 24, 84], yticks=[], xlabel="Months from baseline")
    schematic = [
        "Shared frozen ResNet encoder → trainable projection + acquisition time",
        "B: latest MRI → survival head",
        "Continued B: same B weights → fresh optimizer → survival head",
        "LTSA: B initialization + causal temporal correction → survival head",
        "Latest-only LTSA: same architecture/training; earlier inputs masked "
        + ("(pending)" if ("latest-only-1l", 42) not in lookup else ""),
    ]
    axes[1].text(0.02, 0.93, "\n\n".join(schematic), transform=axes[1].transAxes, va="top", fontsize=9)
    axes[1].axis("off")
    cohort_flow = []
    axes[2].axis("off")
    if manifest.get("cohort_audit"):
        cohort = read_json(path_for(root, manifest["cohort_audit"]))["cohort"]
        split_cohort = cohort["splits"][split]
        lines = [
            f"Source MRI inventory: {cohort['source_images']:,} images → {cohort['eligible_images']:,} eligible images ({cohort['eligible_patients']:,} patients)",
            "Image exclusions: "
            + "; ".join(f"{r['exclusion_reason'].replace('_', ' ')}: {r['len']}" for r in cohort["exclusions"]),
            f"Unassigned images: {cohort['unassigned_images']}; {split} eligible: {split_cohort['patients']} patients / {split_cohort['images']} images → month-24 cohort: {tables['support'][-1]['patients']} patients",
        ]
        axes[2].text(0.02, 0.85, "\n\n".join(lines), transform=axes[2].transAxes, va="top", fontsize=8)
        cohort_flow = [
            {"quantity": key, "unit": "images" if "images" in key else "patients", "count": cohort[key]}
            for key in ("source_images", "eligible_images", "eligible_patients", "unassigned_images")
        ]
        cohort_flow += [
            {"quantity": r["exclusion_reason"], "unit": "images", "count": r["len"]} for r in cohort["exclusions"]
        ]
    else:
        axes[2].text(
            0.02, 0.5, "Exclusion-flow counts not supplied; no counts inferred.", transform=axes[2].transAxes
        )
    support = tables["support"][:3]
    x = np.arange(3)
    ax = axes[3]
    event = np.array([r["events_within_horizon"] for r in support])
    censor = np.array([r["censored_before_horizon"] for r in support])
    total = np.array([r["patients"] for r in support])
    ax.bar(x, event, label="Events by 60 months", color="#263d77")
    ax.bar(x, censor, bottom=event, label="Censored before 60 months", color="#b87b26", hatch="//")
    ax.bar(x, total - event - censor, bottom=event + censor, label="Event-free through 60 months", color="#cccccc")
    for i, n in enumerate(total):
        ax.text(i, n + 1, f"N={n}; events={event[i]}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=["1 MRI", "2 MRIs", "≥3 MRIs"], ylabel="Patients")
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(total, default=1) * 1.25)
    save(fig, "figure_1", "Study design and longitudinal information")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    absolute = [
        r
        for r in tables["absolute"]
        if r["seed"] == 42 and r["role"] in ("b", "b-continuation", "history-1l", "history-2l", "latest-only-1l")
    ]
    rows = [r for r in tables["comparisons"] if r["seed"] == 42]
    names = [f"{r['comparison']}\n{LABELS[r['candidate']]} − {LABELS[r['reference']]}" for r in rows]
    forest(
        axes[0],
        absolute,
        value="candidate_c_index",
        bounds="candidate_interval",
        labels=[LABELS[r["role"]] for r in absolute],
        zero=False,
    )
    axes[0].set_title("Absolute discrimination")
    axes[0].set_xlabel("C-index · higher is better")
    forest(axes[1], rows, labels=names)
    axes[1].set(title="Paired discrimination differences", xlabel="Δ C-index · positive favors candidate")
    forest(axes[2], rows, value="nll_difference", bounds="nll_interval", labels=names)
    axes[2].set(
        title="Paired 60-month likelihood differences",
        xlabel="Δ conditional NLL (nats) · negative favors candidate",
    )
    save(fig, "figure_2", "Primary prediction performance")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    curves(axes[0])
    for ax, comparison in zip(axes[1:], ("Training-controlled", "Direct history"), strict=True):
        rows = [r for r in tables["comparisons"] if r["comparison"] == comparison]
        display = [
            {"seed": s, "difference": None, "status": "pending"}
            if not any(r["seed"] == s for r in rows)
            else next(r for r in rows if r["seed"] == s)
            for s in (42, 43, 44)
        ]
        relevant = (
            ("b-continuation", "history-1l")
            if comparison == "Training-controlled"
            else ("latest-only-1l", "history-1l")
        )
        for row in display:
            if row.get("status") == "pending" and any(
                r["seed"] == row["seed"] and r["role"] in relevant and r.get("status") == "failed" for r in pending
            ):
                row["status"] = "failed"
        labels = [f"Seed {r['seed']}" for r in display]
        summary = next((r for r in tables["cross_seed"] if r["comparison"] == comparison), None)
        if summary:
            display.append(summary)
            labels.append("Mean across seeds 42/43/44")
        forest(ax, display, labels=labels)
        ax.set(title=comparison, xlabel="Δ C-index · positive favors full-history LTSA-1L")
    save(fig, "figure_3", "Training, architecture and history controls")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    stratum_labels = {
        "all": "All patients",
        "1": "1 MRI",
        "2": "2 MRIs",
        "3+": "≥3 MRIs",
        "2+": "≥2 MRIs (secondary)",
    }
    for row_axes, comparison in zip(
        axes, ("Training-controlled", "Direct history", "Secondary depth"), strict=True
    ):
        rows = [r for r in tables["strata"] if r["seed"] == 42 and r["comparison"] == comparison]
        names = [
            f"{stratum_labels[r['stratum']]}: N={r['patients']}, events={r['events_within_horizon']}, pairs={r['comparable_pairs']}\n"
            f"C bootstrap={r['bootstrap_samples_used']}; NLL bootstrap={r['nll_bootstrap_samples_used']}"
            for r in rows
        ]
        forest(row_axes[0], rows, labels=names)
        forest(row_axes[1], rows, value="nll_difference", bounds="nll_interval", labels=names)
        row_axes[0].set(title=comparison, xlabel="Δ C-index · positive favors candidate")
        row_axes[1].set(title=comparison, xlabel="Δ 60-month NLL (nats) · negative favors candidate")
    save(fig, "figure_4", "Performance by available MRI history")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, role in zip(
        axes.flat,
        ("b", "b-continuation", "history-1l", "history-2l", "latest-only-1l", "historical-b"),
        strict=True,
    ):
        rows = [r for r in tables["calibration"] if r["role"] == role and r["seed"] == 42]
        ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=1)
        ax.set(
            title=LABELS[role],
            xlabel="Mean predicted event probability",
            ylabel="Observed event probability",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        if not rows:
            ax.text(0.1, 0.8, "No completed control", transform=ax.transAxes)
        for row in rows:
            v = row["observed_risk"]
            if v is None:
                ax.text(
                    0.05,
                    0.1 + 0.08 * row["group"],
                    f"Group {row['group']}: unsupported follow-up",
                    transform=ax.transAxes,
                    fontsize=7,
                )
                continue
            ci = row["confidence_interval"]
            if ci:
                ax.plot([row["predicted_risk"]] * 2, ci, color=COLORS[role])
            ax.plot(
                row["predicted_risk"],
                v,
                MARKERS[role],
                color=COLORS[role],
                mfc="white" if row["status"] == "descriptive_sparse" else COLORS[role],
            )
            ax.annotate(
                f"N={row['patients']}, E={row['events_within_horizon']}",
                (row["predicted_risk"], v),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=7,
            )
    save(fig, "figure_5", "Calibration of predicted probabilities")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    available = [r for r in tables["grid"] if r["seed"] == 42 and r["difference"] is not None]
    maximum = max((abs(r["difference"]) for r in available), default=0.01) or 0.01
    for ax, (reference, candidate) in zip(
        axes.flat,
        (
            ("b", "history-1l"),
            ("b-continuation", "history-1l"),
            ("b", "history-2l"),
            ("b-continuation", "history-2l"),
        ),
        strict=True,
    ):
        rows = [
            r
            for r in tables["grid"]
            if r["reference"] == reference and r["candidate"] == candidate and r["seed"] == 42
        ]
        if not rows:
            pending_panel(ax, "Pending matched predictions")
            continue
        landmarks, horizons = sorted({r["landmark"] for r in rows}), sorted({r["horizon"] for r in rows})
        matrix = np.full((len(landmarks), len(horizons)), np.nan)
        for r in rows:
            y, x = landmarks.index(r["landmark"]), horizons.index(r["horizon"])
            matrix[y, x] = r["difference"] if r["difference"] is not None else np.nan
        im = ax.imshow(matrix, cmap="PuOr", vmin=-maximum, vmax=maximum, aspect="auto")
        for r in rows:
            y, x = landmarks.index(r["landmark"]), horizons.index(r["horizon"])
            text = (
                "×"
                if r["difference"] is None
                else f"{r['difference']:+.3f}" + ("*" if r["status"] == "descriptive_sparse" else "")
            )
            ax.text(
                x,
                y,
                text + f"\nN={r['patients']}, E={r['events_within_horizon']}",
                ha="center",
                va="center",
                fontsize=6,
            )
            if r["landmark"] == 24 and r["horizon"] == 60:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=2, ec="black"))
        ax.set(
            xticks=range(len(horizons)),
            xticklabels=horizons,
            yticks=range(len(landmarks)),
            yticklabels=landmarks,
            title=f"{LABELS[candidate]} − {LABELS[reference]}",
            xlabel="Horizon (months)",
            ylabel="Landmark (months)",
        )
        fig.colorbar(im, ax=ax, shrink=0.7, label="Δ C-index")
    save(fig, "supplement_1", "Secondary endpoint grid (* descriptive sparse)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    component_tables = []
    for ax, scope, title in zip(
        axes[:2], ("full", "horizon"), ("Full-validation, full follow-up", "Primary 60-month window"), strict=True
    ):
        if evaluation is None:
            pending_panel(ax, "Pending explicit likelihood evaluation")
            continue
        rows = []
        for e in experiments:
            if e["seed"] == 42:
                value = evaluation["experiments"][f"{e['role']}:42"][scope]
                rows.append({"role": e["role"], "scope": scope, **value})
        component_tables.extend(rows)
        y = np.arange(len(rows))
        ax.barh(
            y - 0.15,
            [r["cumulative_hazard"] for r in rows],
            height=0.3,
            label="Cumulative hazard",
            color="#52758c",
        )
        ax.barh(
            y + 0.15,
            [r["event_log_hazard"] for r in rows],
            height=0.3,
            label="Negative event-log hazard",
            color="#b87b26",
        )
        ax.set(
            yticks=y,
            yticklabels=[LABELS[r["role"]] for r in rows],
            xlabel="Patient-weighted NLL contribution (nats)",
            title=title,
        )
        ax.legend(fontsize=7)
    architecture_rows = []
    if manifest.get("architecture_report"):
        architecture = read_json(path_for(root, manifest["architecture_report"]))
        architecture_rows = [
            {
                "arm": r["arm"],
                "experiment": r["experiment"],
                "validation_nll": r["validation_nll"],
                **r["components"],
            }
            for r in architecture["results"]
        ]
        for arm, marker in (("a", "s"), ("b", "o")):
            rows = [r for r in architecture_rows if r["arm"] == arm]
            axes[2].plot(
                [r["validation_nll"] for r in rows], [r["experiment"] for r in rows], marker, label=arm.upper()
            )
        axes[2].set(xlabel="Full-follow-up validation NLL (nats)", title="Shared-architecture investigation")
        axes[2].legend()
    else:
        pending_panel(axes[2], "No architecture investigation supplied")
    save(fig, "supplement_2", "Likelihood components and architecture investigation")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    optimization_rows, diagnostic_rows, parameter_rows = [], [], []
    for role in ("b-continuation", "history-1l", "history-2l"):
        e = lookup.get((role, 42))
        if e is None:
            continue
        for source in sorted(e["directory"].glob("optimization_epoch_*.json")):
            epoch = int(source.stem.rsplit("_", 1)[1])
            batches = read_json(source)
            for component in ("projection", "survival_head", "attention", "feed_forward", "residual_gate"):
                values = [r["update_norms"].get(component) for r in batches if "update_norms" in r]
                values = [v for v in values if v is not None and math.isfinite(v)]
                if values:
                    optimization_rows.append(
                        {
                            "role": role,
                            "epoch": epoch,
                            "component": component,
                            "median_update_norm": float(np.median(values)),
                            "min_update_norm": min(values),
                            "max_update_norm": max(values),
                        }
                    )
        for source in sorted(e["directory"].glob("diagnostics_epoch_*.json")):
            value = read_json(source)
            epoch = int(source.stem.rsplit("_", 1)[1])
            variance = value.get("risk_variance_by_horizon", [])
            diagnostic_rows.append(
                {
                    "role": role,
                    "epoch": epoch,
                    "risk_variance": variance[e["horizons"].index(60)] if variance else None,
                    "warnings": value.get("warnings", []),
                    "common_query_months": value.get("hazard_query_months"),
                    "risk_near_constant": value.get("panel_variation", {}).get("risk", {}).get("near_constant"),
                }
            )
        counts_path = e["directory"] / "parameter_counts.json"
        if counts_path.exists():
            for component, counts in read_json(counts_path).items():
                parameter_rows.append({"role": role, "component": component, **counts})
    styles = {
        "projection": "-",
        "survival_head": "--",
        "attention": ":",
        "feed_forward": "-.",
        "residual_gate": (0, (1, 1)),
    }
    for role in ("b-continuation", "history-1l", "history-2l"):
        for component, style in styles.items():
            rows = [r for r in optimization_rows if r["role"] == role and r["component"] == component]
            if rows:
                axes[0, 0].plot(
                    [r["epoch"] for r in rows],
                    [r["median_update_norm"] for r in rows],
                    color=COLORS[role],
                    ls=style,
                    label=f"{LABELS[role]}: {component}",
                )
        rows = [r for r in diagnostic_rows if r["role"] == role and r["risk_variance"] is not None]
        if rows:
            axes[0, 1].plot(
                [r["epoch"] for r in rows],
                [r["risk_variance"] for r in rows],
                marker=MARKERS[role],
                color=COLORS[role],
                label=LABELS[role],
            )
    axes[0, 0].set(
        xlabel="Epoch",
        ylabel="Median minibatch optimizer update norm",
        title="Actual updates (zero remains visible)",
    )
    axes[0, 0].set_yscale("symlog", linthresh=1e-6)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], color=COLORS[r], label=LABELS[r]) for r in ("b-continuation", "history-1l", "history-2l")
    ]
    legend += [Line2D([0], [0], color="#555555", ls=style, label=component) for component, style in styles.items()]
    axes[0, 0].legend(handles=legend, fontsize=7, ncol=2)
    axes[0, 1].set(xlabel="Epoch", ylabel="Fixed-panel 60-month risk variance", title="Fixed diagnostic panel")
    axes[0, 1].legend(fontsize=7)
    scatter_bins = []
    if ("b-continuation", 42) in lookup:
        reference = selected(lookup["b-continuation", 42]["rows"])
        j = lookup["b-continuation", 42]["horizons"].index(60)
        for role in ("history-1l", "history-2l"):
            if (role, 42) in lookup:
                candidate = selected(lookup[role, 42]["rows"])
                x = [1 - r["survival"][j] for r in reference]
                y = [1 - r["survival"][j] for r in candidate]
                axes[1, 0].scatter(
                    x, y, color=COLORS[role], marker=MARKERS[role], s=12, alpha=0.6, label=LABELS[role]
                )
                bins, edges_x, edges_y = np.histogram2d(x, y, bins=20, range=((0, 1), (0, 1)))
                for xi, yi in zip(*np.where(bins > 0), strict=True):
                    scatter_bins.append(
                        {
                            "role": role,
                            "reference_risk_low": float(edges_x[xi]),
                            "reference_risk_high": float(edges_x[xi + 1]),
                            "candidate_risk_low": float(edges_y[yi]),
                            "candidate_risk_high": float(edges_y[yi + 1]),
                            "patients": int(bins[xi, yi]),
                        }
                    )
        axes[1, 0].plot([0, 1], [0, 1], color="#555555", ls="--")
    axes[1, 0].set(
        xlabel="Continued B event probability",
        ylabel="LTSA event probability",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Paired primary-cohort predictions",
    )
    axes[1, 0].legend(fontsize=7)
    lines = []
    for role in ("b-continuation", "history-1l", "history-2l"):
        counts = [r for r in parameter_rows if r["role"] == role]
        diagnostics = [r for r in diagnostic_rows if r["role"] == role]
        warnings = [r for r in diagnostic_rows if r["role"] == role and (r["warnings"] or r["risk_near_constant"])]
        parameter_label = f"{sum(r['trainable'] for r in counts):,}" if counts else "not reported"
        warning_label = str(len(warnings)) if diagnostics else "not reported"
        lines.append(
            f"{LABELS[role]}: {parameter_label} trainable parameters\n"
            f"Diagnostic epochs with warnings/constant risk: {warning_label}"
        )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "\n\n".join(lines) + "\n\nWarnings are screening flags, not causal explanations.",
        va="top",
        transform=axes[1, 1].transAxes,
    )
    save(fig, "supplement_3", "Optimization and prediction diagnostics")

    tables.update(
        components=component_tables,
        architecture=architecture_rows,
        cohort_flow=cohort_flow,
        learning_curves=learning_rows,
        optimization=optimization_rows,
        diagnostics=diagnostic_rows,
        parameters=parameter_rows,
        paired_risk_bins=scatter_bins,
    )
    for name, rows in tables.items():
        export_csv(output / f"source_{name}.csv", rows)
    captions = "# Scientific figure captions\n\n" + f"Edition: {edition}.\n\n"
    captions += (
        "All validation findings are exploratory. No practical-effect margin or equivalence claim is used.\n\n"
    )
    for name, caption in CAPTIONS.items():
        captions += f"## {name}\n\n{caption}\n\n"
    captions += "## Unavailable experiments\n\n" + "\n".join(
        f"- {LABELS[r['role']]}: seed {r['seed']}, {r.get('status', 'pending')}"
        + (f" — {r['reason']}" if r.get("reason") else "")
        for r in pending
    )
    if evaluation is None:
        captions += (
            "\n\nPrimary-window likelihood and full-cohort component panels await explicit checkpoint evaluation."
        )
    (output / "captions.md").write_text(captions, encoding="utf-8")
    write_json(
        output / "provenance.json",
        {
            "format_version": VERSION,
            "split": split,
            "edition": edition,
            "analysis_sha256": sha256(__file__),
            "sources": sources,
            "outputs": output_files,
            "pending": pending,
            "bootstrap_samples": manifest.get("bootstrap_samples", 1000),
            "bootstrap_seed": manifest.get("bootstrap_seed", 42),
            "likelihood_evaluated": evaluation is not None,
            "runtime": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "interpretation": "Effect estimates and uncertainty; no equivalence or meaningful-benefit threshold claim.",
        },
    )
    print(f"Saved eight figures and aggregate source tables to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "evaluate", "render"):
        p = sub.add_parser(command)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--split", choices=("validation", "test"), default="validation")
        if command != "audit":
            p.add_argument("--output", type=Path, required=True)
        if command == "evaluate":
            p.add_argument("--device", default="cpu")
        if command == "render":
            p.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.manifest, args.split), indent=2))
    elif args.command == "evaluate":
        evaluate(args.manifest, args.split, args.device, args.output)
    else:
        render(args.manifest, args.split, args.output, args.evaluation)


if __name__ == "__main__":
    main()
