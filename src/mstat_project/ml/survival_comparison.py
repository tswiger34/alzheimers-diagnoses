"""Paired evaluation of continuous-survival model predictions.

Compare only identical patients, outcomes and available MRI histories at a fixed
landmark. Bootstrap whole patients within that landmark; never pool repeated
observations from several landmarks as independent patients.
"""

import math
import random
from itertools import combinations

import torch

from mstat_project.ml.metrics import paired_c_index_difference
from mstat_project.ml.utils import concordance_index


def compare_predictions(predictions_by_variant, horizons, *, bootstrap_samples=1000, seed=42,
                        pairs=None, expected_split="test"):
    """Return B-minus-A and, when present, history-minus-each-baseline C-indices.

    Horizons are measured forward from the common prediction landmark. Outcomes
    are administratively censored there for both the point estimate and bootstrap.
    Non-estimable comparisons have null scores/intervals and an explicit status.
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    indexed = {}
    for variant, predictions in predictions_by_variant.items():
        rows = {}
        for p in predictions:
            key = (float(p["landmark_months"]), p["ptid"])
            if key in rows:
                raise ValueError(f"Duplicate patient/landmark prediction in {variant}")
            if len(p["survival"]) != len(horizons):
                raise ValueError("Survival predictions and horizons do not match")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p["survival"]):
                raise ValueError("Invalid survival probability")
            rows[key] = p
        indexed[variant] = rows
    pairs = pairs if pairs is not None else [("a", "b"), ("a", "history"), ("b", "history")]
    results = []
    for reference, candidate in pairs:
        if reference not in indexed or candidate not in indexed:
            continue
        left, right = indexed[reference], indexed[candidate]
        if left.keys() != right.keys():
            raise ValueError(f"Patient/landmark mismatch between {reference} and {candidate}")
        fields = ("split", "outcome_months", "event", "acquisition_months", "available_history_ids", "visit_nums")
        for key in left:
            if any(left[key][f] != right[key][f] for f in fields):
                raise ValueError(f"Outcome or MRI-history mismatch between {reference} and {candidate}")
            if left[key]["split"] != expected_split:
                raise ValueError(f"Paired reporting expects {expected_split} predictions")
        for landmark in sorted({key[0] for key in left}):
            keys = sorted(key for key in left if key[0] == landmark)
            times = torch.tensor([left[k]["outcome_months"] - landmark for k in keys], dtype=torch.float64)
            events = torch.tensor([left[k]["event"] for k in keys], dtype=torch.bool)
            if not torch.isfinite(times).all() or (times <= 0).any():
                raise ValueError("Test patients must be at risk at the prediction landmark")
            for j, horizon in enumerate(horizons):
                reference_risk = torch.tensor([1 - left[k]["survival"][j] for k in keys], dtype=torch.float64)
                candidate_risk = torch.tensor([1 - right[k]["survival"][j] for k in keys], dtype=torch.float64)
                observed = times.clamp_max(horizon)
                observed_events = events & (times <= horizon)
                baseline_c = concordance_index(reference_risk, observed, observed_events)
                candidate_c = concordance_index(candidate_risk, observed, observed_events)
                comparable = int((observed_events[:, None] & (observed[:, None] < observed[None, :])).sum())
                result = {
                    "reference": reference, "candidate": candidate,
                    "landmark_months": landmark, "horizon_months": float(horizon),
                    "patients": len(keys), "events_within_horizon": int(observed_events.sum()),
                    "comparable_pairs": comparable,
                    "reference_c_index": None, "candidate_c_index": None,
                    "difference": None, "confidence_interval_low": None, "confidence_interval_high": None,
                    "bootstrap_samples_requested": bootstrap_samples, "bootstrap_samples_used": 0,
                    "status": "no_comparable_pairs",
                }
                if math.isfinite(baseline_c) and math.isfinite(candidate_c):
                    result.update(reference_c_index=baseline_c, candidate_c_index=candidate_c,
                                  difference=candidate_c - baseline_c, status="point_estimate_only")
                    if bootstrap_samples:
                        try:
                            paired = paired_c_index_difference(
                                reference_risk, candidate_risk, observed, observed_events,
                                bootstrap_samples=bootstrap_samples, seed=seed,
                            )
                        except ValueError as exc:
                            if str(exc) != "No bootstrap sample contained comparable survival pairs":
                                raise
                            result["status"] = "no_estimable_bootstrap_samples"
                        else:
                            result.update(
                                confidence_interval_low=paired.confidence_interval_low,
                                confidence_interval_high=paired.confidence_interval_high,
                                bootstrap_samples_used=paired.bootstrap_samples,
                                status="estimated",
                            )
                results.append(result)
    return results


def compare_experiments(predictions, horizons, *, bootstrap_samples=1000, seed=42, expected_split="test"):
    """Named experiments, paired patient strata, and individual score intervals."""
    pairs = list(combinations(predictions, 2))
    # Validate the complete cohorts before stratification could hide mismatches.
    compare_predictions(predictions, horizons, bootstrap_samples=0, pairs=pairs, expected_split=expected_split)
    results = []
    for stratum in ("all", "1", "2", "3+"):
        selected = {name: [r for r in rows if stratum == "all" or
                    ("3+" if len(r["available_history_ids"]) >= 3 else str(len(r["available_history_ids"]))) == stratum]
                    for name, rows in predictions.items()}
        comparisons = compare_predictions(selected, horizons, bootstrap_samples=0, pairs=pairs,
                                          expected_split=expected_split)
        for result in comparisons:
            result["mri_count_stratum"] = stratum
            result["primary"] = (result["reference"] == "b" and result["candidate"] == "history-1l"
                                 and result["landmark_months"] == 24 and result["horizon_months"] == 60
                                 and stratum == "all")
            result["reference_confidence_interval"] = None
            result["candidate_confidence_interval"] = None
            for role in ("reference", "candidate"):
                rows = [r for r in selected[result[role]] if r["landmark_months"] == result["landmark_months"]]
                nll = [r.get("nll") for r in rows]
                result[f"{role}_mean_nll"] = (sum(nll) / len(nll) if nll and
                    all(isinstance(v, (int, float)) and math.isfinite(v) for v in nll) else None)
            result["bootstrap_samples_requested"] = bootstrap_samples
            if result["patients"] < 30 or result["events_within_horizon"] < 5:
                if result["difference"] is not None:
                    result["status"] = "descriptive_sparse"
                results.append(result)
                continue
            if result["difference"] is None or not bootstrap_samples:
                results.append(result)
                continue
            landmark, horizon = result["landmark_months"], result["horizon_months"]
            j = list(horizons).index(horizon)
            left = sorted((r for r in selected[result["reference"]] if r["landmark_months"] == landmark),
                          key=lambda r: r["ptid"])
            right = sorted((r for r in selected[result["candidate"]] if r["landmark_months"] == landmark),
                           key=lambda r: r["ptid"])
            times = torch.tensor([r["outcome_months"] - landmark for r in left], dtype=torch.float64)
            events = torch.tensor([r["event"] for r in left], dtype=torch.bool) & (times <= horizon)
            times = times.clamp_max(horizon)
            risks = [torch.tensor([1-r["survival"][j] for r in rows], dtype=torch.float64) for rows in (left, right)]
            rng, samples = random.Random(seed), []
            for _ in range(bootstrap_samples):
                indices = torch.tensor([rng.randrange(len(left)) for _ in left])
                a, b = [concordance_index(risk[indices], times[indices], events[indices]) for risk in risks]
                if math.isfinite(a) and math.isfinite(b):
                    samples.append((a, b, b-a))
            if samples:
                intervals = torch.quantile(torch.tensor(samples, dtype=torch.float64),
                    torch.tensor([.025, .975], dtype=torch.float64), dim=0)
                result.update(reference_confidence_interval=intervals[:, 0].tolist(),
                              candidate_confidence_interval=intervals[:, 1].tolist(),
                              confidence_interval_low=float(intervals[0, 2]),
                              confidence_interval_high=float(intervals[1, 2]),
                              bootstrap_samples_used=len(samples), status="estimated")
            else:
                result["status"] = "no_estimable_bootstrap_samples"
            results.append(result)
    return results
