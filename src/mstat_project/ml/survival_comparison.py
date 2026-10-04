"""Paired evaluation of continuous-survival model predictions.

Compare only identical patients, outcomes and available MRI histories at a fixed
landmark. Bootstrap whole patients within that landmark; never pool repeated
observations from several landmarks as independent patients.
"""

import math

import torch

from mstat_project.ml.metrics import paired_c_index_difference
from mstat_project.ml.utils import concordance_index


def compare_predictions(predictions_by_variant, horizons, *, bootstrap_samples=1000, seed=42):
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
    pairs = [("a", "b"), ("a", "history"), ("b", "history")]
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
            if left[key]["split"] != "test":
                raise ValueError("Paired reporting expects held-out test predictions")
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
