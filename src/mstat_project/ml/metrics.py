"""Metrics and paired uncertainty estimates for survival experiments."""

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from mstat_project.ml.utils import concordance_index


@dataclass(frozen=True)
class PairedCIndexComparison:
    baseline_c_index: float
    ltsa_c_index: float
    difference: float
    confidence_interval_low: float
    confidence_interval_high: float
    bootstrap_samples: int


def paired_c_index_difference(
    baseline_risk: Tensor,
    ltsa_risk: Tensor,
    times: Tensor,
    events: Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> PairedCIndexComparison:
    """Compute a paired patient-bootstrap interval for an LTSA-minus-baseline C-index."""

    n_patients = baseline_risk.numel()
    if not (n_patients == ltsa_risk.numel() == times.numel() == events.numel()):
        raise ValueError("Paired predictions and outcomes must have equal lengths")
    if n_patients < 2:
        raise ValueError("At least two patients are required for paired comparison")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    baseline_c_index = concordance_index(baseline_risk, times, events)
    ltsa_c_index = concordance_index(ltsa_risk, times, events)
    if math.isnan(baseline_c_index) or math.isnan(ltsa_c_index):
        raise ValueError("The paired cohort contains no comparable survival pairs")

    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(bootstrap_samples):
        indices = torch.tensor([rng.randrange(n_patients) for _ in range(n_patients)], dtype=torch.long)
        baseline_sample = concordance_index(
            baseline_risk[indices],
            times[indices],
            events[indices],
        )
        ltsa_sample = concordance_index(ltsa_risk[indices], times[indices], events[indices])
        if not math.isnan(baseline_sample) and not math.isnan(ltsa_sample):
            differences.append(ltsa_sample - baseline_sample)
    if not differences:
        raise ValueError("No bootstrap sample contained comparable survival pairs")
    quantiles = torch.quantile(
        torch.tensor(differences, dtype=torch.float64),
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    return PairedCIndexComparison(
        baseline_c_index=baseline_c_index,
        ltsa_c_index=ltsa_c_index,
        difference=ltsa_c_index - baseline_c_index,
        confidence_interval_low=float(quantiles[0].item()),
        confidence_interval_high=float(quantiles[1].item()),
        bootstrap_samples=len(differences),
    )
