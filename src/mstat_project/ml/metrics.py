"""Compute paired survival-model performance and bootstrap uncertainty.

The module compares matched baseline and LTSA predictions with Harrell's
C-index. Uncertainty is estimated by resampling patient indices jointly so
that each bootstrap replicate preserves the pairing between both models and
the corresponding survival outcome.
"""

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from mstat_project.ml.utils import concordance_index


@dataclass(frozen=True)
class PairedCIndexComparison:
    """C-index comparison and percentile bootstrap confidence interval.

    Attributes:
        baseline_c_index: Harrell C-index for baseline risk predictions.
        ltsa_c_index: Harrell C-index for LTSA risk predictions.
        difference: LTSA C-index minus baseline C-index.
        confidence_interval_low: Lower bound of the two-sided 95% percentile
            bootstrap confidence interval.
        confidence_interval_high: Upper bound of the two-sided 95% percentile
            bootstrap confidence interval.
        bootstrap_samples: Number of valid bootstrap replicates used to
            calculate the confidence interval. Replicates without comparable
            survival pairs are excluded, so this may be less than the
            requested count.
    """

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
    """Compare matched C-indices with a paired patient-bootstrap interval.

    The function first calculates both C-indices on the complete cohort. It
    then samples patient indices with replacement and applies each sampled
    index set to both models and the outcomes. Replicates with no comparable
    survival pairs are discarded. The 2.5th and 97.5th percentiles of the
    remaining LTSA-minus-baseline differences form the confidence interval.

    Args:
        baseline_risk: One-dimensional baseline risk scores, where larger
            values indicate an earlier expected event.
        ltsa_risk: One-dimensional LTSA risk scores aligned patient-for-patient
            with ``baseline_risk``.
        times: Observed event or censoring times aligned with the risk scores.
        events: Event indicators aligned with the risk scores, where true
            means the event was observed.
        bootstrap_samples: Number of paired bootstrap replicates to request.
        seed: Seed for the local pseudo-random number generator used to sample
            patient indices.

    Returns:
        A ``PairedCIndexComparison`` containing complete-cohort C-indices, the
        LTSA-minus-baseline difference, the percentile confidence interval,
        and the number of valid bootstrap replicates.

    Raises:
        ValueError: If input tensors contain different numbers of patients,
            fewer than two patients are supplied, ``bootstrap_samples`` is not
            positive, the complete cohort has no comparable survival pairs,
            or no bootstrap replicate contains a comparable pair.
    """

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
