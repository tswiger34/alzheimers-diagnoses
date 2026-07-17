"""Survival loss functions for discrete-time and Cox models."""

import torch
from torch import Tensor


def _validate_discrete_inputs(
    hazards: Tensor,
    survival: Tensor | None,
    labels: Tensor,
    censorship: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if hazards.ndim != 2 or hazards.shape[1] < 2:
        raise ValueError("hazards must have shape [observations, at least two time bins]")
    batch_size = hazards.shape[0]
    labels = labels.reshape(-1).to(device=hazards.device, dtype=torch.long)
    censorship = censorship.reshape(-1).to(device=hazards.device, dtype=hazards.dtype)
    if labels.numel() != batch_size or censorship.numel() != batch_size:
        raise ValueError("hazards, labels, and censorship must contain the same number of observations")
    if (labels < 0).any() or (labels >= hazards.shape[1]).any():
        raise ValueError("labels contain an out-of-range time-bin index")
    if ((censorship != 0) & (censorship != 1)).any():
        raise ValueError("censorship values must be 0 (event) or 1 (censored)")
    if not torch.isfinite(hazards).all() or (hazards < 0).any() or (hazards > 1).any():
        raise ValueError("hazards must contain finite probabilities between 0 and 1")
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)
    if survival.shape != hazards.shape:
        raise ValueError("survival must have the same shape as hazards")
    return hazards, survival, labels.unsqueeze(1), censorship.unsqueeze(1)


def nll_loss(
    hazards: Tensor,
    survival: Tensor | None,
    labels: Tensor,
    censorship: Tensor,
    *,
    beta: float = 0.15,
    eps: float = 1e-7,
) -> Tensor:
    """Negative log-likelihood for right-censored discrete hazards."""

    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    hazards, survival, labels, censorship = _validate_discrete_inputs(
        hazards,
        survival,
        labels,
        censorship,
    )
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored = 1 - censorship
    uncensored_loss = -uncensored * (
        torch.log(torch.gather(survival_padded, 1, labels).clamp_min(eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp_min(eps))
    )
    censored_loss = -censorship * torch.log(torch.gather(survival_padded, 1, labels + 1).clamp_min(eps))
    return ((1 - beta) * (uncensored_loss + censored_loss) + beta * uncensored_loss).mean()


def ce_surv_loss(
    hazards: Tensor,
    survival: Tensor | None,
    labels: Tensor,
    censorship: Tensor,
    *,
    beta: float = 0.15,
    eps: float = 1e-7,
) -> Tensor:
    """Cross-entropy-style loss with discrete survival regularization."""

    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    hazards, survival, labels, censorship = _validate_discrete_inputs(
        hazards,
        survival,
        labels,
        censorship,
    )
    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored = 1 - censorship
    regularization = -uncensored * (
        torch.log(torch.gather(survival_padded, 1, labels).clamp_min(eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp_min(eps))
    )
    survival_at_label = torch.gather(survival, 1, labels).clamp(min=eps, max=1 - eps)
    cross_entropy = -censorship * torch.log(survival_at_label) - uncensored * torch.log(1 - survival_at_label)
    return ((1 - beta) * cross_entropy + beta * regularization).mean()


def cox_ph_loss(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Negative Cox partial log-likelihood with Breslow tie handling."""

    risk = risk.reshape(-1)
    time = time.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    event = event.reshape(-1).to(device=risk.device, dtype=torch.bool)
    if not (risk.numel() == time.numel() == event.numel()):
        raise ValueError("risk, time, and event must have the same number of elements")
    if risk.numel() == 0:
        raise ValueError("Cox loss requires at least one observation")
    if not torch.isfinite(risk).all() or not torch.isfinite(time).all():
        raise ValueError("risk and time must contain only finite values")
    if not event.any():
        return risk.sum() * 0.0

    log_likelihood_terms: list[Tensor] = []
    for event_time in torch.unique(time[event]):
        tied_events = event & (time == event_time)
        log_likelihood_terms.append(
            risk[tied_events].sum() - tied_events.sum() * torch.logsumexp(risk[time >= event_time], dim=0)
        )
    return -torch.stack(log_likelihood_terms).sum() / event.sum()


class CrossEntropySurvLoss:
    def __init__(self, beta: float = 0.15) -> None:
        self.beta = beta

    def __call__(
        self,
        hazards: Tensor,
        survival: Tensor | None,
        labels: Tensor,
        censorship: Tensor,
        *,
        beta: float | None = None,
    ) -> Tensor:
        return ce_surv_loss(
            hazards,
            survival,
            labels,
            censorship,
            beta=self.beta if beta is None else beta,
        )


class NLLSurvLoss:
    def __init__(self, beta: float = 0.15) -> None:
        self.beta = beta

    def __call__(
        self,
        hazards: Tensor,
        survival: Tensor | None,
        labels: Tensor,
        censorship: Tensor,
        *,
        beta: float | None = None,
    ) -> Tensor:
        return nll_loss(
            hazards,
            survival,
            labels,
            censorship,
            beta=self.beta if beta is None else beta,
        )


class CoxPHSurvLoss:
    def __call__(self, risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
        return cox_ph_loss(risk, time, event)
