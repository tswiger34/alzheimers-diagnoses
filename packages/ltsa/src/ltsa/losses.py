"""Survival loss functions for discrete-time and Cox models."""

import torch
from torch import Tensor


def validate_discrete_inputs(
    hazards: Tensor,
    survival: Tensor | None,
    labels: Tensor,
    censorship: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validate and normalize inputs shared by the discrete survival losses.

    Args:
        hazards (Tensor): Per-bin hazard probabilities with shape ``[observations, time_bins]``.
        survival (Tensor | None): Per-bin survival probabilities with the same shape as ``hazards``. If
            ``None``, survival is calculated as the cumulative product of ``1 - hazards``.
        labels (Tensor): Event or censoring time-bin indices with one value per observation.
        censorship (Tensor): Censoring indicators with one value per observation, where ``0`` means
            an observed event and ``1`` means right-censored.

    Returns:
        A tuple containing hazards, survival probabilities, column-shaped integer labels,
        and column-shaped censoring indicators. Returned tensors are placed on the hazards
        device and use compatible dtypes.

    Raises:
        ValueError: If tensor shapes, observation counts, label indices, censoring values,
            or hazard probabilities are invalid.
    """

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
    """Compute negative log-likelihood for right-censored discrete hazards.

    For an observed event, the likelihood combines survival through all preceding bins
    with the hazard in the labeled bin. For a censored observation, it uses survival
    through the labeled bin.

    Args:
        hazards (Tensor): Per-bin hazard probabilities with shape ``[observations, time_bins]``.
        survival (Tensor | None): Per-bin survival probabilities with the same shape as ``hazards``. If
            ``None``, survival is calculated from ``hazards``.
        labels (Tensor): Event or censoring time-bin indices with one value per observation.
        censorship (Tensor): Censoring indicators, where ``0`` means an observed event and ``1``
            means right-censored.
        beta (float): Additional weighting for the uncensored component. Must be between ``0`` and
            ``1``. A value of ``0`` applies equal weight to event and censoring terms. Defaults to ``0.15``.
        eps (float): Minimum probability used before taking logarithms. Defaults to ``1e-7``.

    Returns:
        A scalar tensor containing the mean negative log-likelihood.

    Raises:
        ValueError: If ``beta`` is outside ``[0, 1]`` or any discrete survival input is
            invalid.
    """

    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    hazards, survival, labels, censorship = validate_discrete_inputs(
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
    """Compute cross-entropy survival loss with event-likelihood regularization.

    The cross-entropy term distinguishes survival from failure at the labeled time bin.
    The regularization term adds the discrete event likelihood for uncensored
    observations.

    Args:
        hazards (Tensor): Per-bin hazard probabilities with shape ``[observations, time_bins]``.
        survival (Tensor | None): Per-bin survival probabilities with the same shape as ``hazards``. If
            ``None``, survival is calculated from ``hazards``.
        labels (Tensor): Event or censoring time-bin indices with one value per observation.
        censorship (Tensor): Censoring indicators, where ``0`` means an observed event and ``1``
            means right-censored.
        beta (float): Weight assigned to the event-likelihood regularization term. Must be between
            ``0`` and ``1``.
        eps (float): Numerical stability bound applied before taking logarithms.

    Returns:
        A scalar tensor containing the mean weighted cross-entropy survival loss.

    Raises:
        ValueError: If ``beta`` is outside ``[0, 1]`` or any discrete survival input is
            invalid.
    """

    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    hazards, survival, labels, censorship = validate_discrete_inputs(
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
    """Compute the negative Cox partial log-likelihood.

    Higher risk values indicate an earlier expected event. Each event's risk set contains
    observations whose recorded time is greater than or equal to that event time. Tied
    event times are handled with the Breslow approximation.

    Args:
        risk (Tensor): Predicted log-risk scores with one value per observation.
        time (Tensor): Observed event or censoring times with one value per observation.
        event (Tensor): Event indicators with one value per observation. Values are interpreted as
            booleans, where true means the event was observed.

    Returns:
        A scalar tensor containing the negative partial log-likelihood averaged over
        observed events. If no events are observed, returns a differentiable zero.

    Raises:
        ValueError: If inputs have different element counts, contain no observations, or
            include non-finite risk scores or times.
    """

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
    """Callable cross-entropy survival loss with a configurable default weight."""

    def __init__(self, beta: float = 0.15) -> None:
        """Initialize the loss wrapper.

        Args:
            beta: Default weight assigned to the event-likelihood regularization term.
        """

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
        """Compute cross-entropy survival loss.

        Args:
            hazards: Per-bin hazard probabilities with shape
                ``[observations, time_bins]``.
            survival: Per-bin survival probabilities, or ``None`` to calculate them from
                ``hazards``.
            labels: Event or censoring time-bin indices.
            censorship: Censoring indicators, where ``0`` means an observed event and
                ``1`` means right-censored.
            beta: Per-call regularization weight. If ``None``, uses the value supplied
                when this wrapper was initialized.

        Returns:
            A scalar tensor containing the mean cross-entropy survival loss.

        Raises:
            ValueError: If the selected ``beta`` or any discrete survival input is invalid.
        """

        return ce_surv_loss(
            hazards,
            survival,
            labels,
            censorship,
            beta=self.beta if beta is None else beta,
        )


class NLLSurvLoss:
    """Callable discrete negative log-likelihood with a configurable default weight."""

    def __init__(self, beta: float = 0.15) -> None:
        """Initialize the loss wrapper.

        Args:
            beta: Default additional weight applied to the uncensored loss component.
        """

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
        """Compute discrete negative log-likelihood.

        Args:
            hazards: Per-bin hazard probabilities with shape
                ``[observations, time_bins]``.
            survival: Per-bin survival probabilities, or ``None`` to calculate them from
                ``hazards``.
            labels: Event or censoring time-bin indices.
            censorship: Censoring indicators, where ``0`` means an observed event and
                ``1`` means right-censored.
            beta: Per-call uncensored weight. If ``None``, uses the value supplied when
                this wrapper was initialized.

        Returns:
            A scalar tensor containing the mean negative log-likelihood.

        Raises:
            ValueError: If the selected ``beta`` or any discrete survival input is invalid.
        """

        return nll_loss(
            hazards,
            survival,
            labels,
            censorship,
            beta=self.beta if beta is None else beta,
        )


class CoxPHSurvLoss:
    """Callable wrapper around the Cox proportional-hazards loss."""

    def __call__(self, risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
        """Compute negative Cox partial log-likelihood.

        Args:
            risk: Predicted log-risk scores with one value per observation.
            time: Observed event or censoring times.
            event: Event indicators interpreted as booleans.

        Returns:
            A scalar tensor containing the event-averaged negative partial
            log-likelihood.

        Raises:
            ValueError: If the Cox loss inputs are empty, inconsistent, or non-finite.
        """

        return cox_ph_loss(risk, time, event)
