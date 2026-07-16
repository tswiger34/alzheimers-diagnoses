"""Survival loss functions used by the LTSA training pipeline.

This module provides discrete-time and Cox-style losses for right-censored
survival analysis:

- `nll_loss`: negative log-likelihood for discrete hazard predictions.
- `ce_surv_loss`: cross-entropy-style survival loss with an NLL-style
  regularization term.
- `cox_surv_loss`: Cox partial log-likelihood objective.

It also exposes callable wrapper classes (`NLLSurvLoss`, `CrossEntropySurvLoss`, and `CoxSurvLoss`) to support
object-style loss configuration in training code.

Implementation credit:
- Mahmood Lab MCAT utilities:
  https://github.com/mahmoodlab/MCAT/blob/b9cca63be83c67de7f95308d54a58f80b78b0da1/utils/utils.py
- BioNLP Lab longitudinal transformer survival repository:
  https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis/blob/main/src/losses.py

Tensor conventions:
- `hazards`: per-bin hazard probabilities, shape `(batch_size, num_time_bins)`.
- `S`: per-bin survival probabilities, shape `(batch_size, num_time_bins)`; computed internally when optional in
  discrete-time losses.
- `Y`: discrete event/censor bin indices, shape `(batch_size,)` or `(batch_size, 1)`.
- `c`: censoring indicator (`0` observed event, `1` censored), shape `(batch_size,)` or `(batch_size, 1)`.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor, device


def nll_loss(
    hazards: Tensor, S: Tensor | None, Y: Tensor, c: Tensor, beta: float = 0.15, eps: float = 1e-7, **kwargs
):
    """Negative log-likelihood survival loss for discrete-time survival models.

    This function computes the negative log-likelihood (NLL) loss for discrete-time survival analysis. The model
    predicts hazard probabilities for each time interval, and the loss evaluates how well those predicted hazards
    match the observed event times and censoring indicators.

    The loss combines contributions from:
    - **Uncensored observations**: log-likelihood of surviving up to the
      event time and failing at that interval.
    - **Censored observations**: log-likelihood of surviving beyond the
      censoring interval.

    Args:
        hazards (Tensor):
            Predicted hazard probabilities for each observation and time bin. The shape should be
            `(batch_size, num_time_bins)` and the values should be in the range `[0, 1]`

        S (Tensor):
            Survival probabilities corresponding to `hazards`. Shape `(batch_size, num_time_bins)`. Should equal
            the cumulative product of `(1 - hazards)` along the time dimension. If `None`, then computed internally

        Y (Tensor):
            Ground-truth event time indices for each observation. Each value indicates the discrete time
            bin in which the event or censoring occurred. Shape should be `(batch_size,)` or `(batch_size, 1)`

        c (Tensor):
            Censoring indicator for each observation where:
                - ``0`` = event observed
                - ``1`` = right-censored

            Shape ``(batch_size,)`` or ``(batch_size, 1)``.
        beta (float, optional):
            Weight applied to the uncensored loss component. This can help stabilize training
            by emphasizing event observations relative to censored observations. Defaults to `0.15`

        eps (float, optional):
            Small constant used to clamp probabilities before taking the logarithm, preventing numerical
            instability from `log(0)`. Defaults to `1e-7`.

        **kwargs:
            Additional keyword arguments included for API compatibility. These values are ignored.

    Returns:
        Tensor: Scalar tensor containing the mean negative log-likelihood loss across the batch.
    """
    batch_size: int = len(Y)
    Y: Tensor = Y.view(batch_size, 1)
    c: Tensor = c.view(batch_size, 1).float()
    S: Tensor = S or torch.cumprod(input=1 - hazards, dim=1)
    S_padded: Tensor = torch.cat(tensors=[torch.ones_like(input=c), S], dim=1)
    uncensored: Tensor = 1 - c

    uncensored_loss: Tensor = uncensored.neg() * (
        torch.log(input=torch.gather(input=S_padded, dim=1, index=Y).clamp(min=eps))
        + torch.log(input=torch.gather(input=hazards, dim=1, index=Y).clamp(min=eps))
    )
    censored_loss: Tensor = c.neg() * torch.log(
        input=torch.gather(input=S_padded, dim=1, index=Y + 1).clamp(min=eps)
    )
    neg_l: Tensor = censored_loss + uncensored_loss
    loss: Tensor = (1 - beta) * neg_l + beta * uncensored_loss
    loss: Tensor = loss.mean()
    return loss


def ce_surv_loss(
    hazards: Tensor, S: Tensor | None, Y: Tensor, c: Tensor, beta: float, eps: float = 1e-7, **kwargs
) -> Tensor:
    """Cross-entropy survival loss for discrete-time survival models.

    This loss function combines a cross-entropy formulation of survival likelihood with a regularization component
    derived from the negative log-likelihood of the discrete hazard formulation.

    The loss consists of two components:

    1. **Cross-entropy survival term (`ce_l`)**
       - For censored observations: encourages the model to assign high
         survival probability at the censoring time.
       - For uncensored observations: encourages the model to assign high
         probability of failure at the observed event interval.

    2. **Regularization term (`reg`)**
       - Equivalent to the discrete hazard negative log-likelihood used in
         standard survival modeling.

    The parameter ``beta`` controls the contribution of the regularization component relative to the cross-entropy
    survival term.

    Args:
        hazards (Tensor):
            Predicted hazard probabilities for each observation and time bin. The shape should be
            `(batch_size, num_time_bins)` and the values should be in the range `[0, 1]`

        S (Tensor):
            Survival probabilities corresponding to `hazards`. Shape `(batch_size, num_time_bins)`. Should equal
            the cumulative product of `(1 - hazards)` along the time dimension. If `None`, then computed internally

        Y (Tensor):
            Ground-truth event time indices for each observation. Each value indicates the discrete time
            bin in which the event or censoring occurred. Shape should be `(batch_size,)` or `(batch_size, 1)`

        c (Tensor):
            Censoring indicator for each observation where:
                - ``0`` = event observed
                - ``1`` = right-censored

            Shape ``(batch_size,)`` or ``(batch_size, 1)``.
        beta (float, optional):
            Weight applied to the uncensored loss component. This can help stabilize training
            by emphasizing event observations relative to censored observations. Defaults to `0.15`

        eps (float, optional):
            Small constant used to clamp probabilities before taking the logarithm, preventing numerical
            instability from `log(0)`. Defaults to `1e-7`.

        **kwargs:
            Additional keyword arguments included for API compatibility. These values are ignored.

    Returns:
        Tensor: Scalar tensor containing the mean cross-entropy survival loss across the batch.
    """
    batch_size: int = len(Y)
    Y: Tensor = Y.view(batch_size, 1)
    c: Tensor = c.view(batch_size, 1).float()
    S: Tensor = S or torch.cumprod(input=1 - hazards, dim=1)
    S_padded: Tensor = torch.cat(tensors=[torch.ones_like(input=c), S], dim=1)
    c_flipped: Tensor = 1 - c

    reg: Tensor = c_flipped.neg() * (
        torch.log(input=torch.gather(input=S_padded, dim=1, index=Y) + eps)
        + torch.log(input=torch.gather(input=hazards, dim=1, index=Y).clamp(min=eps))
    )
    ce_l: Tensor = c.neg() * torch.log(input=torch.gather(input=S, dim=1, index=Y).clamp(min=eps)) - (
        c_flipped
    ) * torch.log(input=1 - torch.gather(input=S, dim=1, index=Y).clamp(min=eps))
    loss: Tensor = (1 - beta) * ce_l + beta * reg
    loss: Tensor = loss.mean()

    return loss


def cox_surv_loss(hazards: Tensor, S: Tensor, c: Tensor, device: device | None, **kwargs) -> Tensor:
    """Cox proportional hazards loss function for neural-network-based survival models.

    This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data

    Args:
        hazards (Tensor): Tensor of hazard values for each obs 1,2,...,k
        S (Tensor): Tensor of survival scores for obs 1,2,...,k, should be the cumulative product of `1 - hazards`
        c (Tensor): Tensor of censorship statuses for obs 1,2,...,k, values should be either 0 or 1
        device (device | None): Optionally provide the device being used for computing

    Returns:
        Tensor: Scalar tensor representing the mean negative Cox partial log-likelihood across the batch.

    """
    current_batch_len: int = len(S)
    R_mat: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.zeros(
        shape=[current_batch_len, current_batch_len], dtype=int
    )
    ## TODO: Check this nested loop, there is probably a better way to do this
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = S[j] >= S[i]

    R_mat: Tensor = torch.FloatTensor(R_mat).to(device=device)
    theta: Tensor = hazards.reshape(-1)
    exp_theta: Tensor = torch.exp(input=theta)
    loss_cox: Tensor = torch.mean(
        input=(theta - torch.log(input=torch.sum(input=exp_theta * R_mat, dim=1))) * (1 - c)
    ).neg()

    return loss_cox


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood using Breslow handling for ties.

    Higher ``risk`` means an earlier event.  Each event's risk set contains
    patients whose observed time is greater than or equal to its event time.
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

    event_times = torch.unique(time[event])
    if event_times.numel() == 0:
        return risk.sum() * 0.0

    log_likelihood_terms: list[torch.Tensor] = []
    for event_time in event_times:
        tied_events = event & (time == event_time)
        event_count = tied_events.sum()
        risk_set = time >= event_time
        term = risk[tied_events].sum() - event_count * torch.logsumexp(risk[risk_set], dim=0)
        log_likelihood_terms.append(term)

    return -torch.stack(log_likelihood_terms).sum() / event.sum()


class CrossEntropySurvLoss(object):
    """Cross entropy survival loss object"""

    def __init__(self, beta: float = 0.15):
        self.beta: float = beta

    def __call__(self, hazards: Tensor, S: Tensor, Y: Tensor, c: Tensor, beta: float | None = None, **kwargs):
        beta: float = beta or self.beta
        return ce_surv_loss(hazards, S, Y, c, beta=beta)


class NLLSurvLoss(object):
    """Negative log-likelihood survival loss object"""

    def __init__(self, beta: float = 0.15):
        self.beta: float = beta

    def __call__(self, hazards, S, Y, c, beta=None, **kwargs):
        beta: float = beta or self.beta
        return nll_loss(hazards, S, Y, c, beta=beta)


class CoxSurvLoss(object):
    """Cox survival loss object, `__call__` calls the `cox_surv_loss` function

    Args:
        hazards (Tensor): Tensor of hazard values for each obs 1,2,...,k
        S (Tensor): Tensor of survival scores for obs 1,2,...,k, should be the cumulative product of `1 - hazards`
        c (Tensor): Tensor of censorship statuses for obs 1,2,...,k, values should be either 0 or 1
        device (device | None): Optionally provide the device being used for computing

    Notes
    -----
    - The risk set matrix is constructed such that R[i, j] = 1 if subject j is still at risk at time S[i]
      (i.e., S[j] >= S[i]), otherwise 0
    - The implementation assumes right-censored survival data
    - This loss function is differentiable and suitable for optimization via standard gradient-based training in
      PyTorch
    """

    def __call__(hazards: Tensor, S: Tensor, c, device: device | None, **kwargs):
        return cox_surv_loss(hazards=hazards, S=S, c=c, device=device, **kwargs)


class CoxPHSurvLoss(object):
    """Cox proportional hazards survival loss object"""

    def __call__(self, risk: Tensor, time: Tensor, event: Tensor, **kwargs):
        return cox_ph_loss(risk=risk, time=time, event=event, **kwargs)
