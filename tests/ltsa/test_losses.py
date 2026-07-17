import pytest
import torch

from ltsa.losses import CoxPHSurvLoss, NLLSurvLoss, ce_surv_loss, cox_ph_loss, nll_loss


def test_nll_loss_accepts_supplied_survival_and_matches_manual_value() -> None:
    hazards = torch.tensor([[0.2, 0.3], [0.1, 0.4]])
    survival = torch.cumprod(1 - hazards, dim=1)
    labels = torch.tensor([0, 1])
    censorship = torch.tensor([0, 1])
    expected = (-torch.log(hazards[0, 0]) - 0.85 * torch.log(survival[1, 1])) / 2

    torch.testing.assert_close(
        nll_loss(hazards, survival, labels, censorship, beta=0.15),
        expected,
    )


def test_discrete_losses_are_finite_at_probability_boundaries() -> None:
    hazards = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1])
    censorship = torch.tensor([0, 1])

    assert torch.isfinite(nll_loss(hazards, None, labels, censorship))
    assert torch.isfinite(ce_surv_loss(hazards, None, labels, censorship))


def test_nll_wrapper_respects_explicit_zero_beta() -> None:
    hazards = torch.tensor([[0.2, 0.3]])
    labels = torch.tensor([0])
    censorship = torch.tensor([1])

    wrapped = NLLSurvLoss(beta=0.8)(hazards, None, labels, censorship, beta=0.0)
    direct = nll_loss(hazards, None, labels, censorship, beta=0.0)

    torch.testing.assert_close(wrapped, direct)


def test_cox_ph_loss_and_wrapper_match_breslow_calculation() -> None:
    risk = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([True, True, False])
    expected = -(
        risk[0] - torch.logsumexp(risk, dim=0) + risk[1] - torch.logsumexp(risk[1:], dim=0)
    ) / 2

    torch.testing.assert_close(cox_ph_loss(risk, time, event), expected)
    torch.testing.assert_close(CoxPHSurvLoss()(risk, time, event), expected)


def test_discrete_loss_rejects_out_of_range_label() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        nll_loss(torch.tensor([[0.2, 0.3]]), None, torch.tensor([2]), torch.tensor([0]))
