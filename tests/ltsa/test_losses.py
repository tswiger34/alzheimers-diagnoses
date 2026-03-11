import pytest  # noqa
import torch
import numpy as np


def test_smoke_test():
    pass


class TestRefactorBehavior:
    def test_cox_surv_loss():

        class OriginalCoxSurvLoss(object):
            def __call__(
                hazards: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, c: torch.Tensor, beta: float, **kwargs
            ):
                # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
                # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
                current_batch_len = len(S)
                R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
                for i in range(current_batch_len):
                    for j in range(current_batch_len):
                        R_mat[i, j] = S[j] >= S[i]

                R_mat = torch.FloatTensor(R_mat).to(device=None)
                theta = hazards.reshape(-1)
                exp_theta = torch.exp(theta)
                loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))

                return loss_cox

    def test_nll_surv_loss():
        def original_nll_loss(
            hazards: torch.Tensor,
            S: torch.Tensor,
            Y: torch.Tensor,
            c: torch.Tensor,
            obs_times: torch.Tensor,
            beta: float = 0.15,
            eps: float = 1e-7,
        ):
            batch_size = len(Y)
            Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
            c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
            if S is None:
                S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards

            S_padded = torch.cat(
                [torch.ones_like(c), S], 1
            )  # S(-1) = 0, all patients are alive from (-inf, 0) by definition

            uncensored_loss = -(1 - c) * (
                torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
                + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
            )
            censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
            neg_l = censored_loss + uncensored_loss
            loss = (1 - beta) * neg_l + beta * uncensored_loss
            loss = loss.mean()
            return loss

    def test_ce_surv_loss():
        def original_ce_loss(
            hazards: torch.Tensor,
            S: torch.Tensor,
            Y: torch.Tensor,
            c: torch.Tensor,
            obs_times=None,
            beta: float = 0.15,
            eps: float = 1e-7,
        ):
            batch_size = len(Y)
            Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
            c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
            if S is None:
                S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards

            S_padded = torch.cat([torch.ones_like(c), S], 1)

            reg = -(1 - c) * (
                torch.log(torch.gather(S_padded, 1, Y) + eps)
                + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
            )
            ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
                1 - torch.gather(S, 1, Y).clamp(min=eps)
            )
            loss = (1 - beta) * ce_l + beta * reg
            loss = loss.mean()

            return loss
