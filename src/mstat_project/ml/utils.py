import math
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


def default_tensor_dir() -> Path:
    return Path(os.getenv("IMAGES_PATH", "data/images")) / "tensors"


def default_checkpoint_dir(model_type: str = "baseline") -> Path:
    return Path(os.getenv("DATA_DIR", "data")) / "artifacts" / "model_checkpoints" / model_type


def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """Compute Harrell's C-index for right-censored outcomes.

    A pair is comparable when the patient with the shorter observed time had
    an event.  Risk ties receive half credit.  ``nan`` is returned when no
    comparable pairs exist.
    """

    risk = risk.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    time = time.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    event = event.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
    if not (risk.numel() == time.numel() == event.numel()):
        raise ValueError("risk, time, and event must have the same number of elements")

    comparable = event[:, None] & (time[:, None] < time[None, :])
    comparable_count = int(comparable.sum().item())
    if comparable_count == 0:
        return math.nan

    risk_difference = risk[:, None] - risk[None, :]
    concordant = ((risk_difference > 0) & comparable).sum(dtype=torch.float64)
    tied = ((risk_difference == 0) & comparable).sum(dtype=torch.float64)
    return float(((concordant + 0.5 * tied) / comparable_count).item())
