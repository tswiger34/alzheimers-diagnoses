import math
from pathlib import Path

import polars as pl
import pytest
import torch

from mstat_project.ml.baseline import (
    PatientRecord,
    SingleImageSurvivalDataset,
    SingleImageSurvivalModel,
    build_patient_records,
    concordance_index,
    cox_ph_loss,
    restore_checkpoint,
    save_checkpoint,
    TrainingConfig,
)


def test_cox_ph_loss_matches_manual_breslow_calculation() -> None:
    risk = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([True, True, False])

    expected = -(risk[0] - torch.logsumexp(risk, dim=0) + risk[1] - torch.logsumexp(risk[1:], dim=0)) / 2
    actual = cox_ph_loss(risk, time, event)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()


def test_cox_ph_loss_uses_full_risk_set_for_tied_events() -> None:
    risk = torch.tensor([0.0, 1.0, -1.0])
    time = torch.tensor([2.0, 2.0, 3.0])
    event = torch.tensor([True, True, False])

    expected = -(risk[:2].sum() - 2 * torch.logsumexp(risk, dim=0)) / 2

    torch.testing.assert_close(cox_ph_loss(risk, time, event), expected)


def test_concordance_index_handles_concordance_discordance_and_ties() -> None:
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([True, True, False])

    assert concordance_index(torch.tensor([3.0, 2.0, 1.0]), time, event) == pytest.approx(1.0)
    assert concordance_index(torch.tensor([1.0, 2.0, 3.0]), time, event) == pytest.approx(0.0)
    assert concordance_index(torch.ones(3), time, event) == pytest.approx(0.5)
    assert math.isnan(concordance_index(torch.ones(3), time, torch.zeros(3, dtype=torch.bool)))


def test_single_image_dataset_loads_the_selected_image(tmp_path: Path) -> None:
    tensor_path = tmp_path / "P001.pt"
    images = torch.stack(
        [
            torch.zeros(1, 20, 20, 20),
            torch.arange(20**3, dtype=torch.float32).reshape(1, 20, 20, 20),
        ]
    )
    torch.save({"ptid": "P001", "img_ids": ["10", "20"], "images": images}, tensor_path)
    record = PatientRecord("P001", "20", "train", 12.0, True, tensor_path)

    sample = SingleImageSurvivalDataset([record], spatial_size=(16, 16, 16))[0]

    assert sample["ptid"] == "P001"
    assert sample["image_id"] == "20"
    assert sample["image"].shape == (1, 16, 16, 16)
    assert sample["image"].mean().item() == pytest.approx(0.0, abs=1e-5)
    assert sample["image"].std().item() == pytest.approx(1.0, rel=1e-5)
    assert sample["time"].item() == pytest.approx(12.0)
    assert sample["event"].item() is True


def test_build_patient_records_enforces_one_patient_per_split(tmp_path: Path) -> None:
    rows = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        ptid = f"P{index:03d}"
        torch.save(
            {"ptid": ptid, "img_ids": [str(index)], "images": torch.ones(1, 1, 16, 16, 16)},
            tmp_path / f"{ptid}.pt",
        )
        rows.append(
            {
                "image_id": index,
                "ptid": ptid,
                "image_date": None,
                "baseline_diagnosis": "MCI",
                "final_diagnosis": "AD",
                "is_censored": False,
                "observed_time_months": float(index),
                "train_test_split": split,
            }
        )

    records = build_patient_records(pl.DataFrame(rows), tmp_path)

    assert len(records) == 3
    assert {record.ptid for record in records} == {"P001", "P002", "P003"}
    assert all(record.event_observed for record in records)


def test_resnet_model_accepts_a_3d_mri() -> None:
    model = SingleImageSurvivalModel(weights=None).eval()
    image = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        resnet_input = model._volume_to_resnet_input(image)
        risk = model(image)

    assert resnet_input.shape == (1, 3, 224, 224)
    assert risk.shape == (1,)
    assert torch.isfinite(risk).all()
    assert model.encoder.conv1.in_channels == 3

    model.train()
    assert not any(
        module.training for module in model.encoder.modules() if isinstance(module, torch.nn.BatchNorm2d)
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(2, 2)

    risk = model(features)
    loss = risk.square().mean()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "run" / "epoch_001.pt"
    config = TrainingConfig(epochs=1, spatial_size=(32, 32, 32))
    save_checkpoint(
        checkpoint_path,
        "test-run",
        1,
        model,
        optimizer,
        config,
        {"train_loss": float(loss.item())},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint_path.is_file()
    assert checkpoint["run_id"] == "test-run"
    assert checkpoint["epoch"] == 1
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())


def test_restore_checkpoint_restores_training_state_and_applies_current_optimizer_config(
    tmp_path: Path,
) -> None:
    source_model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = source_model(torch.randn(2, 2)).square().mean()
    loss.backward()
    source_optimizer.step()

    checkpoint_path = tmp_path / "epoch_010.pt"
    save_checkpoint(
        checkpoint_path,
        "source-run",
        10,
        source_model,
        source_optimizer,
        TrainingConfig(epochs=10),
        {"validation_loss": 0.25},
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=2e-3, weight_decay=2e-4)
    epoch, metrics = restore_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        learning_rate=2e-3,
        weight_decay=2e-4,
    )

    assert epoch == 10
    assert metrics["validation_loss"] == pytest.approx(0.25)
    for expected, actual in zip(source_model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.state
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert restored_optimizer.param_groups[0]["weight_decay"] == pytest.approx(2e-4)


def test_training_config_rejects_missing_resume_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(resume_from=tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError, match="Resume checkpoint does not exist"):
        config.validate()
