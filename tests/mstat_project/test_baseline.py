import math
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
import torch

import mstat_project.ml.baseline as baseline_module
from mstat_project.ml.baseline import (
    BaselineResultStore,
    PatientRecord,
    SingleImageSurvivalDataset,
    SingleImageSurvivalModel,
    _parse_args,
    build_patient_records,
    concordance_index,
    cox_ph_loss,
    load_initial_weights,
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


def test_training_config_rejects_missing_initial_weights_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(initial_weights_from=tmp_path / "missing.pt")

    with pytest.raises(FileNotFoundError, match="Initial-weights checkpoint does not exist"):
        config.validate()


def test_load_initial_weights_starts_with_model_state_only(tmp_path: Path) -> None:
    source_model = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    loss = source_model(torch.randn(2, 2)).square().mean()
    loss.backward()
    source_optimizer.step()

    checkpoint_path = tmp_path / "source" / "epoch_010.pt"
    save_checkpoint(
        checkpoint_path,
        "source-run",
        10,
        source_model,
        source_optimizer,
        TrainingConfig(epochs=10),
        {"validation_loss": 0.25},
    )

    initialized_model = torch.nn.Linear(2, 1)
    fresh_optimizer = torch.optim.AdamW(initialized_model.parameters(), lr=2e-3)
    source_run_id = load_initial_weights(
        checkpoint_path,
        model=initialized_model,
        device=torch.device("cpu"),
    )

    assert source_run_id == "source-run"
    for expected, actual in zip(source_model.parameters(), initialized_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert fresh_optimizer.state == {}
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(2e-3)


def test_training_config_rejects_conflicting_checkpoint_modes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "epoch_001.pt"
    checkpoint_path.touch()
    config = TrainingConfig(
        resume_from=checkpoint_path,
        initial_weights_from=checkpoint_path,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_parse_args_rejects_conflicting_checkpoint_modes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "epoch_001.pt"

    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--resume-from",
                str(checkpoint_path),
                "--initial-weights-from",
                str(checkpoint_path),
            ]
        )


def test_result_store_records_initial_weight_provenance(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source-run" / "epoch_010.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.touch()
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    record = PatientRecord(
        ptid="P001",
        image_id="10",
        split="train",
        observed_time_months=12.0,
        event_observed=True,
        tensor_path=tmp_path / "P001.pt",
    )
    config = TrainingConfig(initial_weights_from=source_checkpoint)

    BaselineResultStore(engine).start_run(
        "new-run",
        config,
        torch.device("cpu"),
        tmp_path / "new-run",
        [record],
        initial_weights_source_run_id="source-run",
    )

    run_insert = connection.execute.call_args_list[0]
    statement = str(run_insert.args[0])
    parameters = run_insert.args[1]
    assert "initial_weights_kind" in statement
    assert "initial_weights_source_run_id" in statement
    assert "initial_weights_checkpoint_path" in statement
    assert parameters["initial_weights_kind"] == "baseline_checkpoint"
    assert parameters["initial_weights_source_run_id"] == "source-run"
    assert parameters["initial_weights_checkpoint_path"] == str(source_checkpoint.resolve())


def test_result_store_schema_migrates_initial_weight_provenance_columns() -> None:
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value

    BaselineResultStore(engine).ensure_schema()

    statements = "\n".join(str(call.args[0]) for call in connection.execute.call_args_list)
    assert "ADD COLUMN IF NOT EXISTS initial_weights_kind" in statements
    assert "ADD COLUMN IF NOT EXISTS initial_weights_source_run_id" in statements
    assert "ADD COLUMN IF NOT EXISTS initial_weights_checkpoint_path" in statements


def test_run_training_initializes_an_independent_run_at_epoch_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source_model.weight.fill_(1.5)
        source_model.bias.fill_(-0.5)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    source_checkpoint = tmp_path / "source" / "epoch_010.pt"
    save_checkpoint(
        source_checkpoint,
        "source-run",
        10,
        source_model,
        source_optimizer,
        TrainingConfig(epochs=10),
        {"validation_loss": 0.25},
    )

    records = [
        PatientRecord(
            ptid=f"P{index:03d}",
            image_id=str(index),
            split=split,
            observed_time_months=float(index),
            event_observed=True,
            tensor_path=tmp_path / f"P{index:03d}.pt",
        )
        for index, split in enumerate(("train", "validation", "test"), start=1)
    ]
    result_store = MagicMock()
    initialized_with = []

    def make_model(*, weights: object) -> torch.nn.Module:
        initialized_with.append(weights)
        return torch.nn.Linear(2, 1)

    def train_one_epoch(
        model: torch.nn.Module,
        loader: list[PatientRecord],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gradient_clip_norm: float,
    ) -> tuple[float, float]:
        del device, gradient_clip_norm
        assert loader[0].split == "train"
        assert optimizer.state == {}
        for expected, actual in zip(source_model.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        return 0.4, 0.5

    def evaluate(
        model: torch.nn.Module,
        loader: list[PatientRecord],
        device: torch.device,
    ) -> tuple[float, float, baseline_module.PredictionBundle]:
        del model, device
        record = loader[0]
        predictions = baseline_module.PredictionBundle(
            risks=torch.tensor([0.1]),
            times=torch.tensor([record.observed_time_months]),
            events=torch.tensor([record.event_observed]),
            indices=torch.tensor([0]),
            ptids=[record.ptid],
            image_ids=[record.image_id],
        )
        return 0.3, 0.6, predictions

    monkeypatch.setattr(baseline_module, "get_last_scan", lambda engine: None)
    monkeypatch.setattr(
        baseline_module,
        "build_patient_records",
        lambda last_scans, tensor_dir: records,
    )
    monkeypatch.setattr(
        baseline_module,
        "_make_loader",
        lambda split_records, config, device: split_records,
    )
    monkeypatch.setattr(baseline_module, "SingleImageSurvivalModel", make_model)
    monkeypatch.setattr(baseline_module, "BaselineResultStore", lambda engine: result_store)
    monkeypatch.setattr(baseline_module, "train_one_epoch", train_one_epoch)
    monkeypatch.setattr(baseline_module, "evaluate", evaluate)

    run_id = baseline_module.run_training(
        TrainingConfig(
            epochs=1,
            learning_rate=2e-3,
            device="cpu",
            checkpoint_root=tmp_path / "new-runs",
            initial_weights_from=source_checkpoint,
        ),
        engine=MagicMock(),
    )

    assert initialized_with == [None]
    assert run_id != "source-run"
    assert (tmp_path / "new-runs" / run_id / "epoch_001.pt").is_file()
    assert result_store.start_run.call_args.kwargs["initial_weights_source_run_id"] == "source-run"
    assert result_store.record_epoch.call_args.args[1] == 1
    assert result_store.complete_run.call_args.args[1] == 1
