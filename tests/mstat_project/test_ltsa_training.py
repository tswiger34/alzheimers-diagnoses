from pathlib import Path

import torch
from ltsa.ltsa_model import LTSA
from torch.utils.data import DataLoader

from mstat_project.ml.landmark_baseline import train_baseline_epoch
from mstat_project.ml.landmarks import (
    DiscreteTimeGrid,
    LandmarkPatientRecord,
    LandmarkSequenceDataset,
    collate_landmark_samples,
)
from mstat_project.ml.ltsa import LTSATrainingConfig, _run_epoch, save_ltsa_checkpoint
from mstat_project.ml.results import EpochMetrics


class DummyImageEncoder(torch.nn.Module):
    n_features = 8

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=tuple(range(1, images.ndim))).unsqueeze(1)
        return pooled.repeat(1, self.n_features)


class TinyCoxModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.scale * images.mean(dim=(1, 2, 3, 4))


def _training_loader(tmp_path: Path) -> tuple[list[LandmarkPatientRecord], DataLoader]:
    records = []
    for index, duration in enumerate((6.0, 12.0, 18.0), start=1):
        ptid = f"P{index:03d}"
        image_ids = (f"{index}0", f"{index}1")
        images = torch.randn(2, 1, 16, 16, 16) + index
        tensor_path = tmp_path / f"{ptid}.pt"
        torch.save({"ptid": ptid, "img_ids": list(image_ids), "images": images}, tensor_path)
        records.append(
            LandmarkPatientRecord(
                ptid=ptid,
                split="train",
                landmark_months=12.0,
                observed_time_months=duration,
                event_observed=index < 3,
                image_ids=image_ids,
                relative_times=(0.0, 12.0),
                tensor_path=tensor_path,
            )
        )
    loader = DataLoader(
        LandmarkSequenceDataset(records, spatial_size=(16, 16, 16)),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_landmark_samples,
    )
    return records, loader


def test_tiny_ltsa_and_baseline_complete_training_epochs(tmp_path: Path) -> None:
    records, loader = _training_loader(tmp_path)
    grid = DiscreteTimeGrid.fit(records, bin_width_months=6.0)
    ltsa = LTSA(
        DummyImageEncoder(),  # type: ignore[arg-type]
        n_heads=2,
        dropout=0.0,
        n_layers=1,
        max_sequence_length=2,
        max_time_index=12.0,
        n_time_bins=grid.n_time_bins,
    )
    ltsa_optimizer = torch.optim.AdamW(ltsa.parameters(), lr=1e-3)
    config = LTSATrainingConfig(
        landmark_months=12.0,
        epochs=1,
        patience=1,
        batch_size=2,
        pretrained=False,
        tensor_dir=tmp_path,
        spatial_size=(16, 16, 16),
    )

    train_metrics = _run_epoch(
        ltsa,
        loader,
        grid,
        config=config,
        device=torch.device("cpu"),
        optimizer=ltsa_optimizer,
    )
    validation_metrics = _run_epoch(
        ltsa,
        loader,
        grid,
        config=config,
        device=torch.device("cpu"),
        optimizer=None,
    )

    assert all(torch.isfinite(torch.tensor(train_metrics)))
    assert all(torch.isfinite(torch.tensor(validation_metrics)))

    baseline = TinyCoxModel()
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    loss, c_index = train_baseline_epoch(
        baseline,  # type: ignore[arg-type]
        loader,
        baseline_optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=5.0,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(c_index))

    checkpoint_path = tmp_path / "checkpoints" / "epoch_001.pt"
    save_ltsa_checkpoint(
        checkpoint_path,
        run_id="test-run",
        epoch=1,
        model=ltsa,
        optimizer=ltsa_optimizer,
        config=config,
        grid=grid,
        metrics=EpochMetrics(
            epoch=1,
            train_total_loss=train_metrics[0],
            train_survival_loss=train_metrics[1],
            train_auxiliary_loss=train_metrics[2],
            train_c_index=train_metrics[3],
            validation_total_loss=validation_metrics[0],
            validation_survival_loss=validation_metrics[1],
            validation_auxiliary_loss=validation_metrics[2],
            validation_c_index=validation_metrics[3],
            learning_rate=1e-3,
            checkpoint_path=checkpoint_path,
        ),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["run_id"] == "test-run"
    assert checkpoint["time_grid"]["n_time_bins"] == grid.n_time_bins
    assert set(checkpoint["model_state_dict"]) == set(ltsa.state_dict())
