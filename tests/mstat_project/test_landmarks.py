from pathlib import Path

import polars as pl
import pytest
import torch

from mstat_project.ml.landmarks import (
    DiscreteTimeGrid,
    LandmarkCohortConfig,
    LandmarkPatientRecord,
    LandmarkSequenceDataset,
    build_landmark_records,
    collate_landmark_samples,
)
from mstat_project.ml.metrics import paired_c_index_difference


def _write_patient_tensor(path: Path, ptid: str, image_ids: list[str]) -> None:
    images = torch.arange(len(image_ids) * 16**3, dtype=torch.float32).reshape(len(image_ids), 1, 16, 16, 16)
    torch.save({"ptid": ptid, "img_ids": image_ids, "images": images}, path / f"{ptid}.pt")


def _landmark_frame(tmp_path: Path) -> pl.DataFrame:
    rows = []
    splits = ("train", "train", "validation", "validation", "test", "test")
    for patient_number, split in enumerate(splits, start=1):
        ptid = f"P{patient_number:03d}"
        image_ids = [f"{patient_number}0", f"{patient_number}1"]
        _write_patient_tensor(tmp_path, ptid, image_ids)
        for image_id, relative_time in zip(image_ids, (0.0, 12.0), strict=True):
            rows.append(
                {
                    "image_id": image_id,
                    "ptid": ptid,
                    "months_since_baseline_image": relative_time,
                    "months_to_ad_from_baseline": 30.0 + patient_number,
                    "is_censored": False,
                    "train_test_split": split,
                }
            )
    return pl.DataFrame(rows)


def test_build_landmark_records_and_collate_prefixes(tmp_path: Path) -> None:
    config = LandmarkCohortConfig(
        landmark_months=12.0,
        tensor_dir=tmp_path,
        spatial_size=(16, 16, 16),
    )
    records = build_landmark_records(_landmark_frame(tmp_path), config)

    assert len(records) == 6
    assert records[0].image_ids == ("10", "11")
    assert records[0].relative_times == (0.0, 12.0)
    assert records[0].observed_time_months == pytest.approx(19.0)

    dataset = LandmarkSequenceDataset(records, spatial_size=(16, 16, 16))
    batch = collate_landmark_samples([dataset[0], dataset[1]])

    assert batch.images.shape == (2, 2, 1, 16, 16, 16)
    assert batch.sequence_lengths.tolist() == [2, 2]
    assert batch.relative_times.tolist() == [[0.0, 12.0], [0.0, 12.0]]
    torch.testing.assert_close(batch.images.mean(dim=(2, 3, 4, 5)), torch.zeros(2, 2), atol=1e-5, rtol=0)


def test_landmark_records_reject_patient_not_at_risk(tmp_path: Path) -> None:
    _write_patient_tensor(tmp_path, "P001", ["10"])
    frame = pl.DataFrame(
        [
            {
                "image_id": "10",
                "ptid": "P001",
                "months_since_baseline_image": 0.0,
                "months_to_ad_from_baseline": 12.0,
                "is_censored": False,
                "train_test_split": "train",
            }
        ]
    )

    with pytest.raises(ValueError, match="not at risk"):
        build_landmark_records(
            frame,
            LandmarkCohortConfig(landmark_months=12.0, tensor_dir=tmp_path),
        )


def test_discrete_time_grid_reserves_overflow_bin() -> None:
    records = [
        LandmarkPatientRecord("A", "train", 0.0, 5.9, True, ("1",), (0.0,), Path("A.pt")),
        LandmarkPatientRecord("B", "train", 0.0, 18.0, True, ("2",), (0.0,), Path("B.pt")),
    ]
    grid = DiscreteTimeGrid.fit(records, bin_width_months=6.0)

    assert grid.n_time_bins == 5
    assert grid.encode(torch.tensor([0.0, 5.9, 6.0, 18.0, 120.0])).tolist() == [0, 0, 1, 3, 4]
    assert grid.risk_from_survival(torch.ones(1, 5)).item() == pytest.approx(-30.0)


def test_paired_c_index_bootstrap_uses_aligned_patients() -> None:
    times = torch.tensor([1.0, 2.0, 3.0, 4.0])
    events = torch.tensor([True, True, True, False])
    baseline = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ltsa = torch.tensor([4.0, 3.0, 2.0, 1.0])

    comparison = paired_c_index_difference(
        baseline,
        ltsa,
        times,
        events,
        bootstrap_samples=100,
        seed=42,
    )

    assert comparison.baseline_c_index == pytest.approx(0.0)
    assert comparison.ltsa_c_index == pytest.approx(1.0)
    assert comparison.difference == pytest.approx(1.0)
    assert comparison.bootstrap_samples > 0
