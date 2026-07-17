"""Shared fixed-landmark cohort and tensor loading logic."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from sqlalchemy import Engine, text
from torch import Tensor
from torch.utils.data import Dataset

from mstat_project.ml.utils import default_tensor_dir

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class LandmarkCohortConfig:
    landmark_months: float
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        if self.landmark_months < 0:
            raise ValueError("landmark_months cannot be negative")
        if self.spatial_size is not None and any(size < 16 for size in self.spatial_size):
            raise ValueError("all spatial dimensions must be at least 16")


@dataclass(frozen=True)
class LandmarkPatientRecord:
    ptid: str
    split: str
    landmark_months: float
    observed_time_months: float
    event_observed: bool
    image_ids: tuple[str, ...]
    relative_times: tuple[float, ...]
    tensor_path: Path

    @property
    def selected_image_id(self) -> str:
        return self.image_ids[-1]


@dataclass(frozen=True)
class LandmarkSample:
    images: Tensor
    relative_times: Tensor
    observed_time_months: Tensor
    event_observed: Tensor
    ptid: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class LandmarkBatch:
    images: Tensor
    sequence_lengths: Tensor
    relative_times: Tensor
    observed_times: Tensor
    events: Tensor
    ptids: list[str]
    image_ids: list[tuple[str, ...]]

    def to(self, device: torch.device) -> "LandmarkBatch":
        return LandmarkBatch(
            images=self.images.to(device=device, non_blocking=True),
            sequence_lengths=self.sequence_lengths.to(device=device, non_blocking=True),
            relative_times=self.relative_times.to(device=device, non_blocking=True),
            observed_times=self.observed_times.to(device=device, non_blocking=True),
            events=self.events.to(device=device, non_blocking=True),
            ptids=self.ptids,
            image_ids=self.image_ids,
        )


@dataclass(frozen=True)
class DiscreteTimeGrid:
    bin_width_months: float
    n_time_bins: int

    @classmethod
    def fit(
        cls,
        records: list[LandmarkPatientRecord],
        *,
        bin_width_months: float,
    ) -> "DiscreteTimeGrid":
        if bin_width_months <= 0:
            raise ValueError("bin_width_months must be positive")
        if not records:
            raise ValueError("Cannot fit a time grid without training records")
        max_duration = max(record.observed_time_months for record in records)
        return cls(
            bin_width_months=bin_width_months,
            # The maximum training duration receives an ordinary bin; the final
            # index is reserved for durations beyond the training support.
            n_time_bins=max(2, int(max_duration // bin_width_months) + 2),
        )

    def encode(self, durations: Tensor) -> Tensor:
        if (durations < 0).any() or not torch.isfinite(durations).all():
            raise ValueError("durations must contain finite non-negative values")
        return torch.floor(durations / self.bin_width_months).to(torch.long).clamp_max(self.n_time_bins - 1)

    def risk_from_survival(self, survival: Tensor) -> Tensor:
        if survival.shape[-1] != self.n_time_bins:
            raise ValueError(f"Expected {self.n_time_bins} survival bins, got {survival.shape[-1]}")
        return -self.bin_width_months * survival.sum(dim=-1)


def load_landmark_frame(engine: Engine, *, landmark_months: float) -> pl.DataFrame:
    """Load eligible, chronologically ordered images for one landmark."""

    query = text(
        """
        SELECT
            imgs.image_id,
            imgs.ptid,
            imgs.months_since_baseline_image,
            imgs.months_to_ad_from_baseline,
            imgs.is_censored,
            tts.split::TEXT AS train_test_split
        FROM _core.core_image_set AS imgs
        INNER JOIN _raw.train_test_split AS tts
            ON tts.ptid = imgs.ptid
        WHERE imgs.mri_is_valid
            AND imgs.months_since_baseline_image <= :landmark_months
            AND imgs.months_to_ad_from_baseline > :landmark_months
        ORDER BY imgs.ptid, imgs.image_date, imgs.image_id
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(query, {"landmark_months": landmark_months}).mappings().all()
    return pl.DataFrame(rows)


def build_landmark_records(
    frame: pl.DataFrame,
    config: LandmarkCohortConfig,
) -> list[LandmarkPatientRecord]:
    """Validate a landmark frame and build one longitudinal record per patient."""

    config.validate()
    required = {
        "image_id",
        "ptid",
        "months_since_baseline_image",
        "months_to_ad_from_baseline",
        "is_censored",
        "train_test_split",
    }
    if frame.is_empty() or not required.issubset(frame.columns):
        raise ValueError("Landmark data is empty or missing required columns")
    if any(frame[column].null_count() for column in required):
        raise ValueError("Landmark data contains null required values")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frame.iter_rows(named=True):
        ptid = str(row["ptid"])
        grouped.setdefault(ptid, []).append(row)

    records: list[LandmarkPatientRecord] = []
    missing_tensors: list[Path] = []
    for ptid, rows in grouped.items():
        outcome_times = {float(row["months_to_ad_from_baseline"]) for row in rows}
        censor_values = {bool(row["is_censored"]) for row in rows}
        split_values = {str(row["train_test_split"]).lower() for row in rows}
        split_values = {"validation" if value == "val" else value for value in split_values}
        if len(outcome_times) != 1 or len(censor_values) != 1 or len(split_values) != 1:
            raise ValueError(f"Patient {ptid} has inconsistent outcome or split metadata")
        split = next(iter(split_values))
        if split not in SPLIT_NAMES:
            raise ValueError(f"Patient {ptid} has invalid split '{split}'")
        relative_times = tuple(float(row["months_since_baseline_image"]) for row in rows)
        if any(time_value < 0 or time_value > config.landmark_months for time_value in relative_times):
            raise ValueError(f"Patient {ptid} has an image outside the landmark window")
        if any(current < prior for prior, current in zip(relative_times, relative_times[1:], strict=False)):
            raise ValueError(f"Patient {ptid} images are not chronologically ordered")

        tensor_path = config.tensor_dir / f"{ptid}.pt"
        if not tensor_path.is_file():
            missing_tensors.append(tensor_path)
            continue
        residual_duration = next(iter(outcome_times)) - config.landmark_months
        if residual_duration <= 0:
            raise ValueError(f"Patient {ptid} is not at risk after landmark {config.landmark_months}")
        records.append(
            LandmarkPatientRecord(
                ptid=ptid,
                split=split,
                landmark_months=config.landmark_months,
                observed_time_months=residual_duration,
                event_observed=not next(iter(censor_values)),
                image_ids=tuple(str(row["image_id"]) for row in rows),
                relative_times=relative_times,
                tensor_path=tensor_path,
            )
        )

    if missing_tensors:
        examples = ", ".join(str(path) for path in missing_tensors[:3])
        raise FileNotFoundError(f"Missing {len(missing_tensors)} patient tensor files. Examples: {examples}")
    for split in SPLIT_NAMES:
        split_records = [record for record in records if record.split == split]
        if not split_records:
            raise ValueError(f"Landmark split '{split}' is empty")
        if not any(record.event_observed for record in split_records):
            raise ValueError(f"Landmark split '{split}' contains no observed events")
        if not any(
            event_record.event_observed and event_record.observed_time_months < other_record.observed_time_months
            for event_record in split_records
            for other_record in split_records
        ):
            raise ValueError(f"Landmark split '{split}' contains no comparable survival pairs")
    return records


class LandmarkSequenceDataset(Dataset[LandmarkSample]):
    def __init__(
        self,
        records: list[LandmarkPatientRecord],
        *,
        spatial_size: tuple[int, int, int] | None,
    ) -> None:
        self.records = records
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LandmarkSample:
        record = self.records[index]
        package = torch.load(record.tensor_path, map_location="cpu", weights_only=True, mmap=True)
        if str(package.get("ptid")) != record.ptid:
            raise ValueError(f"Tensor {record.tensor_path} does not belong to {record.ptid}")
        package_image_ids = [str(image_id) for image_id in package["img_ids"]]
        missing_image_ids = [image_id for image_id in record.image_ids if image_id not in package_image_ids]
        if missing_image_ids:
            raise ValueError(f"Tensor {record.tensor_path} is missing images {missing_image_ids}")
        image_indices = [package_image_ids.index(image_id) for image_id in record.image_ids]
        images = package["images"][image_indices].to(dtype=torch.float32).clone()
        if images.ndim != 5 or images.shape[1] != 1:
            raise ValueError(f"Expected images shaped [visits, 1, D, H, W], got {tuple(images.shape)}")
        if self.spatial_size is not None and tuple(images.shape[2:]) != self.spatial_size:
            images = F.interpolate(
                images,
                size=self.spatial_size,
                mode="trilinear",
                align_corners=False,
            )
        image_means = images.mean(dim=(1, 2, 3, 4), keepdim=True)
        image_stds = images.std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
        return LandmarkSample(
            images=(images - image_means) / image_stds,
            relative_times=torch.tensor(record.relative_times, dtype=torch.float32),
            observed_time_months=torch.tensor(record.observed_time_months, dtype=torch.float32),
            event_observed=torch.tensor(record.event_observed, dtype=torch.bool),
            ptid=record.ptid,
            image_ids=record.image_ids,
        )


def collate_landmark_samples(samples: list[LandmarkSample]) -> LandmarkBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    max_visits = max(sample.images.shape[0] for sample in samples)
    padded_images: list[Tensor] = []
    padded_times: list[Tensor] = []
    for sample in samples:
        visit_padding = max_visits - sample.images.shape[0]
        padded_images.append(F.pad(sample.images, (0, 0, 0, 0, 0, 0, 0, 0, 0, visit_padding)))
        padded_times.append(F.pad(sample.relative_times, (0, visit_padding)))
    return LandmarkBatch(
        images=torch.stack(padded_images),
        sequence_lengths=torch.tensor([sample.images.shape[0] for sample in samples], dtype=torch.long),
        relative_times=torch.stack(padded_times),
        observed_times=torch.stack([sample.observed_time_months for sample in samples]),
        events=torch.stack([sample.event_observed for sample in samples]),
        ptids=[sample.ptid for sample in samples],
        image_ids=[sample.image_ids for sample in samples],
    )
