"""Patient-level single-MRI Cox survival baseline.

The baseline uses exactly one MRI and one outcome per patient.  For patients
who convert to Alzheimer's disease (AD), the image is the final eligible MRI
strictly before the first AD diagnosis.  For censored patients, it is their
last observed MRI.  Survival time is measured once per patient from baseline
to AD diagnosis or censoring.

Run with::

    python -m mstat_project.ml.baseline --epochs 10 --batch-size 2

Every epoch is checkpointed below ``data/artifacts/model_checkpoints/baseline``
by default.  Run metadata, epoch metrics, the patient cohort, and test
predictions are stored in the Postgres ``_ml`` schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sqlalchemy import Engine, text
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet101_Weights, resnet101

from mstat_project.utils import get_db_engine

load_dotenv()


def _default_tensor_dir() -> Path:
    return Path(os.getenv("IMAGES_PATH", "data/images")) / "tensors"


def _default_checkpoint_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data")) / "artifacts" / "model_checkpoints" / "baseline"


TENSOR_PATH = _default_tensor_dir()
CHECKPOINT_PATH = _default_checkpoint_dir()
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration persisted with every experiment run."""

    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    tensor_dir: Path = field(default_factory=_default_tensor_dir)
    checkpoint_root: Path = field(default_factory=_default_checkpoint_dir)
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.spatial_size is not None and any(size < 16 for size in self.spatial_size):
            raise ValueError("all spatial dimensions must be at least 16")


@dataclass(frozen=True)
class PatientRecord:
    """The single image and single survival outcome used for one patient."""

    ptid: str
    image_id: str
    split: str
    observed_time_months: float
    event_observed: bool
    tensor_path: Path


@dataclass
class PredictionBundle:
    risks: torch.Tensor
    times: torch.Tensor
    events: torch.Tensor
    indices: torch.Tensor
    ptids: list[str]
    image_ids: list[str]


class SingleImageSurvivalModel(nn.Module):
    """ImageNet-pretrained ResNet-101 producing one Cox risk per MRI.

    Torchvision's ResNet-101 is a 2D, three-channel network. Each 3D MRI is
    therefore represented by its axial, coronal, and sagittal center slices,
    stacked as the three input channels. This preserves the pretrained first
    convolution instead of replacing it with a randomly initialized layer.
    """

    def __init__(
        self,
        weights: ResNet101_Weights | None = ResNet101_Weights.IMAGENET1K_V2,
    ):
        super().__init__()
        self.encoder = resnet101(weights=weights)
        encoded_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.risk_head = nn.Linear(encoded_features, 1)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True) -> SingleImageSurvivalModel:
        """Set training mode while retaining pretrained BatchNorm statistics."""

        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _volume_to_resnet_input(self, x: torch.Tensor) -> torch.Tensor:
        volume = x[:, 0]
        depth_center = volume.shape[1] // 2
        height_center = volume.shape[2] // 2
        width_center = volume.shape[3] // 2

        orthogonal_slices = (
            volume[:, depth_center, :, :],
            volume[:, :, height_center, :],
            volume[:, :, :, width_center],
        )
        resized_slices = [
            F.interpolate(
                image_slice.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(1)
            for image_slice in orthogonal_slices
        ]
        resnet_input = torch.stack(resized_slices, dim=1)

        image_min = resnet_input.amin(dim=(1, 2, 3), keepdim=True)
        image_max = resnet_input.amax(dim=(1, 2, 3), keepdim=True)
        resnet_input = (resnet_input - image_min) / (image_max - image_min).clamp_min(1e-6)
        imagenet_mean = self.get_buffer("imagenet_mean")
        imagenet_std = self.get_buffer("imagenet_std")
        return (resnet_input - imagenet_mean) / imagenet_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [batch, channels, depth, height, width], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected one MRI channel, got {x.shape[1]}")
        features = self.encoder(self._volume_to_resnet_input(x))
        return self.risk_head(features).squeeze(-1)


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


def get_last_scan(engine: Engine | None = None) -> pl.DataFrame:
    """Load one eligible pre-event/censoring image and outcome per patient."""

    query = """
        WITH eligible_images AS (
            SELECT
                imgs.image_id,
                imgs.ptid,
                imgs.image_date,
                imgs.baseline_diagnosis,
                imgs.final_diagnosis,
                imgs.is_censored,
                imgs.months_to_ad_from_baseline AS observed_time_months,
                tts.split::TEXT AS train_test_split,
                ROW_NUMBER() OVER (
                    PARTITION BY imgs.ptid
                    ORDER BY imgs.image_date DESC, imgs.image_id DESC
                ) AS image_rank
            FROM _core.core_image_set AS imgs
            INNER JOIN _raw.train_test_split AS tts
                ON tts.ptid = imgs.ptid
            WHERE imgs.mri_is_valid
        )
        SELECT
            image_id,
            ptid,
            image_date,
            baseline_diagnosis,
            final_diagnosis,
            is_censored,
            observed_time_months,
            train_test_split
        FROM eligible_images
        WHERE image_rank = 1
        ORDER BY ptid
    """
    database_engine = engine or get_db_engine()
    with database_engine.connect() as connection:
        return pl.read_database(query, connection)


def validate_last_scan(df: pl.DataFrame) -> bool:
    """Return whether the frame satisfies the one-row-per-patient contract."""

    required = {
        "image_id",
        "ptid",
        "is_censored",
        "observed_time_months",
        "train_test_split",
    }
    if not required.issubset(df.columns) or df.is_empty():
        return False
    if df["ptid"].n_unique() != df.height:
        return False
    if any(df[column].null_count() for column in required):
        return False
    split_values = set(df["train_test_split"].cast(pl.String).str.to_lowercase().unique().to_list())
    split_values = {"validation" if value == "val" else value for value in split_values}
    return split_values.issubset(set(SPLIT_NAMES))


def df_transform(df: pl.DataFrame) -> pl.DataFrame:
    """Add the event indicator and normalize validation split naming."""

    return df.with_columns(
        (~pl.col("is_censored")).cast(pl.Boolean).alias("event_observed"),
        pl.col("final_diagnosis").eq("AD").alias("is_ad"),
        pl.when(pl.col("train_test_split").cast(pl.String).str.to_lowercase() == "val")
        .then(pl.lit("validation"))
        .otherwise(pl.col("train_test_split").cast(pl.String).str.to_lowercase())
        .alias("train_test_split"),
    )


def build_patient_records(df: pl.DataFrame, tensor_dir: Path) -> list[PatientRecord]:
    """Build and validate the immutable patient-level modeling cohort."""

    if not validate_last_scan(df):
        raise ValueError("Last-scan data must contain one complete row per patient and valid split labels")

    transformed = df_transform(df)
    negative_times = transformed.filter(pl.col("observed_time_months") < 0)
    if negative_times.height:
        raise ValueError(f"Found {negative_times.height} patients with negative baseline-to-outcome survival time")

    records: list[PatientRecord] = []
    missing_tensors: list[Path] = []
    for row in transformed.iter_rows(named=True):
        tensor_path = tensor_dir / f"{row['ptid']}.pt"
        if not tensor_path.is_file():
            missing_tensors.append(tensor_path)
            continue
        records.append(
            PatientRecord(
                ptid=str(row["ptid"]),
                image_id=str(row["image_id"]),
                split=str(row["train_test_split"]),
                observed_time_months=float(row["observed_time_months"]),
                event_observed=bool(row["event_observed"]),
                tensor_path=tensor_path,
            )
        )

    if missing_tensors:
        examples = ", ".join(str(path) for path in missing_tensors[:3])
        raise FileNotFoundError(
            f"Missing {len(missing_tensors)} patient tensor files in {tensor_dir}. Examples: {examples}"
        )

    for split in SPLIT_NAMES:
        split_records = [record for record in records if record.split == split]
        if not split_records:
            raise ValueError(f"Patient split '{split}' is empty")
        if not any(record.event_observed for record in split_records):
            raise ValueError(f"Patient split '{split}' contains no observed events")

    return records


class SingleImageSurvivalDataset(Dataset[dict[str, Any]]):
    """Lazily load the selected MRI from each longitudinal patient tensor."""

    def __init__(
        self,
        records: Sequence[PatientRecord],
        spatial_size: tuple[int, int, int] | None = (96, 112, 96),
    ):
        self.records = list(records)
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        package = torch.load(record.tensor_path, map_location="cpu", weights_only=True, mmap=True)

        package_ptid = str(package.get("ptid"))
        if package_ptid != record.ptid:
            raise ValueError(f"Tensor {record.tensor_path} belongs to {package_ptid}, expected {record.ptid}")

        image_ids = [str(image_id) for image_id in package["img_ids"]]
        try:
            image_index = image_ids.index(record.image_id)
        except ValueError as exc:
            raise ValueError(f"Image {record.image_id} is absent from {record.tensor_path}") from exc

        image = package["images"][image_index].to(dtype=torch.float32).clone()
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected a [1, D, H, W] MRI in {record.tensor_path}, got {tuple(image.shape)}")
        if self.spatial_size is not None and tuple(image.shape[1:]) != self.spatial_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.spatial_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        image_mean = image.mean()
        image_std = image.std().clamp_min(1e-6)
        image = (image - image_mean) / image_std

        return {
            "image": image,
            "time": torch.tensor(record.observed_time_months, dtype=torch.float32),
            "event": torch.tensor(record.event_observed, dtype=torch.bool),
            "index": torch.tensor(index, dtype=torch.long),
            "ptid": record.ptid,
            "image_id": record.image_id,
        }


class BaselineResultStore:
    """Persist reproducible baseline run results in Postgres."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def ensure_schema(self) -> None:
        statements = [
            "CREATE SCHEMA IF NOT EXISTS _ml",
            """
            CREATE TABLE IF NOT EXISTS _ml.baseline_runs (
                run_id UUID PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                config JSONB NOT NULL,
                device TEXT NOT NULL,
                checkpoint_dir TEXT NOT NULL,
                train_patients INTEGER NOT NULL,
                validation_patients INTEGER NOT NULL,
                test_patients INTEGER NOT NULL,
                train_events INTEGER NOT NULL,
                validation_events INTEGER NOT NULL,
                test_events INTEGER NOT NULL,
                best_epoch INTEGER,
                best_validation_loss DOUBLE PRECISION,
                test_c_index DOUBLE PRECISION,
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.baseline_run_patients (
                run_id UUID NOT NULL REFERENCES _ml.baseline_runs(run_id) ON DELETE CASCADE,
                ptid TEXT NOT NULL,
                image_id TEXT NOT NULL,
                split TEXT NOT NULL,
                observed_time_months DOUBLE PRECISION NOT NULL,
                event_observed BOOLEAN NOT NULL,
                PRIMARY KEY (run_id, ptid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.baseline_epoch_metrics (
                run_id UUID NOT NULL REFERENCES _ml.baseline_runs(run_id) ON DELETE CASCADE,
                epoch INTEGER NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                train_loss DOUBLE PRECISION NOT NULL,
                train_c_index DOUBLE PRECISION,
                validation_loss DOUBLE PRECISION NOT NULL,
                validation_c_index DOUBLE PRECISION,
                learning_rate DOUBLE PRECISION NOT NULL,
                checkpoint_path TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.baseline_test_predictions (
                run_id UUID NOT NULL REFERENCES _ml.baseline_runs(run_id) ON DELETE CASCADE,
                ptid TEXT NOT NULL,
                image_id TEXT NOT NULL,
                observed_time_months DOUBLE PRECISION NOT NULL,
                event_observed BOOLEAN NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (run_id, ptid)
            )
            """,
        ]
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))

    def start_run(
        self,
        run_id: str,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
        records: Sequence[PatientRecord],
    ) -> None:
        counts = {split: sum(record.split == split for record in records) for split in SPLIT_NAMES}
        event_counts = {
            split: sum(record.split == split and record.event_observed for record in records)
            for split in SPLIT_NAMES
        }
        config_json = json.dumps(asdict(config), default=str)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.baseline_runs (
                        run_id, status, config, device, checkpoint_dir,
                        train_patients, validation_patients, test_patients,
                        train_events, validation_events, test_events
                    ) VALUES (
                        :run_id, 'running', CAST(:config AS JSONB), :device, :checkpoint_dir,
                        :train_patients, :validation_patients, :test_patients,
                        :train_events, :validation_events, :test_events
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "config": config_json,
                    "device": str(device),
                    "checkpoint_dir": str(checkpoint_dir.resolve()),
                    "train_patients": counts["train"],
                    "validation_patients": counts["validation"],
                    "test_patients": counts["test"],
                    "train_events": event_counts["train"],
                    "validation_events": event_counts["validation"],
                    "test_events": event_counts["test"],
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.baseline_run_patients (
                        run_id, ptid, image_id, split, observed_time_months, event_observed
                    ) VALUES (
                        :run_id, :ptid, :image_id, :split, :observed_time_months, :event_observed
                    )
                    """
                ),
                [
                    {
                        "run_id": run_id,
                        "ptid": record.ptid,
                        "image_id": record.image_id,
                        "split": record.split,
                        "observed_time_months": record.observed_time_months,
                        "event_observed": record.event_observed,
                    }
                    for record in records
                ],
            )

    def record_epoch(
        self,
        run_id: str,
        epoch: int,
        train_loss: float,
        train_c_index: float,
        validation_loss: float,
        validation_c_index: float,
        learning_rate: float,
        checkpoint_path: Path,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.baseline_epoch_metrics (
                        run_id, epoch, train_loss, train_c_index, validation_loss,
                        validation_c_index, learning_rate, checkpoint_path
                    ) VALUES (
                        :run_id, :epoch, :train_loss, :train_c_index, :validation_loss,
                        :validation_c_index, :learning_rate, :checkpoint_path
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_c_index": train_c_index,
                    "validation_loss": validation_loss,
                    "validation_c_index": validation_c_index,
                    "learning_rate": learning_rate,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                },
            )

    def record_test_predictions(self, run_id: str, predictions: PredictionBundle) -> None:
        rows = [
            {
                "run_id": run_id,
                "ptid": ptid,
                "image_id": image_id,
                "observed_time_months": float(observed_time),
                "event_observed": bool(event),
                "risk_score": float(risk),
            }
            for ptid, image_id, observed_time, event, risk in zip(
                predictions.ptids,
                predictions.image_ids,
                predictions.times.tolist(),
                predictions.events.tolist(),
                predictions.risks.tolist(),
                strict=True,
            )
        ]
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.baseline_test_predictions (
                        run_id, ptid, image_id, observed_time_months, event_observed, risk_score
                    ) VALUES (
                        :run_id, :ptid, :image_id, :observed_time_months, :event_observed, :risk_score
                    )
                    """
                ),
                rows,
            )

    def complete_run(
        self,
        run_id: str,
        best_epoch: int,
        best_validation_loss: float,
        test_c_index: float,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE _ml.baseline_runs
                    SET status = 'completed', completed_at = NOW(), best_epoch = :best_epoch,
                        best_validation_loss = :best_validation_loss, test_c_index = :test_c_index
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "test_c_index": test_c_index,
                },
            )

    def fail_run(self, run_id: str, error_message: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE _ml.baseline_runs
                    SET status = 'failed', completed_at = NOW(), error_message = :error_message
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "error_message": error_message[:10_000]},
            )


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_loader(
    records: Sequence[PatientRecord],
    config: TrainingConfig,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    dataset = SingleImageSurvivalDataset(records, spatial_size=config.spatial_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> PredictionBundle:
    risks: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    events: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    ptids: list[str] = []
    image_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, non_blocking=True)
            risks.append(model(images).detach().cpu())
            times.append(batch["time"].detach().cpu())
            events.append(batch["event"].detach().cpu())
            indices.append(batch["index"].detach().cpu())
            ptids.extend(batch["ptid"])
            image_ids.extend(batch["image_id"])

    if not risks:
        raise ValueError("Cannot collect predictions from an empty data loader")
    return PredictionBundle(
        risks=torch.cat(risks),
        times=torch.cat(times),
        events=torch.cat(events),
        indices=torch.cat(indices),
        ptids=ptids,
        image_ids=image_ids,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    """Take one exact, memory-bounded full-risk-set Cox gradient step.

    The first pass computes every patient risk and the derivative of the full
    cohort Cox loss with respect to those risks.  The second pass recomputes
    small MRI batches and backpropagates those score derivatives through the
    CNN.  This avoids the biased risk sets and event-free batches produced by
    ordinary mini-batch Cox training.
    """

    model.train()
    predictions = collect_predictions(model, loader, device)
    risk_leaf = predictions.risks.detach().requires_grad_(True)
    loss = cox_ph_loss(risk_leaf, predictions.times, predictions.events)
    score_gradients = torch.autograd.grad(loss, risk_leaf)[0].detach()

    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        images = batch["image"].to(device=device, non_blocking=True)
        batch_indices = batch["index"].to(dtype=torch.long)
        batch_score_gradients = score_gradients[batch_indices].to(device=device)
        batch_risks = model(images)
        (batch_risks * batch_score_gradients).sum().backward()

    if gradient_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()

    return float(loss.item()), concordance_index(
        predictions.risks,
        predictions.times,
        predictions.events,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[float, float, PredictionBundle]:
    model.eval()
    predictions = collect_predictions(model, loader, device)
    loss = cox_ph_loss(predictions.risks, predictions.times, predictions.events)
    c_index = concordance_index(predictions.risks, predictions.times, predictions.events)
    return float(loss.item()), c_index, predictions


def save_checkpoint(
    checkpoint_path: Path,
    run_id: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "run_id": run_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def run_training(config: TrainingConfig, engine: Engine | None = None) -> str:
    """Train, validate, test, checkpoint, and persist one baseline run."""

    config.validate()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    database_engine = engine or get_db_engine()

    last_scans = get_last_scan(database_engine)
    records = build_patient_records(last_scans, config.tensor_dir)
    split_records = {split: [record for record in records if record.split == split] for split in SPLIT_NAMES}
    loaders = {split: _make_loader(split_records[split], config, device) for split in SPLIT_NAMES}

    run_id = str(uuid.uuid4())
    checkpoint_dir = config.checkpoint_root / run_id
    result_store = BaselineResultStore(database_engine)
    result_store.ensure_schema()
    result_store.start_run(run_id, config, device, checkpoint_dir, records)

    print(
        f"Run {run_id} on {device}: "
        + ", ".join(
            f"{split}={len(split_records[split])} ({sum(r.event_observed for r in split_records[split])} events)"
            for split in SPLIT_NAMES
        )
    )

    try:
        model = SingleImageSurvivalModel(weights=ResNet101_Weights.IMAGENET1K_V2).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_epoch = 0
        best_validation_loss = math.inf
        best_checkpoint: Path | None = None

        for epoch in range(1, config.epochs + 1):
            train_loss, train_c_index = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                config.gradient_clip_norm,
            )
            validation_loss, validation_c_index, _ = evaluate(
                model,
                loaders["validation"],
                device,
            )
            metrics = {
                "train_loss": train_loss,
                "train_c_index": train_c_index,
                "validation_loss": validation_loss,
                "validation_c_index": validation_c_index,
            }
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                checkpoint_path,
                run_id,
                epoch,
                model,
                optimizer,
                config,
                metrics,
            )
            result_store.record_epoch(
                run_id,
                epoch,
                train_loss,
                train_c_index,
                validation_loss,
                validation_c_index,
                optimizer.param_groups[0]["lr"],
                checkpoint_path,
            )

            if validation_loss < best_validation_loss:
                best_epoch = epoch
                best_validation_loss = validation_loss
                best_checkpoint = checkpoint_path

            print(
                f"Epoch {epoch:03d}/{config.epochs:03d} | "
                f"train loss {train_loss:.4f}, C-index {train_c_index:.4f} | "
                f"validation loss {validation_loss:.4f}, C-index {validation_c_index:.4f}"
            )

        if best_checkpoint is None:
            raise RuntimeError("Training did not produce a best checkpoint")
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_c_index, test_predictions = evaluate(model, loaders["test"], device)
        result_store.record_test_predictions(run_id, test_predictions)
        result_store.complete_run(run_id, best_epoch, best_validation_loss, test_c_index)
        print(
            f"Completed run {run_id}: best epoch={best_epoch}, "
            f"test C-index={test_c_index:.4f}, checkpoints={checkpoint_dir}"
        )
        return run_id
    except Exception as exc:
        try:
            result_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        except Exception as persistence_exc:
            exc.add_note(f"Additionally failed to mark Postgres run as failed: {persistence_exc}")
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--tensor-dir", type=Path, default=_default_tensor_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=_default_checkpoint_dir())
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        default=(96, 112, 96),
        help="CNN input size; use 182 218 182 to retain the preprocessed resolution",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    args = _parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        tensor_dir=args.tensor_dir,
        checkpoint_root=args.checkpoint_dir,
        spatial_size=tuple(args.spatial_size),
    )
    return run_training(config)


if __name__ == "__main__":
    main()
