"""Patient-level single-MRI Cox survival baseline.

The baseline uses exactly one MRI and one outcome per patient.  For patients
who convert to Alzheimer's disease (AD), the image is the final eligible MRI
strictly before the first AD diagnosis.  For censored patients, it is their
last observed MRI.  Survival time is measured once per patient from baseline
to AD diagnosis or censoring.

Run with::

    python -m mstat_project.ml.baseline --epochs 10 --batch-size 2

Continue a previous run through epoch 20 with::

    python -m mstat_project.ml.baseline \
        --resume-from data/artifacts/model_checkpoints/baseline/<run-id>/epoch_010.pt \
        --epochs 20

Start a separate run at epoch 1 using another run's model weights with::

    python -m mstat_project.ml.baseline \
        --initial-weights-from data/artifacts/model_checkpoints/baseline/<run-id>/epoch_010.pt \
        --learning-rate 0.0001 \
        --epochs 10

Every epoch is checkpointed below ``data/artifacts/model_checkpoints/baseline``
by default.  Run metadata, epoch metrics, the patient cohort, and test
predictions are stored in the Postgres ``_ml`` schema.
"""

from __future__ import annotations

import argparse
import json
import math
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
from ltsa.losses import cox_ph_loss
from sqlalchemy import Engine, text
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet101_Weights

from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.utils import get_db_engine

from .utils import concordance_index, default_checkpoint_dir, default_tensor_dir

load_dotenv()


TENSOR_PATH = default_tensor_dir()
CHECKPOINT_PATH = default_checkpoint_dir()
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration persisted with every legacy baseline experiment.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Number of patients processed per ResNet mini-batch.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight-decay coefficient.
        gradient_clip_norm: Maximum parameter-gradient norm. Values less than
            or equal to zero disable clipping.
        seed: Seed applied to Python, PyTorch, and CUDA random generators.
        num_workers: Worker processes assigned to each data loader.
        device: PyTorch device specification, or ``"auto"`` for automatic
            selection.
        tensor_dir: Directory containing patient tensor packages.
        checkpoint_root: Root directory for run-specific checkpoints.
        resume_from: Optional checkpoint whose model and optimizer state
            should be restored before training. ``epochs`` remains the total
            target epoch, rather than the number of additional epochs.
        initial_weights_from: Optional checkpoint whose model weights should
            initialize a separate run. The optimizer starts fresh and epoch
            numbering begins at one.
        spatial_size: Optional ``(depth, height, width)`` used to resize MRI
            volumes. ``None`` preserves stored dimensions.
    """

    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    checkpoint_root: Path = field(default_factory=default_checkpoint_dir)
    resume_from: Path | None = None
    initial_weights_from: Path | None = None
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        """Validate training, loading, and preprocessing settings.

        Raises:
            ValueError: If epoch or batch counts are not positive, optimizer
                settings are out of range, ``num_workers`` is negative, or a
                configured spatial dimension is smaller than 16 voxels.
        """

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
        if self.resume_from is not None and self.initial_weights_from is not None:
            raise ValueError("resume_from and initial_weights_from are mutually exclusive")
        if self.resume_from is not None and not self.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_from}")
        if self.initial_weights_from is not None and not self.initial_weights_from.is_file():
            raise FileNotFoundError(f"Initial-weights checkpoint does not exist: {self.initial_weights_from}")
        if self.spatial_size is not None and any(size < 16 for size in self.spatial_size):
            raise ValueError("all spatial dimensions must be at least 16")


@dataclass(frozen=True)
class PatientRecord:
    """Single selected MRI and survival outcome for one patient.

    Attributes:
        ptid: Patient identifier.
        image_id: Identifier of the selected MRI.
        split: Patient-level train, validation, or test assignment.
        observed_time_months: Baseline-to-event or baseline-to-censoring time.
        event_observed: Whether Alzheimer's disease conversion was observed.
        tensor_path: Path to the patient's longitudinal tensor package.
    """

    ptid: str
    image_id: str
    split: str
    observed_time_months: float
    event_observed: bool
    tensor_path: Path


@dataclass
class PredictionBundle:
    """Aligned scalar risks, outcomes, and identifiers for one split.

    Attributes:
        risks: Cox log-risk score for each patient.
        times: Baseline-relative event or censoring times.
        events: Boolean event indicators.
        indices: Dataset indices used to align two-pass score gradients.
        ptids: Patient identifiers in prediction order.
        image_ids: Selected MRI identifiers in prediction order.
    """

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
        """Initialize the shared MRI encoder and scalar Cox risk head.

        Args:
            weights: Optional torchvision ResNet-101 weights. ``None`` uses
                random initialization.
        """

        super().__init__()
        self.image_encoder = OrthogonalSliceResNet101Encoder(weights=weights)
        self.risk_head = nn.Linear(self.image_encoder.n_features, 1)

    @property
    def encoder(self):
        """Return the underlying torchvision ResNet-101 module."""

        return self.image_encoder.model

    def train(self, mode: bool = True) -> SingleImageSurvivalModel:
        """Set model mode while retaining frozen BatchNorm statistics.

        Args:
            mode: ``True`` for training mode or ``False`` for evaluation mode.

        Returns:
            This model instance.
        """

        super().train(mode)
        return self

    def _volume_to_resnet_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 3D MRIs to normalized orthogonal-slice ResNet inputs.

        Args:
            x: MRI tensor shaped ``[batch, 1, depth, height, width]``.

        Returns:
            ImageNet-normalized tensor shaped ``[batch, 3, 224, 224]``.
        """

        return self.image_encoder.volume_to_resnet_input(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one Cox log-risk score per MRI.

        Args:
            x: MRI tensor shaped ``[batch, 1, depth, height, width]``.

        Returns:
            One-dimensional tensor of scalar risk scores.
        """

        features = self.image_encoder(x)
        return self.risk_head(features).squeeze(-1)


def get_last_scan(engine: Engine | None = None) -> pl.DataFrame:
    """Load the latest valid MRI and outcome metadata for every patient.

    The query joins the existing patient-level split and ranks valid MRIs by
    descending acquisition date and image identifier, retaining one row per
    patient.

    Args:
        engine: Optional SQLAlchemy engine. If ``None``, uses the project
            database engine.

    Returns:
        Polars frame containing selected images, diagnoses, outcomes,
        censoring status, and split assignments.
    """

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
    """Check whether a last-scan frame satisfies the modeling contract.

    Args:
        df: Candidate patient-level frame.

    Returns:
        ``True`` when required columns are complete, patient identifiers are
        unique, and all split labels are recognized; otherwise ``False``.
    """

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
    """Add modeling columns and normalize validation split naming.

    Args:
        df: Valid patient-level last-scan frame.

    Returns:
        Frame with ``event_observed`` and ``is_ad`` indicators and lowercase
        split names, with ``"val"`` normalized to ``"validation"``.
    """

    return df.with_columns(
        (~pl.col("is_censored")).cast(pl.Boolean).alias("event_observed"),
        pl.col("final_diagnosis").eq("AD").alias("is_ad"),
        pl.when(pl.col("train_test_split").cast(pl.String).str.to_lowercase() == "val")
        .then(pl.lit("validation"))
        .otherwise(pl.col("train_test_split").cast(pl.String).str.to_lowercase())
        .alias("train_test_split"),
    )


def build_patient_records(df: pl.DataFrame, tensor_dir: Path) -> list[PatientRecord]:
    """Build and validate the immutable patient-level modeling cohort.

    Args:
        df: One-row-per-patient last-scan frame.
        tensor_dir: Directory containing patient tensor packages.

    Returns:
        Validated patient records across train, validation, and test splits.

    Raises:
        FileNotFoundError: If any expected patient tensor package is missing.
        ValueError: If the frame contract is invalid, survival times are
            negative, a split is empty, or a split has no observed events.
    """

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
    """Lazily load and preprocess one selected MRI per patient."""

    def __init__(
        self,
        records: Sequence[PatientRecord],
        spatial_size: tuple[int, int, int] | None = (96, 112, 96),
    ):
        """Initialize the single-image dataset.

        Args:
            records: Patient records defining selected images and outcomes.
            spatial_size: Optional ``(depth, height, width)`` target for
                trilinear resizing.
        """

        self.records = list(records)
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        """Return the number of patient observations."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and standardize one patient's selected MRI.

        The method memory-maps a weights-only tensor package, locates the
        selected image, optionally resizes it, and standardizes the complete
        volume to zero mean and unit variance.

        Args:
            index: Positional patient-record index.

        Returns:
            Mapping containing the image tensor, outcome, dataset index,
            patient identifier, and image identifier.

        Raises:
            ValueError: If the tensor package belongs to another patient,
                omits the selected image, or contains an MRI with an invalid
                shape.
        """

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
    """Persist legacy baseline run artifacts in PostgreSQL."""

    def __init__(self, engine: Engine):
        """Initialize the legacy result store.

        Args:
            engine: SQLAlchemy engine connected to the project PostgreSQL
                database.
        """

        self.engine = engine

    def ensure_schema(self) -> None:
        """Create legacy baseline tables when they do not already exist.

        The operation creates ``_ml.baseline_runs``,
        ``_ml.baseline_run_patients``, ``_ml.baseline_epoch_metrics``, and
        ``_ml.baseline_test_predictions`` without modifying existing rows.
        """

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
                initial_weights_kind TEXT NOT NULL DEFAULT 'resnet101_imagenet1k_v2'
                    CHECK (
                        initial_weights_kind IN (
                            'resnet101_imagenet1k_v2',
                            'baseline_checkpoint',
                            'resumed_baseline_checkpoint'
                        )
                    ),
                initial_weights_source_run_id TEXT,
                initial_weights_checkpoint_path TEXT,
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
            ALTER TABLE _ml.baseline_runs
            ADD COLUMN IF NOT EXISTS initial_weights_kind TEXT
                NOT NULL DEFAULT 'resnet101_imagenet1k_v2'
            """,
            """
            ALTER TABLE _ml.baseline_runs
            ADD COLUMN IF NOT EXISTS initial_weights_source_run_id TEXT
            """,
            """
            ALTER TABLE _ml.baseline_runs
            ADD COLUMN IF NOT EXISTS initial_weights_checkpoint_path TEXT
            """,
            """
            UPDATE _ml.baseline_runs
            SET initial_weights_kind = 'resumed_baseline_checkpoint',
                initial_weights_checkpoint_path = COALESCE(
                    initial_weights_checkpoint_path,
                    config ->> 'resume_from'
                )
            WHERE config ->> 'resume_from' IS NOT NULL
              AND initial_weights_kind = 'resnet101_imagenet1k_v2'
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
        initial_weights_source_run_id: str | None = None,
    ) -> None:
        """Create a running experiment and persist its patient cohort.

        Args:
            run_id: Unique identifier for the baseline run.
            config: Training configuration serialized as JSON.
            device: PyTorch device used for the run.
            checkpoint_dir: Directory assigned to run checkpoints.
            records: Exact patient records used across all splits.
            initial_weights_source_run_id: Run identifier stored in the
                checkpoint used to initialize or resume this run.
        """

        counts = {split: sum(record.split == split for record in records) for split in SPLIT_NAMES}
        event_counts = {
            split: sum(record.split == split and record.event_observed for record in records)
            for split in SPLIT_NAMES
        }
        config_json = json.dumps(asdict(config), default=str)
        if config.resume_from is not None:
            initial_weights_kind = "resumed_baseline_checkpoint"
            initial_weights_checkpoint_path = str(config.resume_from.resolve())
        elif config.initial_weights_from is not None:
            initial_weights_kind = "baseline_checkpoint"
            initial_weights_checkpoint_path = str(config.initial_weights_from.resolve())
        else:
            initial_weights_kind = "resnet101_imagenet1k_v2"
            initial_weights_checkpoint_path = None
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.baseline_runs (
                        run_id, status, config, device, checkpoint_dir,
                        initial_weights_kind, initial_weights_source_run_id,
                        initial_weights_checkpoint_path,
                        train_patients, validation_patients, test_patients,
                        train_events, validation_events, test_events
                    ) VALUES (
                        :run_id, 'running', CAST(:config AS JSONB), :device, :checkpoint_dir,
                        :initial_weights_kind, :initial_weights_source_run_id,
                        :initial_weights_checkpoint_path,
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
                    "initial_weights_kind": initial_weights_kind,
                    "initial_weights_source_run_id": initial_weights_source_run_id,
                    "initial_weights_checkpoint_path": initial_weights_checkpoint_path,
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
        """Persist training and validation measurements for one epoch.

        Args:
            run_id: Identifier of the model run.
            epoch: One-based epoch number.
            train_loss: Full-cohort training Cox loss.
            train_c_index: Training Harrell C-index.
            validation_loss: Full-cohort validation Cox loss.
            validation_c_index: Validation Harrell C-index.
            learning_rate: Optimizer learning rate.
            checkpoint_path: Checkpoint saved for the epoch.
        """

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
        """Persist patient-level test risks and outcomes.

        Args:
            run_id: Identifier of the evaluated run.
            predictions: Aligned test predictions and patient metadata.
        """

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
        """Mark a run completed and store its selected results.

        Args:
            run_id: Identifier of the completed run.
            best_epoch: Epoch selected by minimum validation loss.
            best_validation_loss: Lowest observed validation Cox loss.
            test_c_index: Test C-index produced by the selected checkpoint.
        """

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
        """Mark a run failed and persist a bounded error description.

        Args:
            run_id: Identifier of the failed run.
            error_message: Failure description. Only the first 10,000
                characters are stored.
        """

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
    """Resolve a requested training device.

    Automatic selection prefers CUDA, then Apple MPS, and finally CPU.

    Args:
        requested_device: PyTorch device string or ``"auto"``.

    Returns:
        Resolved PyTorch device.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """

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
    """Seed random generators and request deterministic cuDNN behavior.

    Args:
        seed: Seed applied to Python, PyTorch, and every available CUDA device.
    """

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
    """Build a deterministic single-image data loader.

    Record order is preserved so dataset indices can align full-risk-set score
    gradients with the second mini-batch training pass.

    Args:
        records: Patient records for one data split.
        config: Batch, worker, and preprocessing configuration.
        device: Device used to determine whether pinned memory is beneficial.

    Returns:
        Deterministically ordered single-image data loader.
    """

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
    """Collect scalar Cox risks without retaining gradients.

    The function preserves the model's current training or evaluation mode and
    returns prediction tensors on the CPU.

    Args:
        model: Model producing one risk score per MRI.
        loader: Single-image data loader.
        device: Device receiving image batches.

    Returns:
        Concatenated risks, outcomes, dataset indices, and identifiers.

    Raises:
        ValueError: If the loader yields no observations.
    """

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

    Args:
        model: Cox risk model to optimize.
        loader: Deterministically ordered training loader.
        optimizer: Optimizer updated after score-gradient recomputation.
        device: Device receiving each image batch.
        gradient_clip_norm: Maximum parameter-gradient norm. Values less than
            or equal to zero disable clipping.

    Returns:
        A tuple containing full-risk-set Cox loss and training C-index.

    Raises:
        ValueError: If prediction collection or Cox loss validation fails.
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
    """Evaluate a Cox model on one complete data split.

    Args:
        model: Cox risk model to evaluate.
        loader: Single-image data loader for one split.
        device: Device receiving each image batch.

    Returns:
        A tuple containing Cox loss, Harrell C-index, and aligned predictions.

    Raises:
        ValueError: If the loader is empty or Cox inputs are invalid.
    """

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
    """Atomically save model, optimizer, configuration, and epoch metrics.

    The checkpoint is written to a sibling temporary file before replacement
    into the final path.

    Args:
        checkpoint_path: Final checkpoint destination.
        run_id: Identifier of the experiment run.
        epoch: Epoch represented by the checkpoint.
        model: Baseline model whose state is saved.
        optimizer: Optimizer whose state is saved.
        config: Training configuration serialized into the checkpoint.
        metrics: Training and validation metrics for the epoch.
    """

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


def _load_baseline_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    required_keys: set[str],
) -> dict[str, Any]:
    """Load a checkpoint dictionary and validate its required top-level keys."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")

    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required fields ({missing}): {checkpoint_path}")
    return checkpoint


def _checkpoint_source_run_id(checkpoint: dict[str, Any], checkpoint_path: Path) -> str:
    """Return and validate the run identifier embedded in a checkpoint."""
    source_run_id = checkpoint["run_id"]
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError(f"Checkpoint run_id must be a non-empty string: {checkpoint_path}")
    return source_run_id


def _restore_checkpoint_state(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    weight_decay: float,
) -> tuple[int, dict[str, float], str]:
    """Apply a loaded checkpoint's model and optimizer training state."""

    epoch = checkpoint["epoch"]
    metrics = checkpoint["metrics"]
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"Checkpoint epoch must be a positive integer: {checkpoint_path}")
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint metrics must be a dictionary: {checkpoint_path}")
    if "validation_loss" not in metrics:
        raise ValueError(f"Checkpoint metrics are missing validation_loss: {checkpoint_path}")
    source_run_id = _checkpoint_source_run_id(checkpoint, checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
        parameter_group["weight_decay"] = weight_decay

    return epoch, metrics, source_run_id


def restore_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> tuple[int, dict[str, float]]:
    """Restore training state from a baseline checkpoint.

    Optimizer moments are restored while the current run's learning rate and
    weight decay are applied to every optimizer parameter group.

    Args:
        checkpoint_path: Existing baseline checkpoint to restore.
        model: Model receiving the saved parameter state.
        optimizer: Optimizer receiving the saved training state.
        device: Device onto which tensors in the checkpoint are mapped.
        learning_rate: Learning rate for the continued training run.
        weight_decay: Weight decay for the continued training run.

    Returns:
        The completed checkpoint epoch and its saved metrics.

    Raises:
        ValueError: If the file is not a compatible baseline checkpoint.
        RuntimeError: If saved model or optimizer state is incompatible.
    """

    checkpoint = _load_baseline_checkpoint(
        checkpoint_path,
        device=device,
        required_keys={"run_id", "epoch", "model_state_dict", "optimizer_state_dict", "metrics"},
    )
    epoch, metrics, _ = _restore_checkpoint_state(
        checkpoint,
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return epoch, metrics


def load_initial_weights(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    device: torch.device,
) -> str:
    """Initialize a model from another baseline run without resuming it.

    Only ``model_state_dict`` is loaded. Optimizer state, source epoch, source
    metrics, and source configuration are intentionally ignored so the caller
    starts an independent run at epoch one.

    Args:
        checkpoint_path: Existing baseline checkpoint containing model
            parameters.
        model: Model receiving the saved parameter state.
        device: Device onto which tensors in the checkpoint are mapped.

    Returns:
        Identifier of the run that produced the initial weights.

    Raises:
        ValueError: If the file is not a compatible baseline checkpoint.
        RuntimeError: If the saved model state is incompatible.
    """

    checkpoint = _load_baseline_checkpoint(
        checkpoint_path,
        device=device,
        required_keys={"run_id", "model_state_dict"},
    )
    source_run_id = _checkpoint_source_run_id(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return source_run_id


def run_training(config: TrainingConfig, engine: Engine | None = None) -> str:
    """Train, validate, test, checkpoint, and persist one legacy baseline run.

    The function queries one selected MRI per patient, validates tensor and
    split availability, and trains with the exact full-risk-set Cox objective.
    Every epoch is checkpointed and written to the legacy baseline result
    tables. When ``config.resume_from`` is set, model and optimizer state are
    restored and training begins at the following epoch. When
    ``config.initial_weights_from`` is set, only model weights are loaded into
    an otherwise new run that begins at epoch one. The checkpoint with minimum
    validation Cox loss is restored for final test evaluation.

    Args:
        config: Baseline training, loading, and checkpoint configuration.
        engine: Optional SQLAlchemy engine. If ``None``, uses the project
            database engine.

    Returns:
        Unique identifier of the completed baseline run.

    Raises:
        FileNotFoundError: If an expected patient tensor package is missing.
        ValueError: If configuration, cohort, loader, or Cox-loss validation
            fails.
        RuntimeError: If training does not produce a selectable checkpoint.

    Note:
        Exceptions raised after run creation attempt to mark the persisted run
        as failed. If that persistence update also fails, its error is attached
        as a note to the original exception.
    """

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
    run_started = False

    try:
        uses_checkpoint_weights = config.resume_from is not None or config.initial_weights_from is not None
        initial_weights = None if uses_checkpoint_weights else ResNet101_Weights.IMAGENET1K_V2
        model = SingleImageSurvivalModel(weights=initial_weights).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_epoch = 0
        best_validation_loss = math.inf
        best_checkpoint: Path | None = None
        first_epoch = 1
        initial_weights_source_run_id: str | None = None

        if config.resume_from is not None:
            resume_checkpoint = _load_baseline_checkpoint(
                config.resume_from,
                device=device,
                required_keys={
                    "run_id",
                    "epoch",
                    "model_state_dict",
                    "optimizer_state_dict",
                    "metrics",
                },
            )
            resumed_epoch, resumed_metrics, initial_weights_source_run_id = _restore_checkpoint_state(
                resume_checkpoint,
                config.resume_from,
                model=model,
                optimizer=optimizer,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            if config.epochs <= resumed_epoch:
                raise ValueError(
                    f"epochs ({config.epochs}) must be greater than the resumed checkpoint epoch ({resumed_epoch})"
                )
            first_epoch = resumed_epoch + 1
            best_epoch = resumed_epoch
            best_validation_loss = float(resumed_metrics["validation_loss"])
            best_checkpoint = config.resume_from
            print(
                f"Resumed model and optimizer from {config.resume_from} at epoch {resumed_epoch}; "
                f"continuing through epoch {config.epochs}"
            )
        elif config.initial_weights_from is not None:
            initial_weights_source_run_id = load_initial_weights(
                config.initial_weights_from,
                model=model,
                device=device,
            )
            print(
                f"Initialized separate run from model weights in {config.initial_weights_from} "
                f"(source run {initial_weights_source_run_id}); starting at epoch 1 with a fresh optimizer"
            )

        result_store.start_run(
            run_id,
            config,
            device,
            checkpoint_dir,
            records,
            initial_weights_source_run_id=initial_weights_source_run_id,
        )
        run_started = True
        print(
            f"Run {run_id} on {device}: "
            + ", ".join(
                f"{split}={len(split_records[split])} "
                f"({sum(r.event_observed for r in split_records[split])} events)"
                for split in SPLIT_NAMES
            )
        )

        for epoch in range(first_epoch, config.epochs + 1):
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
        if run_started:
            try:
                result_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
            except Exception as persistence_exc:
                exc.add_note(f"Additionally failed to mark Postgres run as failed: {persistence_exc}")
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a legacy baseline run.

    Args:
        argv: Optional argument sequence. ``None`` reads arguments from the
            current process.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--tensor-dir", type=Path, default=default_tensor_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    checkpoint_source_group = parser.add_mutually_exclusive_group()
    checkpoint_source_group.add_argument(
        "--resume-from",
        type=Path,
        help="checkpoint to resume; --epochs is the total target epoch",
    )
    checkpoint_source_group.add_argument(
        "--initial-weights-from",
        type=Path,
        help="model checkpoint for a separate run starting at epoch 1 with a fresh optimizer",
    )
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
    """Run legacy baseline training from command-line arguments.

    Args:
        argv: Optional argument sequence. ``None`` reads arguments from the
            current process.

    Returns:
        Unique identifier of the completed baseline run.
    """

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
        resume_from=args.resume_from,
        initial_weights_from=args.initial_weights_from,
        spatial_size=tuple(args.spatial_size),
    )
    return run_training(config)


if __name__ == "__main__":
    main()
