"""Unified Postgres persistence for baseline and LTSA experiments."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from sqlalchemy import Engine, text

from mstat_project.ml.landmarks import LandmarkPatientRecord

type ModelType = Literal["baseline", "ltsa"]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_survival_loss: float
    train_auxiliary_loss: float | None
    train_c_index: float
    validation_total_loss: float
    validation_survival_loss: float
    validation_auxiliary_loss: float | None
    validation_c_index: float
    learning_rate: float
    checkpoint_path: Path


@dataclass(frozen=True)
class PredictionRecord:
    ptid: str
    image_ids: tuple[str, ...]
    observed_time_months: float
    event_observed: bool
    risk_score: float
    survival_curve: list[float] | None


class SurvivalResultStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        statements = [
            "CREATE SCHEMA IF NOT EXISTS _ml",
            """
            CREATE TABLE IF NOT EXISTS _ml.survival_runs (
                run_id UUID PRIMARY KEY,
                comparison_id UUID NOT NULL,
                model_type TEXT NOT NULL CHECK (model_type IN ('baseline', 'ltsa')),
                landmark_months DOUBLE PRECISION NOT NULL,
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
                best_validation_c_index DOUBLE PRECISION,
                test_c_index DOUBLE PRECISION,
                paired_c_index_difference DOUBLE PRECISION,
                paired_ci_low DOUBLE PRECISION,
                paired_ci_high DOUBLE PRECISION,
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.survival_epoch_metrics (
                run_id UUID NOT NULL REFERENCES _ml.survival_runs(run_id) ON DELETE CASCADE,
                epoch INTEGER NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                train_total_loss DOUBLE PRECISION NOT NULL,
                train_survival_loss DOUBLE PRECISION NOT NULL,
                train_auxiliary_loss DOUBLE PRECISION,
                train_c_index DOUBLE PRECISION,
                validation_total_loss DOUBLE PRECISION NOT NULL,
                validation_survival_loss DOUBLE PRECISION NOT NULL,
                validation_auxiliary_loss DOUBLE PRECISION,
                validation_c_index DOUBLE PRECISION,
                learning_rate DOUBLE PRECISION NOT NULL,
                checkpoint_path TEXT NOT NULL,
                PRIMARY KEY (run_id, epoch)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.survival_run_patients (
                run_id UUID NOT NULL REFERENCES _ml.survival_runs(run_id) ON DELETE CASCADE,
                ptid TEXT NOT NULL,
                split TEXT NOT NULL,
                landmark_months DOUBLE PRECISION NOT NULL,
                observed_time_months DOUBLE PRECISION NOT NULL,
                event_observed BOOLEAN NOT NULL,
                selected_image_id TEXT NOT NULL,
                image_ids JSONB NOT NULL,
                PRIMARY KEY (run_id, ptid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS _ml.survival_test_predictions (
                run_id UUID NOT NULL REFERENCES _ml.survival_runs(run_id) ON DELETE CASCADE,
                ptid TEXT NOT NULL,
                image_ids JSONB NOT NULL,
                observed_time_months DOUBLE PRECISION NOT NULL,
                event_observed BOOLEAN NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                survival_curve JSONB,
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
        *,
        comparison_id: str,
        model_type: ModelType,
        landmark_months: float,
        config: Any,
        device: torch.device,
        checkpoint_dir: Path,
        records: list[LandmarkPatientRecord],
    ) -> None:
        counts = {
            split: sum(record.split == split for record in records) for split in ("train", "validation", "test")
        }
        event_counts = {
            split: sum(record.split == split and record.event_observed for record in records)
            for split in ("train", "validation", "test")
        }
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        config_data = asdict(config) if hasattr(config, "__dataclass_fields__") else config
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.survival_runs (
                        run_id, comparison_id, model_type, landmark_months, status, config, device,
                        checkpoint_dir, train_patients, validation_patients, test_patients,
                        train_events, validation_events, test_events
                    ) VALUES (
                        :run_id, :comparison_id, :model_type, :landmark_months, 'running',
                        CAST(:config AS JSONB), :device, :checkpoint_dir, :train_patients,
                        :validation_patients, :test_patients, :train_events, :validation_events,
                        :test_events
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "comparison_id": comparison_id,
                    "model_type": model_type,
                    "landmark_months": landmark_months,
                    "config": json.dumps(config_data, default=str),
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
                    INSERT INTO _ml.survival_run_patients (
                        run_id, ptid, split, landmark_months, observed_time_months,
                        event_observed, selected_image_id, image_ids
                    ) VALUES (
                        :run_id, :ptid, :split, :landmark_months, :observed_time_months,
                        :event_observed, :selected_image_id, CAST(:image_ids AS JSONB)
                    )
                    """
                ),
                [
                    {
                        "run_id": run_id,
                        "ptid": record.ptid,
                        "split": record.split,
                        "landmark_months": record.landmark_months,
                        "observed_time_months": record.observed_time_months,
                        "event_observed": record.event_observed,
                        "selected_image_id": record.selected_image_id,
                        "image_ids": json.dumps(record.image_ids),
                    }
                    for record in records
                ],
            )

    def record_epoch(self, run_id: str, metrics: EpochMetrics) -> None:
        if not metrics.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {metrics.checkpoint_path}")
        values = asdict(metrics)
        values["run_id"] = run_id
        values["checkpoint_path"] = str(metrics.checkpoint_path.resolve())
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.survival_epoch_metrics (
                        run_id, epoch, train_total_loss, train_survival_loss, train_auxiliary_loss,
                        train_c_index, validation_total_loss, validation_survival_loss,
                        validation_auxiliary_loss, validation_c_index, learning_rate, checkpoint_path
                    ) VALUES (
                        :run_id, :epoch, :train_total_loss, :train_survival_loss,
                        :train_auxiliary_loss, :train_c_index, :validation_total_loss,
                        :validation_survival_loss, :validation_auxiliary_loss,
                        :validation_c_index, :learning_rate, :checkpoint_path
                    )
                    """
                ),
                values,
            )

    def record_predictions(self, run_id: str, predictions: list[PredictionRecord]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO _ml.survival_test_predictions (
                        run_id, ptid, image_ids, observed_time_months, event_observed,
                        risk_score, survival_curve
                    ) VALUES (
                        :run_id, :ptid, CAST(:image_ids AS JSONB), :observed_time_months,
                        :event_observed, :risk_score, CAST(:survival_curve AS JSONB)
                    )
                    """
                ),
                [
                    {
                        "run_id": run_id,
                        "ptid": prediction.ptid,
                        "image_ids": json.dumps(prediction.image_ids),
                        "observed_time_months": prediction.observed_time_months,
                        "event_observed": prediction.event_observed,
                        "risk_score": prediction.risk_score,
                        "survival_curve": json.dumps(prediction.survival_curve),
                    }
                    for prediction in predictions
                ],
            )

    def complete_run(
        self,
        run_id: str,
        *,
        best_epoch: int,
        best_validation_loss: float,
        best_validation_c_index: float,
        test_c_index: float,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE _ml.survival_runs
                    SET status = 'completed', completed_at = NOW(), best_epoch = :best_epoch,
                        best_validation_loss = :best_validation_loss,
                        best_validation_c_index = :best_validation_c_index,
                        test_c_index = :test_c_index
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "best_validation_c_index": best_validation_c_index,
                    "test_c_index": test_c_index,
                },
            )

    def record_comparison(
        self,
        comparison_id: str,
        *,
        landmark_months: float,
        difference: float,
        confidence_interval_low: float,
        confidence_interval_high: float,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE _ml.survival_runs
                    SET paired_c_index_difference = :difference,
                        paired_ci_low = :confidence_interval_low,
                        paired_ci_high = :confidence_interval_high
                    WHERE comparison_id = :comparison_id AND landmark_months = :landmark_months
                    """
                ),
                {
                    "comparison_id": comparison_id,
                    "landmark_months": landmark_months,
                    "difference": difference,
                    "confidence_interval_low": confidence_interval_low,
                    "confidence_interval_high": confidence_interval_high,
                },
            )

    def fail_run(self, run_id: str, error_message: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE _ml.survival_runs
                    SET status = 'failed', completed_at = NOW(), error_message = :error_message
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "error_message": error_message[:10_000]},
            )
