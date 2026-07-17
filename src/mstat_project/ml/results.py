"""Persist baseline and LTSA survival experiments in a unified PostgreSQL schema.

The module stores run metadata, per-epoch metrics, exact landmark-cohort
membership, patient-level test predictions, and paired C-index comparisons in
the ``_ml`` schema. Each public write method executes within its own database
transaction.
"""

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
    """Training and validation measurements persisted for one epoch.

    Attributes:
        epoch: One-based training epoch number.
        train_total_loss: Total objective averaged over the training cohort.
        train_survival_loss: Survival component of the training objective.
        train_auxiliary_loss: Auxiliary training loss, or ``None`` for models
            without an auxiliary objective.
        train_c_index: Training Harrell C-index.
        validation_total_loss: Total objective averaged over the validation
            cohort.
        validation_survival_loss: Survival component of the validation
            objective.
        validation_auxiliary_loss: Auxiliary validation loss, or ``None`` for
            models without an auxiliary objective.
        validation_c_index: Validation Harrell C-index.
        learning_rate: Optimizer learning rate recorded after the epoch.
        checkpoint_path: Filesystem path to the checkpoint saved for the epoch.
    """

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
    """Patient-level prediction persisted for a completed test evaluation.

    Attributes:
        ptid: Patient identifier.
        image_ids: Chronological image history supplied by the landmark cohort.
        observed_time_months: Event or censoring time measured from the
            landmark.
        event_observed: Whether the modeled event was observed.
        risk_score: Scalar model risk used for concordance evaluation.
        survival_curve: Discrete LTSA survival probabilities, or ``None`` for
            models that produce only a scalar risk.
    """

    ptid: str
    image_ids: tuple[str, ...]
    observed_time_months: float
    event_observed: bool
    risk_score: float
    survival_curve: list[float] | None


class SurvivalResultStore:
    """Store survival experiment artifacts using a SQLAlchemy PostgreSQL engine."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the result store.

        Args:
            engine: SQLAlchemy engine connected to the project PostgreSQL
                database.
        """

        self.engine = engine

    def ensure_schema(self) -> None:
        """Create the unified survival schema and tables when they do not exist.

        The operation creates ``_ml.survival_runs``,
        ``_ml.survival_epoch_metrics``, ``_ml.survival_run_patients``, and
        ``_ml.survival_test_predictions``. Existing tables and rows are left
        unchanged.
        """

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
        """Create a running experiment and persist its exact landmark cohort.

        Patient and observed-event counts are derived from ``records`` for each
        train, validation, and test split. The run row and all cohort-membership
        rows are inserted in one transaction.

        Args:
            run_id: Unique identifier for the model run.
            comparison_id: Identifier shared by matched baseline and LTSA runs.
            model_type: Model family, either ``"baseline"`` or ``"ltsa"``.
            landmark_months: Fixed prediction landmark measured from baseline.
            config: Training configuration. Dataclass instances are converted
                to dictionaries before JSON serialization.
            device: PyTorch device used for the run.
            checkpoint_dir: Existing directory that will contain run
                checkpoints.
            records: Validated patient records defining the exact landmark
                cohort and image histories.

        Raises:
            FileNotFoundError: If ``checkpoint_dir`` does not exist.
        """

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
        """Persist metrics and the checkpoint location for one epoch.

        Args:
            run_id: Identifier of the run that produced the epoch.
            metrics: Training and validation metrics to insert.

        Raises:
            FileNotFoundError: If the checkpoint referenced by ``metrics`` is
                not a file.
        """

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
        """Persist patient-level predictions for a run's test split.

        Args:
            run_id: Identifier of the evaluated run.
            predictions: Test predictions to insert. Image histories and
                optional survival curves are stored as JSON.
        """

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
        """Mark a run complete and store checkpoint-selection results.

        Args:
            run_id: Identifier of the completed run.
            best_epoch: Epoch selected by validation performance.
            best_validation_loss: Validation loss at the selected epoch.
            best_validation_c_index: Validation C-index at the selected epoch.
            test_c_index: Test C-index produced by the selected checkpoint.
        """

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
        """Store a paired LTSA-minus-baseline C-index comparison.

        The comparison values are written to every run with the matching
        ``comparison_id`` and landmark, normally the paired baseline and LTSA
        rows.

        Args:
            comparison_id: Identifier shared by the paired model runs.
            landmark_months: Landmark whose paired result is being recorded.
            difference: LTSA test C-index minus baseline test C-index.
            confidence_interval_low: Lower bound of the paired bootstrap
                confidence interval.
            confidence_interval_high: Upper bound of the paired bootstrap
                confidence interval.
        """

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
                    UPDATE _ml.survival_runs
                    SET status = 'failed', completed_at = NOW(), error_message = :error_message
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "error_message": error_message[:10_000]},
            )
