"""Run paired fixed-landmark baseline and LTSA experiments."""

import argparse
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from sqlalchemy import Engine

from mstat_project.ml.landmarks import (
    LandmarkCohortConfig,
    LandmarkPatientRecord,
    build_landmark_records,
    load_landmark_frame,
)
from mstat_project.ml.landmark_baseline import (
    BaselineLandmarkTrainingConfig,
    BaselineRunResult,
    run_baseline_landmark_training,
)
from mstat_project.ml.ltsa import LTSARunResult, LTSATrainingConfig, run_training
from mstat_project.ml.metrics import PairedCIndexComparison, paired_c_index_difference
from mstat_project.ml.results import SurvivalResultStore
from mstat_project.ml.utils import default_checkpoint_dir, default_tensor_dir
from mstat_project.utils import get_db_engine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonConfig:
    landmarks_months: tuple[float, ...] = (0.0, 12.0, 24.0, 36.0)
    epochs: int = 50
    patience: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    bin_width_months: float = 6.0
    censoring_beta: float = 0.15
    auxiliary_loss_weight: float = 1.0
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.0
    bootstrap_samples: int = 1_000
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    checkpoint_root: Path = field(default_factory=lambda: default_checkpoint_dir("comparison"))
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        if not self.landmarks_months or any(landmark < 0 for landmark in self.landmarks_months):
            raise ValueError("At least one non-negative landmark is required")
        if len(set(self.landmarks_months)) != len(self.landmarks_months):
            raise ValueError("landmarks_months cannot contain duplicates")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.bin_width_months <= 0 or not 0 <= self.censoring_beta <= 1:
            raise ValueError("Invalid discrete survival configuration")
        if self.auxiliary_loss_weight < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LTSA loss or dropout configuration")
        if self.n_heads < 1 or 2_048 % self.n_heads != 0 or self.n_layers < 1:
            raise ValueError("n_heads must divide 2048 and n_layers must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.spatial_size is not None and any(size < 16 for size in self.spatial_size):
            raise ValueError("all spatial dimensions must be at least 16")


@dataclass(frozen=True)
class LandmarkComparisonResult:
    landmark_months: float
    baseline: BaselineRunResult
    ltsa: LTSARunResult
    comparison: PairedCIndexComparison


def preflight_comparison_cohorts(
    config: ComparisonConfig,
    engine: Engine,
) -> dict[float, list[LandmarkPatientRecord]]:
    """Freeze and validate every landmark cohort before either model starts training."""

    return {
        landmark_months: build_landmark_records(
            load_landmark_frame(engine, landmark_months=landmark_months),
            LandmarkCohortConfig(
                landmark_months=landmark_months,
                tensor_dir=config.tensor_dir,
                spatial_size=config.spatial_size,
            ),
        )
        for landmark_months in config.landmarks_months
    }


def _align_predictions(
    baseline: BaselineRunResult,
    ltsa: LTSARunResult,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline_by_ptid = {
        ptid: (risk, time, event)
        for ptid, risk, time, event in zip(
            baseline.predictions.ptids,
            baseline.predictions.risks,
            baseline.predictions.times,
            baseline.predictions.events,
            strict=True,
        )
    }
    ltsa_by_ptid = {
        ptid: (risk, time, event)
        for ptid, risk, time, event in zip(
            ltsa.predictions.ptids,
            ltsa.predictions.risks,
            ltsa.predictions.times,
            ltsa.predictions.events,
            strict=True,
        )
    }
    if set(baseline_by_ptid) != set(ltsa_by_ptid):
        raise ValueError("Baseline and LTSA test cohorts do not contain the same patients")
    ordered_ptids = sorted(baseline_by_ptid)
    for ptid in ordered_ptids:
        if not torch.equal(baseline_by_ptid[ptid][1], ltsa_by_ptid[ptid][1]) or not torch.equal(
            baseline_by_ptid[ptid][2], ltsa_by_ptid[ptid][2]
        ):
            raise ValueError(f"Baseline and LTSA outcomes differ for patient {ptid}")
    return (
        torch.stack([baseline_by_ptid[ptid][0] for ptid in ordered_ptids]),
        torch.stack([ltsa_by_ptid[ptid][0] for ptid in ordered_ptids]),
        torch.stack([baseline_by_ptid[ptid][1] for ptid in ordered_ptids]),
        torch.stack([baseline_by_ptid[ptid][2] for ptid in ordered_ptids]),
    )


def run_comparison(
    config: ComparisonConfig,
    *,
    engine: Engine | None = None,
) -> list[LandmarkComparisonResult]:
    config.validate()
    database_engine = engine or get_db_engine()
    cohorts = preflight_comparison_cohorts(config, database_engine)
    comparison_id = str(uuid.uuid4())
    results: list[LandmarkComparisonResult] = []
    for landmark_months in config.landmarks_months:
        baseline = run_baseline_landmark_training(
            BaselineLandmarkTrainingConfig(
                landmark_months=landmark_months,
                epochs=config.epochs,
                patience=config.patience,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                seed=config.seed,
                num_workers=config.num_workers,
                device=config.device,
                pretrained=config.pretrained,
                tensor_dir=config.tensor_dir,
                checkpoint_root=config.checkpoint_root / "baseline" / f"landmark_{landmark_months:g}",
                spatial_size=config.spatial_size,
            ),
            engine=database_engine,
            comparison_id=comparison_id,
            cohort_records=cohorts[landmark_months],
        )
        ltsa = run_training(
            LTSATrainingConfig(
                landmark_months=landmark_months,
                epochs=config.epochs,
                patience=config.patience,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                bin_width_months=config.bin_width_months,
                censoring_beta=config.censoring_beta,
                auxiliary_loss_weight=config.auxiliary_loss_weight,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                dropout=config.dropout,
                seed=config.seed,
                num_workers=config.num_workers,
                device=config.device,
                pretrained=config.pretrained,
                tensor_dir=config.tensor_dir,
                checkpoint_root=config.checkpoint_root / "ltsa" / f"landmark_{landmark_months:g}",
                spatial_size=config.spatial_size,
            ),
            engine=database_engine,
            comparison_id=comparison_id,
            cohort_records=cohorts[landmark_months],
        )
        baseline_risk, ltsa_risk, times, events = _align_predictions(baseline, ltsa)
        comparison = paired_c_index_difference(
            baseline_risk,
            ltsa_risk,
            times,
            events,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        )
        SurvivalResultStore(database_engine).record_comparison(
            comparison_id,
            landmark_months=landmark_months,
            difference=comparison.difference,
            confidence_interval_low=comparison.confidence_interval_low,
            confidence_interval_high=comparison.confidence_interval_high,
        )
        results.append(
            LandmarkComparisonResult(
                landmark_months=landmark_months,
                baseline=baseline,
                ltsa=ltsa,
                comparison=comparison,
            )
        )
        LOGGER.info(
            "Landmark %.1f months: baseline %.4f, LTSA %.4f, difference %.4f (95%% CI %.4f, %.4f)",
            landmark_months,
            comparison.baseline_c_index,
            comparison.ltsa_c_index,
            comparison.difference,
            comparison.confidence_interval_low,
            comparison.confidence_interval_high,
        )
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=float, nargs="+", default=(0.0, 12.0, 24.0, 36.0))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--bin-width-months", type=float, default=6.0)
    parser.add_argument("--censoring-beta", type=float, default=0.15)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=1.0)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--tensor-dir", type=Path, default=default_tensor_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir("comparison"))
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(96, 112, 96))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[LandmarkComparisonResult]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    return run_comparison(
        ComparisonConfig(
            landmarks_months=tuple(args.landmarks),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            bin_width_months=args.bin_width_months,
            censoring_beta=args.censoring_beta,
            auxiliary_loss_weight=args.auxiliary_loss_weight,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            pretrained=not args.no_pretrained,
            tensor_dir=args.tensor_dir,
            checkpoint_root=args.checkpoint_dir,
            spatial_size=tuple(args.spatial_size),
        )
    )


if __name__ == "__main__":
    main()
