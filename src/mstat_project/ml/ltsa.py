"""Train LTSA on a fixed-landmark longitudinal MRI cohort."""

import argparse
import logging
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from ltsa import LTSA, LTSAOutputs, nll_loss
from sqlalchemy import Engine
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights

from mstat_project.ml.landmarks import (
    DiscreteTimeGrid,
    LandmarkBatch,
    LandmarkCohortConfig,
    LandmarkPatientRecord,
    LandmarkSequenceDataset,
    build_landmark_records,
    collate_landmark_samples,
    load_landmark_frame,
)
from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.ml.results import EpochMetrics, PredictionRecord, SurvivalResultStore
from mstat_project.ml.utils import concordance_index, default_checkpoint_dir, default_tensor_dir
from mstat_project.utils import get_db_engine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LTSATrainingConfig:
    landmark_months: float
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
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    checkpoint_root: Path = field(default_factory=lambda: default_checkpoint_dir("ltsa"))
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        if self.landmark_months < 0:
            raise ValueError("landmark_months cannot be negative")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip_norm < 0 or self.bin_width_months <= 0:
            raise ValueError("gradient_clip_norm must be non-negative and bin_width_months positive")
        if not 0 <= self.censoring_beta <= 1:
            raise ValueError("censoring_beta must be between 0 and 1")
        if self.auxiliary_loss_weight < 0 or not 0 <= self.dropout < 1:
            raise ValueError("auxiliary_loss_weight must be non-negative and dropout in [0, 1)")
        if self.n_heads < 1 or self.n_layers < 1 or self.num_workers < 0:
            raise ValueError("n_heads and n_layers must be positive; num_workers cannot be negative")
        if 2_048 % self.n_heads != 0:
            raise ValueError("n_heads must evenly divide the ResNet-101 feature width (2048)")


@dataclass(frozen=True)
class LTSALossBundle:
    total: Tensor
    survival: Tensor
    auxiliary: Tensor
    risk: Tensor
    survival_curves: Tensor


@dataclass(frozen=True)
class LTSAPredictionBundle:
    risks: Tensor
    times: Tensor
    events: Tensor
    survival_curves: Tensor
    ptids: list[str]
    image_ids: list[tuple[str, ...]]


@dataclass(frozen=True)
class LTSARunResult:
    run_id: str
    comparison_id: str
    best_epoch: int
    validation_c_index: float
    test_c_index: float
    predictions: LTSAPredictionBundle


def resolve_device(requested_device: str) -> torch.device:
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_landmark_loaders(
    records: list[LandmarkPatientRecord],
    *,
    config: LTSATrainingConfig,
    device: torch.device,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record.split == split]
        generator = torch.Generator().manual_seed(config.seed)
        loaders[split] = DataLoader(
            LandmarkSequenceDataset(split_records, spatial_size=config.spatial_size),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            collate_fn=collate_landmark_samples,
            generator=generator,
        )
    return loaders


def _last_visit(values: Tensor, sequence_lengths: Tensor) -> Tensor:
    return values[torch.arange(values.shape[0], device=values.device), sequence_lengths - 1]


def compute_ltsa_loss(
    outputs: LTSAOutputs,
    batch: LandmarkBatch,
    grid: DiscreteTimeGrid,
    *,
    censoring_beta: float,
    auxiliary_loss_weight: float,
) -> LTSALossBundle:
    last_hazards = _last_visit(outputs.hazards, batch.sequence_lengths)
    last_survival = _last_visit(outputs.surv, batch.sequence_lengths)
    survival_loss = nll_loss(
        last_hazards,
        last_survival,
        grid.encode(batch.observed_times),
        ~batch.events,
        beta=censoring_beta,
    )
    if outputs.next_visit_mask.any():
        auxiliary_loss = F.mse_loss(
            outputs.feat_preds[outputs.next_visit_mask],
            outputs.feat_targets[outputs.next_visit_mask],
        )
    else:
        auxiliary_loss = outputs.feat_preds.sum() * 0.0
    return LTSALossBundle(
        total=survival_loss + auxiliary_loss_weight * auxiliary_loss,
        survival=survival_loss,
        auxiliary=auxiliary_loss,
        risk=grid.risk_from_survival(last_survival),
        survival_curves=last_survival,
    )


def _run_epoch(
    model: LTSA,
    loader: DataLoader,
    grid: DiscreteTimeGrid,
    *,
    config: LTSATrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    survival_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    observation_count = 0
    risks: list[Tensor] = []
    times: list[Tensor] = []
    events: list[Tensor] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch.images,
                sequence_lengths=batch.sequence_lengths,
                relative_times=batch.relative_times,
            )
            losses = compute_ltsa_loss(
                outputs,
                batch,
                grid,
                censoring_beta=config.censoring_beta,
                auxiliary_loss_weight=config.auxiliary_loss_weight,
            )
            if optimizer is not None:
                losses.total.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()

            batch_size = batch.observed_times.numel()
            total_loss_sum += float(losses.total.item()) * batch_size
            survival_loss_sum += float(losses.survival.item()) * batch_size
            auxiliary_loss_sum += float(losses.auxiliary.item()) * batch_size
            observation_count += batch_size
            risks.append(losses.risk.detach().cpu())
            times.append(batch.observed_times.detach().cpu())
            events.append(batch.events.detach().cpu())

    if observation_count == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return (
        total_loss_sum / observation_count,
        survival_loss_sum / observation_count,
        auxiliary_loss_sum / observation_count,
        concordance_index(torch.cat(risks), torch.cat(times), torch.cat(events)),
    )


def collect_ltsa_predictions(
    model: LTSA,
    loader: DataLoader,
    grid: DiscreteTimeGrid,
    *,
    device: torch.device,
) -> LTSAPredictionBundle:
    model.eval()
    risks: list[Tensor] = []
    times: list[Tensor] = []
    events: list[Tensor] = []
    survival_curves: list[Tensor] = []
    ptids: list[str] = []
    image_ids: list[tuple[str, ...]] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            outputs = model(
                batch.images,
                sequence_lengths=batch.sequence_lengths,
                relative_times=batch.relative_times,
            )
            last_survival = _last_visit(outputs.surv, batch.sequence_lengths)
            survival_curves.append(last_survival.cpu())
            risks.append(grid.risk_from_survival(last_survival).cpu())
            times.append(batch.observed_times.cpu())
            events.append(batch.events.cpu())
            ptids.extend(batch.ptids)
            image_ids.extend(batch.image_ids)
    if not risks:
        raise ValueError("Cannot collect predictions from an empty data loader")
    return LTSAPredictionBundle(
        risks=torch.cat(risks),
        times=torch.cat(times),
        events=torch.cat(events),
        survival_curves=torch.cat(survival_curves),
        ptids=ptids,
        image_ids=image_ids,
    )


def save_ltsa_checkpoint(
    checkpoint_path: Path,
    *,
    run_id: str,
    epoch: int,
    model: LTSA,
    optimizer: torch.optim.Optimizer,
    config: LTSATrainingConfig,
    grid: DiscreteTimeGrid,
    metrics: EpochMetrics,
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
            "time_grid": asdict(grid),
            "metrics": asdict(metrics),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def run_training(
    config: LTSATrainingConfig,
    *,
    engine: Engine | None = None,
    comparison_id: str | None = None,
    cohort_records: list[LandmarkPatientRecord] | None = None,
) -> LTSARunResult:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    database_engine = engine or get_db_engine()
    records = cohort_records
    if records is None:
        records = build_landmark_records(
            load_landmark_frame(database_engine, landmark_months=config.landmark_months),
            LandmarkCohortConfig(
                landmark_months=config.landmark_months,
                tensor_dir=config.tensor_dir,
                spatial_size=config.spatial_size,
            ),
        )
    elif any(record.landmark_months != config.landmark_months for record in records):
        raise ValueError("Preflighted cohort landmark does not match the LTSA configuration")
    train_records = [record for record in records if record.split == "train"]
    grid = DiscreteTimeGrid.fit(train_records, bin_width_months=config.bin_width_months)
    loaders = make_landmark_loaders(records, config=config, device=device)
    max_sequence_length = max(len(record.image_ids) for record in records)

    run_id = str(uuid.uuid4())
    paired_id = comparison_id or str(uuid.uuid4())
    checkpoint_dir = config.checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    result_store = SurvivalResultStore(database_engine)
    result_store.ensure_schema()
    result_store.start_run(
        run_id,
        comparison_id=paired_id,
        model_type="ltsa",
        landmark_months=config.landmark_months,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        records=records,
    )

    try:
        encoder = OrthogonalSliceResNet101Encoder(
            weights=ResNet101_Weights.IMAGENET1K_V2 if config.pretrained else None
        )
        model = LTSA(
            encoder,
            n_heads=config.n_heads,
            dropout=config.dropout,
            n_layers=config.n_layers,
            max_sequence_length=max_sequence_length,
            max_time_index=max(config.landmark_months, 1.0),
            n_time_bins=grid.n_time_bins,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_epoch = 0
        best_validation_loss = math.inf
        best_validation_c_index = -math.inf
        best_checkpoint: Path | None = None
        epochs_without_improvement = 0

        for epoch in range(1, config.epochs + 1):
            train_total, train_survival, train_auxiliary, train_c_index = _run_epoch(
                model,
                loaders["train"],
                grid,
                config=config,
                device=device,
                optimizer=optimizer,
            )
            validation_total, validation_survival, validation_auxiliary, validation_c_index = _run_epoch(
                model,
                loaders["validation"],
                grid,
                config=config,
                device=device,
                optimizer=None,
            )
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_total_loss=train_total,
                train_survival_loss=train_survival,
                train_auxiliary_loss=train_auxiliary,
                train_c_index=train_c_index,
                validation_total_loss=validation_total,
                validation_survival_loss=validation_survival,
                validation_auxiliary_loss=validation_auxiliary,
                validation_c_index=validation_c_index,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                checkpoint_path=checkpoint_path,
            )
            save_ltsa_checkpoint(
                checkpoint_path,
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                grid=grid,
                metrics=epoch_metrics,
            )
            result_store.record_epoch(run_id, epoch_metrics)
            improved = validation_c_index > best_validation_c_index or (
                validation_c_index == best_validation_c_index and validation_total < best_validation_loss
            )
            if improved:
                best_epoch = epoch
                best_validation_loss = validation_total
                best_validation_c_index = validation_c_index
                best_checkpoint = checkpoint_path
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            LOGGER.info(
                "LTSA epoch %d: train loss %.4f C-index %.4f; validation loss %.4f C-index %.4f",
                epoch,
                train_total,
                train_c_index,
                validation_total,
                validation_c_index,
            )
            if epochs_without_improvement >= config.patience:
                break

        if best_checkpoint is None:
            raise RuntimeError("LTSA training did not produce a best checkpoint")
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions = collect_ltsa_predictions(model, loaders["test"], grid, device=device)
        test_c_index = concordance_index(predictions.risks, predictions.times, predictions.events)
        result_store.record_predictions(
            run_id,
            [
                PredictionRecord(
                    ptid=ptid,
                    image_ids=image_history,
                    observed_time_months=float(time_value),
                    event_observed=bool(event_value),
                    risk_score=float(risk_value),
                    survival_curve=[float(value) for value in survival_curve],
                )
                for ptid, image_history, time_value, event_value, risk_value, survival_curve in zip(
                    predictions.ptids,
                    predictions.image_ids,
                    predictions.times.tolist(),
                    predictions.events.tolist(),
                    predictions.risks.tolist(),
                    predictions.survival_curves.tolist(),
                    strict=True,
                )
            ],
        )
        result_store.complete_run(
            run_id,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            best_validation_c_index=best_validation_c_index,
            test_c_index=test_c_index,
        )
        return LTSARunResult(
            run_id=run_id,
            comparison_id=paired_id,
            best_epoch=best_epoch,
            validation_c_index=best_validation_c_index,
            test_c_index=test_c_index,
            predictions=predictions,
        )
    except Exception as exc:
        result_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark-months", type=float, required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--tensor-dir", type=Path, default=default_tensor_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir("ltsa"))
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(96, 112, 96))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    result = run_training(
        LTSATrainingConfig(
            landmark_months=args.landmark_months,
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
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            pretrained=not args.no_pretrained,
            tensor_dir=args.tensor_dir,
            checkpoint_root=args.checkpoint_dir,
            spatial_size=tuple(args.spatial_size),
        )
    )
    LOGGER.info("Completed LTSA run %s with test C-index %.4f", result.run_id, result.test_c_index)
    return result.run_id


if __name__ == "__main__":
    main()
