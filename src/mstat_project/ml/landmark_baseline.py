"""Train the shared single-MRI Cox baseline at a fixed landmark."""

import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from ltsa.losses import cox_ph_loss
from sqlalchemy import Engine
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights

from mstat_project.ml.landmarks import (
    LandmarkBatch,
    LandmarkCohortConfig,
    LandmarkPatientRecord,
    LandmarkSequenceDataset,
    build_landmark_records,
    collate_landmark_samples,
    load_landmark_frame,
)
from mstat_project.ml.ltsa import resolve_device
from mstat_project.ml.models import CoxRiskModel, OrthogonalSliceResNet101Encoder
from mstat_project.ml.results import EpochMetrics, PredictionRecord, SurvivalResultStore
from mstat_project.ml.utils import concordance_index, default_checkpoint_dir, default_tensor_dir
from mstat_project.utils import get_db_engine


@dataclass(frozen=True)
class BaselineLandmarkTrainingConfig:
    landmark_months: float
    epochs: int = 50
    patience: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    checkpoint_root: Path = field(default_factory=lambda: default_checkpoint_dir("baseline_landmark"))
    spatial_size: tuple[int, int, int] | None = (96, 112, 96)

    def validate(self) -> None:
        if self.landmark_months < 0:
            raise ValueError("landmark_months cannot be negative")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm < 0:
            raise ValueError("Invalid optimizer configuration")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True)
class BaselinePredictionBundle:
    risks: Tensor
    times: Tensor
    events: Tensor
    ptids: list[str]
    image_ids: list[tuple[str, ...]]


@dataclass(frozen=True)
class BaselineRunResult:
    run_id: str
    comparison_id: str
    best_epoch: int
    validation_c_index: float
    test_c_index: float
    predictions: BaselinePredictionBundle


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_loaders(
    records: list[LandmarkPatientRecord],
    *,
    config: BaselineLandmarkTrainingConfig,
    device: torch.device,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            LandmarkSequenceDataset(
                [record for record in records if record.split == split],
                spatial_size=config.spatial_size,
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            collate_fn=collate_landmark_samples,
        )
        for split in ("train", "validation", "test")
    }


def _selected_images(batch: LandmarkBatch) -> Tensor:
    return batch.images[
        torch.arange(batch.images.shape[0], device=batch.images.device),
        batch.sequence_lengths - 1,
    ]


def collect_baseline_predictions(
    model: CoxRiskModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> BaselinePredictionBundle:
    risks: list[Tensor] = []
    times: list[Tensor] = []
    events: list[Tensor] = []
    ptids: list[str] = []
    image_ids: list[tuple[str, ...]] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            risks.append(model(_selected_images(batch)).cpu())
            times.append(batch.observed_times.cpu())
            events.append(batch.events.cpu())
            ptids.extend(batch.ptids)
            image_ids.extend(batch.image_ids)
    if not risks:
        raise ValueError("Cannot collect predictions from an empty loader")
    return BaselinePredictionBundle(
        risks=torch.cat(risks),
        times=torch.cat(times),
        events=torch.cat(events),
        ptids=ptids,
        image_ids=image_ids,
    )


def train_baseline_epoch(
    model: CoxRiskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    """Take one exact full-risk-set Cox step using memory-bounded recomputation."""

    model.train()
    predictions = collect_baseline_predictions(model, loader, device=device)
    risk_leaf = predictions.risks.detach().requires_grad_(True)
    loss = cox_ph_loss(risk_leaf, predictions.times, predictions.events)
    risk_gradients = torch.autograd.grad(loss, risk_leaf)[0].detach()

    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        batch_size = batch.observed_times.numel()
        risks = model(_selected_images(batch))
        gradients = risk_gradients[offset : offset + batch_size].to(device)
        (risks * gradients).sum().backward()
        offset += batch_size
    if offset != risk_gradients.numel():
        raise RuntimeError("Baseline recomputation did not consume every score gradient")
    if gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.item()), concordance_index(
        predictions.risks,
        predictions.times,
        predictions.events,
    )


def evaluate_baseline(
    model: CoxRiskModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, float, BaselinePredictionBundle]:
    model.eval()
    predictions = collect_baseline_predictions(model, loader, device=device)
    loss = cox_ph_loss(predictions.risks, predictions.times, predictions.events)
    return (
        float(loss.item()),
        concordance_index(predictions.risks, predictions.times, predictions.events),
        predictions,
    )


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    run_id: str,
    epoch: int,
    model: CoxRiskModel,
    optimizer: torch.optim.Optimizer,
    config: BaselineLandmarkTrainingConfig,
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
            "metrics": asdict(metrics),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def run_baseline_landmark_training(
    config: BaselineLandmarkTrainingConfig,
    *,
    engine: Engine | None = None,
    comparison_id: str | None = None,
    cohort_records: list[LandmarkPatientRecord] | None = None,
) -> BaselineRunResult:
    config.validate()
    _seed_everything(config.seed)
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
        raise ValueError("Preflighted cohort landmark does not match the baseline configuration")
    loaders = _make_loaders(records, config=config, device=device)
    run_id = str(uuid.uuid4())
    paired_id = comparison_id or str(uuid.uuid4())
    checkpoint_dir = config.checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    result_store = SurvivalResultStore(database_engine)
    result_store.ensure_schema()
    result_store.start_run(
        run_id,
        comparison_id=paired_id,
        model_type="baseline",
        landmark_months=config.landmark_months,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        records=records,
    )

    try:
        model = CoxRiskModel(
            OrthogonalSliceResNet101Encoder(weights=ResNet101_Weights.IMAGENET1K_V2 if config.pretrained else None)
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
            train_loss, train_c_index = train_baseline_epoch(
                model,
                loaders["train"],
                optimizer,
                device=device,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            validation_loss, validation_c_index, _ = evaluate_baseline(
                model,
                loaders["validation"],
                device=device,
            )
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_total_loss=train_loss,
                train_survival_loss=train_loss,
                train_auxiliary_loss=None,
                train_c_index=train_c_index,
                validation_total_loss=validation_loss,
                validation_survival_loss=validation_loss,
                validation_auxiliary_loss=None,
                validation_c_index=validation_c_index,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                checkpoint_path=checkpoint_path,
            )
            _save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=epoch_metrics,
            )
            result_store.record_epoch(run_id, epoch_metrics)
            improved = validation_c_index > best_validation_c_index or (
                validation_c_index == best_validation_c_index and validation_loss < best_validation_loss
            )
            if improved:
                best_epoch = epoch
                best_validation_loss = validation_loss
                best_validation_c_index = validation_c_index
                best_checkpoint = checkpoint_path
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

        if best_checkpoint is None:
            raise RuntimeError("Baseline training did not produce a best checkpoint")
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_c_index, predictions = evaluate_baseline(model, loaders["test"], device=device)
        result_store.record_predictions(
            run_id,
            [
                PredictionRecord(
                    ptid=ptid,
                    image_ids=image_history,
                    observed_time_months=float(time_value),
                    event_observed=bool(event_value),
                    risk_score=float(risk_value),
                    survival_curve=None,
                )
                for ptid, image_history, time_value, event_value, risk_value in zip(
                    predictions.ptids,
                    predictions.image_ids,
                    predictions.times.tolist(),
                    predictions.events.tolist(),
                    predictions.risks.tolist(),
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
        return BaselineRunResult(
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
