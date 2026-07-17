"""Train and evaluate LTSA on a fixed-landmark longitudinal MRI cohort.

The module builds split-specific sequence loaders, optimizes discrete survival
and next-visit feature objectives, selects checkpoints using validation
concordance, evaluates the selected model on the test split, and persists the
complete run through the unified survival result store.
"""

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
    """Configuration for one fixed-landmark LTSA experiment.

    Attributes:
        landmark_months: Prediction landmark measured in months from the
            baseline image.
        epochs: Maximum number of training epochs.
        patience: Consecutive non-improving validation epochs allowed before
            early stopping.
        batch_size: Number of patient sequences per mini-batch.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight-decay coefficient.
        gradient_clip_norm: Maximum gradient norm. Set to ``0`` to disable
            clipping.
        bin_width_months: Width of each discrete survival interval in months.
        censoring_beta: Weighting parameter for the discrete censoring NLL.
        auxiliary_loss_weight: Multiplier applied to next-visit feature MSE.
        n_heads: Number of Transformer attention heads.
        n_layers: Number of causal Transformer encoder layers.
        dropout: Dropout probability used by LTSA.
        seed: Seed used for Python, PyTorch, CUDA, and data-loader shuffling.
        num_workers: Worker processes assigned to each data loader.
        device: PyTorch device specification, or ``"auto"`` to prefer CUDA,
            then MPS, then CPU.
        pretrained: Whether to initialize ResNet-101 with ImageNet weights.
        tensor_dir: Directory containing patient tensor packages.
        checkpoint_root: Root directory for run-specific checkpoints.
        spatial_size: Optional ``(depth, height, width)`` used to resize MRI
            volumes. ``None`` preserves stored dimensions.
    """

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
        """Validate training, optimization, and architecture settings.

        Raises:
            ValueError: If the landmark or optimizer settings are out of
                range, loss weights are invalid, worker or architecture
                counts are invalid, or ``n_heads`` does not divide the
                ResNet-101 feature width.
        """

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
    """Differentiable losses and survival outputs for one batch.

    Attributes:
        total: Weighted sum of survival and auxiliary losses.
        survival: Discrete negative log-likelihood at each patient's final
            valid visit.
        auxiliary: Mean squared error over valid next-visit feature targets.
        risk: Scalar risk derived from negative restricted mean survival time.
        survival_curves: Final-visit discrete survival curve for each patient.
    """

    total: Tensor
    survival: Tensor
    auxiliary: Tensor
    risk: Tensor
    survival_curves: Tensor


@dataclass(frozen=True)
class LTSAPredictionBundle:
    """Aligned test predictions, outcomes, and patient metadata.

    Attributes:
        risks: One scalar risk score per patient.
        times: Landmark-relative event or censoring times.
        events: Boolean event indicators.
        survival_curves: Discrete final-visit survival curves.
        ptids: Patient identifiers in tensor row order.
        image_ids: Chronological image histories corresponding to each patient.
    """

    risks: Tensor
    times: Tensor
    events: Tensor
    survival_curves: Tensor
    ptids: list[str]
    image_ids: list[tuple[str, ...]]


@dataclass(frozen=True)
class LTSARunResult:
    """Summary and patient predictions returned by a completed LTSA run.

    Attributes:
        run_id: Unique identifier for the LTSA model run.
        comparison_id: Identifier shared with a matched baseline run.
        best_epoch: Epoch selected by validation performance.
        validation_c_index: Validation C-index at the selected epoch.
        test_c_index: Test C-index from the selected checkpoint.
        predictions: Patient-level test predictions and outcomes.
    """

    run_id: str
    comparison_id: str
    best_epoch: int
    validation_c_index: float
    test_c_index: float
    predictions: LTSAPredictionBundle


def resolve_device(requested_device: str) -> torch.device:
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


def seed_everything(seed: int) -> None:
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


def make_landmark_loaders(
    records: list[LandmarkPatientRecord],
    *,
    config: LTSATrainingConfig,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Build train, validation, and test sequence loaders.

    Training sequences are shuffled with a seeded generator. Validation and
    test order remain deterministic. CUDA runs enable pinned host memory, and
    worker processes remain persistent when ``num_workers`` is positive.

    Args:
        records: Preflighted landmark-cohort records across all data splits.
        config: LTSA data-loading and preprocessing configuration.
        device: Device used to determine whether pinned memory is beneficial.

    Returns:
        A mapping containing ``"train"``, ``"validation"``, and ``"test"``
        data loaders.
    """

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
    """Select each patient's final valid visit from a padded tensor.

    Args:
        values: Tensor whose first dimensions are ``[batch, visits]``.
        sequence_lengths: Number of valid visits for each batch row.

    Returns:
        Tensor containing one selected visit per patient and preserving all
        dimensions after ``visits``.
    """

    return values[torch.arange(values.shape[0], device=values.device), sequence_lengths - 1]


def compute_ltsa_loss(
    outputs: LTSAOutputs,
    batch: LandmarkBatch,
    grid: DiscreteTimeGrid,
    *,
    censoring_beta: float,
    auxiliary_loss_weight: float,
) -> LTSALossBundle:
    """Compute LTSA survival, auxiliary, and total objectives.

    Survival NLL is evaluated from hazards at each patient's final valid
    landmark visit. Censoring indicators are obtained by negating the batch
    event flags. Auxiliary MSE includes only visits with a valid subsequent
    visit; batches without such pairs receive a differentiable zero auxiliary
    loss.

    Args:
        outputs: Visit-level LTSA predictions and validity masks.
        batch: Landmark batch containing outcomes and sequence lengths.
        grid: Discrete time grid used to encode outcomes and convert survival
            curves to risk.
        censoring_beta: Censoring NLL weighting parameter.
        auxiliary_loss_weight: Multiplier applied to auxiliary feature MSE in
            the total loss.

    Returns:
        Batch losses, scalar risks, and final-visit survival curves.
    """
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
    """Run one training or evaluation epoch.

    Supplying an optimizer enables gradients, parameter updates, and optional
    gradient clipping. Passing ``None`` evaluates the model without gradients.
    Losses are averaged by patient count rather than batch count.

    Args:
        model: LTSA model to train or evaluate.
        loader: Landmark sequence data loader for one split.
        grid: Discrete time grid fitted on the training cohort.
        config: Loss and gradient-clipping configuration.
        device: Device receiving each collated batch.
        optimizer: Optimizer used for training, or ``None`` for evaluation.

    Returns:
        A tuple containing mean total loss, mean survival loss, mean auxiliary
        loss, and Harrell C-index, in that order.

    Raises:
        ValueError: If the data loader yields no patients.
    """

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
    """Collect final-visit LTSA predictions without gradients.

    The model is placed in evaluation mode. Survival curves and scalar risks
    are taken from each patient's final valid visit and returned on the CPU
    alongside aligned outcomes and identifiers.

    Args:
        model: Trained LTSA model.
        loader: Landmark sequence loader to evaluate.
        grid: Discrete time grid used to convert survival curves to risk.
        device: Device receiving each collated batch.

    Returns:
        Concatenated risks, outcomes, survival curves, patient identifiers, and
        image histories.

    Raises:
        ValueError: If the data loader yields no predictions.
    """

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
    """Atomically save model, optimizer, configuration, grid, and metrics.

    The checkpoint is first written to a sibling temporary file and then
    replaced into its final path, reducing the chance of leaving a partially
    written checkpoint after interruption.

    Args:
        checkpoint_path: Final checkpoint destination.
        run_id: Identifier of the experiment run.
        epoch: Epoch represented by the checkpoint.
        model: LTSA model whose state is saved.
        optimizer: Optimizer whose state is saved.
        config: Training configuration serialized into the checkpoint.
        grid: Fitted discrete time grid.
        metrics: Training and validation measurements for the epoch.
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
    """Train, select, evaluate, and persist one fixed-landmark LTSA run.

    When no cohort is supplied, the function queries and validates one from
    the project database. A supplied cohort allows paired orchestration to
    reuse the exact records preflighted for the baseline. The discrete grid is
    fitted only on training durations.

    Every epoch is checkpointed and persisted. Selection maximizes validation
    C-index and uses validation total loss to break C-index ties. Training
    stops after ``patience`` consecutive non-improving epochs. The selected
    checkpoint is restored before test prediction and persistence.

    Args:
        config: Validated experiment, architecture, and data configuration.
        engine: Optional SQLAlchemy engine. If ``None``, uses the project
            database engine.
        comparison_id: Optional identifier shared with a matched baseline.
            A new identifier is generated when omitted.
        cohort_records: Optional preflighted landmark cohort. Every record must
            match ``config.landmark_months``.

    Returns:
        Run identifiers, selected epoch, validation and test C-indices, and
        patient-level test predictions.

    Raises:
        ValueError: If configuration or cohort validation fails, the supplied
            cohort uses a different landmark, or an epoch loader is empty.
        FileExistsError: If the generated run checkpoint directory already
            exists.
        RuntimeError: If training does not produce a selectable checkpoint.

    Note:
        Exceptions raised after the run row is created mark that row as
        failed before the original exception is re-raised.
    """

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
    """Parse command-line arguments for a standalone LTSA run.

    Args:
        argv: Optional argument sequence. ``None`` reads arguments from the
            current process.

    Returns:
        Parsed command-line namespace.
    """

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
    """Run LTSA training from command-line arguments.

    Args:
        argv: Optional argument sequence. ``None`` reads arguments from the
            current process.

    Returns:
        Unique identifier of the completed LTSA run.
    """

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
