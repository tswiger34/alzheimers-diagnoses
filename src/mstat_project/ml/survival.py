"""Train matched continuous-survival MRI baselines A/B (and optional history).

Example: python -m mstat_project.ml.survival --variants a b --epochs 2
The default is two single-image baselines; no test outcome selects a checkpoint.
"""

import argparse
import json
import logging
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sqlalchemy import text
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights

from mstat_project.data_prep.data_splitting import frame_fingerprint
from mstat_project.ml.continuous_survival import ARCHITECTURE_VERSION, TIME_TRANSFORM, ContinuousMRIModel, initial_hazard
from mstat_project.ml.survival_embeddings import (
    CachedEncoder, EmbeddingDataset, collate_embeddings, load_embeddings, state_fingerprint,
    preprocessing_fingerprint,
)
from mstat_project.ml.survival_optimization import (
    parameter_counts, parameter_snapshot, norms, optimization_summary, configure_final_block, optimizer_groups,
    capture_training_activations,
)
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, read_json, require_debug_gate, require_baseline_gate, cohort_fingerprint, implementation_hashes,
    file_sha256,
)
from mstat_project.ml.ltsa import resolve_device
from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.ml.survival_comparison import compare_experiments
from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel
from mstat_project.ml.survival_data import (
    SurvivalDataset, at_landmark, build_observations, collate_survival,
    load_source, preflight_tensors, sample_epoch,
)
from mstat_project.ml.utils import concordance_index, default_checkpoint_dir, default_tensor_dir
from mstat_project.utils import get_db_engine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurvivalConfig:
    variants: tuple[str, ...] = ("a", "b")
    epochs: int = 30
    patience: int = 4
    batch_size: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    spatial_size: tuple[int, int, int] = (96, 112, 96)
    quadrature_points: int = 32
    bootstrap_samples: int = 1000
    cpu_threads: int = 2
    diagnostics: bool = False
    freeze_encoder: bool = True
    cache_path: Path | None = None
    baseline_init: Path | None = None
    evaluate_test: bool = False
    n_layers: int = 1
    d_model: int = 128
    dropout: float = 0.1
    auxiliary_weight: float = 0.0
    anomaly_detection: bool = False
    fine_tune_final_block: bool = False
    encoder_learning_rate: float = 1e-5
    experiment_id: str | None = None
    debug_gate: Path | None = None
    baseline_gate: Path | None = None
    protocol_path: Path = PROTOCOL_PATH
    horizons: tuple[float, ...] = (12.0, 24.0, 60.0)
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    output_dir: Path = field(default_factory=lambda: default_checkpoint_dir("continuous"))

    def validate(self):
        if self.evaluate_test:
            raise ValueError("Training is validation-only; use ltsa_experiments report for locked test evaluation")
        if self.fine_tune_final_block and (not self.baseline_init or self.variants != ("history",)):
            raise ValueError("Final-block fine-tuning requires a history checkpoint and variants=history")
        if self.auxiliary_weight and self.variants != ("history",):
            raise ValueError("Step-ahead experiments require variants=history")
        if self.experiment_id and len(self.variants) != 1:
            raise ValueError("experiment_id requires exactly one variant")
        if self.cache_path and (not self.freeze_encoder or self.fine_tune_final_block):
            raise ValueError("Cached inputs require a fully frozen encoder")
        if self.n_layers < 1 or self.d_model not in (128, 256) or not 0 <= self.dropout < 1:
            raise ValueError("Invalid temporal architecture")
        if not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")
        if self.auxiliary_weight and self.fine_tune_final_block:
            raise ValueError("Step-ahead targets require the frozen-encoder experiment")
        if not 0 < self.encoder_learning_rate < self.learning_rate and self.fine_tune_final_block:
            raise ValueError("Encoder learning rate must be positive and smaller than module learning rate")
        if self.experiment_id and (Path(self.experiment_id).name != self.experiment_id or self.experiment_id in (".", "..")):
            raise ValueError("experiment_id must be a single directory name")
        if not self.variants or len(set(self.variants)) != len(self.variants) or set(self.variants) - {"a", "b", "history"}:
            raise ValueError("Choose unique variants from a, b, history")
        if min(self.epochs, self.patience, self.batch_size, self.cpu_threads) < 1 or self.num_workers < 0:
            raise ValueError("Invalid training counts")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples cannot be negative")
        if len(self.spatial_size) != 3 or min(self.spatial_size) < 16:
            raise ValueError("Spatial size must contain three dimensions >= 16")
        if self.quadrature_points < 8:
            raise ValueError("At least 8 quadrature points required")
        for v in (self.learning_rate, self.weight_decay, self.gradient_clip_norm, *self.horizons):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Training parameters and horizons must be finite and nonnegative")
        if self.learning_rate == 0 or not self.horizons or min(self.horizons) <= 0:
            raise ValueError("Learning rate and horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Horizons must be unique and increasing")


def _replace_artifact(temporary, path):
    """Publish a closed file atomically, tolerating short Windows sharing locks.

    Keep the old destination and the complete temporary file on persistent
    failure. Never delete the destination to make a replacement succeed.
    """
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if not isinstance(exc, PermissionError) and getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == 7:
                raise PermissionError(
                    exc.errno,
                    f"Could not replace {path} after 8 attempts. The previous artifact is unchanged; "
                    f"the new file is retained at {temporary}. Check for an open file or persistent "
                    f"access restrictions. Last error: {exc}",
                    str(path),
                ) from exc
            time.sleep(min(.05 * 2**attempt, 1.))


def write_json(path, value):
    path = Path(path)
    serialized = json.dumps(value, indent=2, default=str, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    _replace_artifact(temporary, path)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(records, config, variant, n_slots):
    if config.cache_path is not None:
        dataset = EmbeddingDataset(records, cache_path=config.cache_path, history=variant == "history",
                                   spatial_size=config.spatial_size)
        collate = collate_embeddings
    else:
        dataset = SurvivalDataset(records, history=variant == "history", n_slots=n_slots, spatial_size=config.spatial_size)
        collate = collate_survival
    return DataLoader(
        dataset,
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        collate_fn=collate, pin_memory=config.device.startswith("cuda"),
    )


def _device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device, clip, *, auxiliary_weight=0., diagnostics=None, anomaly=False):
    device = torch.device(device)
    model.train()
    total, count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        batch = _device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with capture_training_activations(model, diagnostics is not None) as activations:
            with torch.autograd.detect_anomaly(check_nan=True) if anomaly else nullcontext():
                z = model.encode_batch(batch)
                cumulative, event_log = model.head.nll_components(z, batch["entry"], batch["outcome"], batch["event"])
                auxiliary = model.auxiliary_loss(batch) if auxiliary_weight else z.sum() * 0
                loss = (cumulative + event_log).mean() + auxiliary_weight * auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training likelihood")
                loss.backward()
        if diagnostics is not None:
            report = optimization_summary(model, batch, z, cumulative, event_log, auxiliary)
            report["gradient_norms"] = norms(model)
            report["activations"] = activations
            report["warnings"] = [f"Zero gradient norm: {name}" for name, value in report["gradient_norms"].items() if value == 0]
            before = parameter_snapshot(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else math.inf, error_if_nonfinite=True)
        optimizer.step()
        if diagnostics is not None:
            report["update_norms"] = norms(model, before=before)
            diagnostics.append(report)
        total += float((cumulative + event_log).detach().mean()) * len(batch["records"])
        count += len(batch["records"])
        if step % 100 == 0:
            LOGGER.info("  train %d/%d batches, mean NLL %.5f", step, len(loader), total / count)
            if device.type == "cuda":
                LOGGER.info(
                    "  GPU memory allocated %.0f MiB, reserved %.0f MiB, peak allocated %.0f MiB",
                    torch.cuda.memory_allocated(device) / 2**20,
                    torch.cuda.memory_reserved(device) / 2**20,
                    torch.cuda.max_memory_allocated(device) / 2**20,
                )
    if not count:
        raise ValueError("Training loader is empty")
    return total / count


@torch.no_grad()
def validation_loss(model, loader, device):
    """Average MRI-landmark losses within patients, then equally across patients."""
    model.eval()
    numerator, denominator = 0.0, 0.0
    for batch in loader:
        batch = _device_batch(batch, device)
        loss = model(batch)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite validation likelihood")
        numerator += float((loss * batch["weight"]).sum())
        denominator += float(batch["weight"].sum())
    if not denominator:
        raise ValueError("Validation loader is empty")
    return numerator / denominator


@torch.no_grad()
def predict(model, loader, device, horizons):
    model.eval()
    predictions = []
    for batch in loader:
        batch = _device_batch(batch, device)
        z = model.encode_batch(batch)
        nll = model.head.nll(z, batch["entry"], batch["outcome"], batch["event"])
        survival = torch.stack([
            model.head.survival(z, batch["entry"], batch["entry"] + h) for h in horizons
        ], dim=1)
        if not torch.isfinite(survival).all() or not torch.isfinite(nll).all():
            raise FloatingPointError("Non-finite evaluation predictions")
        for r, loss, curve in zip(batch["records"], nll.cpu().tolist(), survival.cpu().tolist(), strict=True):
            predictions.append({
                "ptid": r.ptid, "split": r.split, "landmark_months": r.prediction_months,
                "acquisition_months": r.acquisition_months[-1], "image_age_months": r.prediction_months-r.acquisition_months[-1],
                "outcome_months": r.outcome_months, "event": r.event,
                "image_ids": list(r.image_ids if model.variant == "history" else r.image_ids[-1:]),
                "available_history_ids": list(r.image_ids), "visit_nums": list(r.visit_nums),
                "nll": loss, "survival": curve,
            })
    return predictions


def summarize_predictions(predictions, horizons):
    if not predictions:
        return {"patients": 0, "events": 0, "nll": None, "c_index": [None] * len(horizons)}
    times = torch.tensor([p["outcome_months"] - p["landmark_months"] for p in predictions], dtype=torch.float64)
    events = torch.tensor([p["event"] for p in predictions])
    indices = []
    for j, horizon in enumerate(horizons):
        risk = torch.tensor([1-p["survival"][j] for p in predictions], dtype=torch.float64)
        c = concordance_index(risk, times.clamp_max(horizon), events & (times <= horizon))
        indices.append(c if math.isfinite(c) else None)
    return {"patients": len(predictions), "events": int(events.sum()),
            "nll": sum(p["nll"] for p in predictions) / len(predictions), "c_index": indices}


def _save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    _replace_artifact(temporary, path)


def model_identity(model, records, config):
    if config.cache_path:
        manifest = load_embeddings(config.cache_path, spatial_size=config.spatial_size)[0]["manifest"]
        encoder_hash = manifest["encoder_sha256"]
    else:
        encoder_hash = state_fingerprint(model.encoder)
    return {"architecture_version": ARCHITECTURE_VERSION, "time_transform": TIME_TRANSFORM,
            "spatial_size": list(config.spatial_size), "encoder_sha256": encoder_hash,
            "quadrature_points": config.quadrature_points,
            "preprocessing_sha256": preprocessing_fingerprint(),
            "cohort_sha256": cohort_fingerprint(records)}


def transfer_baseline(model, checkpoint, identity):
    if checkpoint.get("identity") != identity or checkpoint.get("variant") != "b":
        raise ValueError("Baseline transfer requires version-2 B with identical encoder, preprocessing and cohort")
    state = checkpoint["model"]
    if int(state.get("_architecture_version", -1)) != ARCHITECTURE_VERSION:
        raise ValueError("Incompatible baseline architecture")
    for name in ("projection", "projection_norm", "time_embedding", "head"):
        getattr(model, name).load_state_dict({k.removeprefix(name + "."): v for k, v in state.items()
                                            if k.startswith(name + ".")})


@torch.no_grad()
def transfer_prediction_check(model, loader, device, horizons):
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    alpha = model.alpha.detach().clone()
    maximum = 0.
    count = 0
    try:
        for batch in loader:
            batch = _device_batch(batch, device)
            count += len(batch["records"])
            model.alpha.copy_(alpha)
            z_history = model.encode_batch(batch)
            model.alpha.zero_()
            z_baseline = model.encode_batch(batch)
            for horizon in horizons:
                end = batch["entry"] + horizon
                gap = (model.head.survival(z_history, batch["entry"], end)
                       - model.head.survival(z_baseline, batch["entry"], end)).abs()
                if not torch.isfinite(gap).all():
                    raise FloatingPointError("Nonfinite baseline transfer predictions")
                maximum = max(maximum, float(gap.max()))
    finally:
        model.alpha.copy_(alpha)
        for module, training in modes:
            module.training = training
    return {"max_survival_difference": maximum, "tolerance": 0.01,
            "observations": count, "passed": count > 0 and maximum <= 0.01}


def train_variant(config, records, schedule, variant, output, *, encoder_factory=None):
    """Low-level trainer used by synthetic tests and the gated production entry points.

    Production callers supply debug_gate; both CLI entry points require it.
    This function never evaluates held-out test observations.
    """
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    _seed(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("Variant %s on %s; output: %s", variant, device, output.resolve())
    if config.cache_path:
        encoder = CachedEncoder(load_embeddings(config.cache_path, spatial_size=config.spatial_size)[0]["manifest"])
    else:
        encoder = encoder_factory() if encoder_factory else OrthogonalSliceResNet101Encoder(
            weights=ResNet101_Weights.IMAGENET1K_V2 if config.pretrained else None
        )
    model = ContinuousMRIModel(encoder, variant, quadrature_points=config.quadrature_points,
                               freeze_encoder=config.freeze_encoder, baseline_hazard=initial_hazard(records),
                               n_layers=config.n_layers, d_model=config.d_model, dropout=config.dropout,
                               step_ahead=config.auxiliary_weight > 0).to(device)
    identity = model_identity(model, records, config)
    initialization_sha256 = file_sha256(config.baseline_init) if config.baseline_init else None
    if config.debug_gate is not None:
        protocol = read_json(config.protocol_path)
        require_debug_gate(config.debug_gate, protocol, identity, records=records)
        if variant == "history" and not config.fine_tune_final_block:
            if config.baseline_init is None:
                raise ValueError("History training requires initialization from promoted baseline B")
            require_baseline_gate(config.baseline_gate, protocol, identity, config.baseline_init)
    if config.baseline_init:
        checkpoint = torch.load(config.baseline_init, map_location="cpu", weights_only=False)
        if config.fine_tune_final_block:
            if checkpoint.get("identity") != identity or checkpoint.get("variant") != "history":
                raise ValueError("Fine-tuning requires a matching frozen history checkpoint")
            if checkpoint["config"].get("fine_tune_final_block"):
                raise ValueError("Initialize final-block fine-tuning from a frozen history checkpoint")
            if config.debug_gate is not None:
                from mstat_project.ml.ltsa_experiments import stable_run
                parent_baseline = checkpoint["config"].get("baseline_init")
                require_baseline_gate(config.baseline_gate, protocol, identity, parent_baseline)
                if not stable_run(config.baseline_init.parent):
                    raise ValueError("Fine-tuning requires a stable frozen history experiment")
            raw_state = model.state_dict()
            raw_state.update({k: v for k, v in checkpoint["model"].items() if not k.startswith("encoder.")})
            model.load_state_dict(raw_state)
        else:
            if variant != "history":
                raise ValueError("baseline_init is only supported for history")
            transfer_baseline(model, checkpoint, identity)
    if config.fine_tune_final_block:
        configure_final_block(model)
    write_json(output / "parameter_counts.json", parameter_counts(model))
    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info("Encoder %s; trainable parameters: %s",
                "frozen (evaluation mode)" if config.freeze_encoder else "trainable",
                f"{sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.learning_rate, config.encoder_learning_rate),
                                  weight_decay=config.weight_decay)
    # Same scheduler in every arm; stepping depends only on validation likelihood.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    train = [r for r in records if r.split == "train"]
    val = _loader([r for r in records if r.split == "validation"], config, variant, schedule.height)
    best, best_epoch, stale = math.inf, 0, 0
    epochs = []
    panel = select_diagnostic_panel(records, config.seed, latest=True) if config.diagnostics or config.baseline_init else []
    if config.baseline_init and not config.fine_tune_final_block:
        check = transfer_prediction_check(model, _loader(panel, config, variant, schedule.height), device, config.horizons)
        write_json(output / "initialization_check.json", check)
        if not check["passed"]:
            raise ValueError("LTSA initialization exceeds the baseline prediction tolerance; inspect initialization_check.json")

    def save_diagnostics(model, epoch):
        report = trace_diagnostic_panel(model, panel, spatial_size=config.spatial_size,
                                        device=device, horizons=config.horizons, cache_path=config.cache_path)
        path = output / f"diagnostics_epoch_{epoch:03d}.json"
        write_json(path, {"epoch": epoch, "variant": variant, "freeze_encoder": config.freeze_encoder, **report})
        LOGGER.info("  Diagnostics epoch %d: %s", epoch, path)
        for warning in report["warnings"]:
            LOGGER.warning("  Diagnostics: %s", warning)

    if config.diagnostics:
        save_diagnostics(model, 0)
    if config.baseline_init:
        best = validation_loss(model, val, device)
        _save_checkpoint(output / "best.pt", {"model": model.state_dict(), "config": asdict(config),
            "variant": variant, "epoch": 0, "best_epoch": 0, "best_validation_nll": best,
            "identity": identity, "epochs": [], "implementation_hashes": implementation_hashes(),
            "initialization_sha256": initialization_sha256})
        epochs.append({"epoch": 0, "train_nll": None, "validation_nll": best,
                       "learning_rate": config.learning_rate})
    # Auxiliary-head construction must not shift the dropout stream in matched ablations.
    _seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        LOGGER.info("Baseline/model %s epoch %d/%d", variant, epoch, config.epochs)
        sampled = sample_epoch(train, config.seed, epoch)
        pl.DataFrame({"ptid": [r.ptid for r in sampled], "image_id": [r.image_ids[-1] for r in sampled],
                      "acquisition_months": [r.acquisition_months[-1] for r in sampled]}).write_csv(output / f"sampled_epoch_{epoch:03d}.csv")
        optimization = [] if config.diagnostics else None
        training = train_epoch(model, _loader(sampled, config, variant, schedule.height), optimizer, device,
                               config.gradient_clip_norm, auxiliary_weight=config.auxiliary_weight,
                               diagnostics=optimization, anomaly=config.anomaly_detection)
        if optimization is not None:
            write_json(output / f"optimization_epoch_{epoch:03d}.json", optimization)
        validation = validation_loss(model, val, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation)
        epochs.append({"epoch": epoch, "train_nll": training, "validation_nll": validation, "learning_rate": lr})
        improved = validation < best
        if improved:
            best, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": asdict(config), "variant": variant, "epoch": epoch, "best_epoch": best_epoch,
                 "best_validation_nll": best, "epochs": epochs, "identity": identity,
                 "implementation_hashes": implementation_hashes(), "initialization_sha256": initialization_sha256}
        _save_checkpoint(output / "last.pt", state)
        if improved:
            _save_checkpoint(output / "best.pt", state)
        write_json(output / "epochs.json", epochs)
        LOGGER.info("  train NLL %.5f; validation NLL %.5f; selected epoch %d", training, validation, best_epoch)
        if config.diagnostics:
            save_diagnostics(model, epoch)
        if stale >= config.patience:
            break
    # Release the last training state before loading a checkpoint. Loading its
    # optimizer tensors onto the GPU would otherwise duplicate hundreds of MiB.
    del state, optimizer, scheduler
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    metrics, all_predictions = {}, []
    evaluation_split = "validation"
    for t in schedule["follow_up_months"]:
        landmark = float(t)
        rows = at_landmark(records, landmark, evaluation_split)
        LOGGER.info("  %s landmark %.1f months: %d patients", evaluation_split, landmark, len(rows))
        predictions = predict(model, _loader(rows, config, variant, schedule.height), device, config.horizons) if rows else []
        metrics[str(landmark)] = summarize_predictions(predictions, config.horizons)
        all_predictions.extend(predictions)
    summary = {"variant": variant, "experiment_id": config.experiment_id or variant,
               "best_epoch": best_epoch, "validation_nll": best, "identity": identity,
               "horizons_months": config.horizons, f"{evaluation_split}_by_landmark": metrics}
    write_json(output / "summary.json", summary)
    write_json(output / f"{evaluation_split}_predictions.json", all_predictions)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _persist_run(engine, run_id, output, status, payload):
    """Separate new continuous results from incompatible legacy Cox/discrete runs."""
    with engine.begin() as c:
        c.execute(text("CREATE SCHEMA IF NOT EXISTS _ml"))
        c.execute(text("""CREATE TABLE IF NOT EXISTS _ml.continuous_survival_runs (
            run_id TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL, artifact_path TEXT NOT NULL, payload JSONB NOT NULL)"""))
        c.execute(text("""INSERT INTO _ml.continuous_survival_runs (run_id,status,artifact_path,payload)
            VALUES (:id,:status,:path,CAST(:payload AS JSONB))
            ON CONFLICT (run_id) DO UPDATE SET status=EXCLUDED.status, payload=EXCLUDED.payload, updated_at=NOW()"""),
            {"id": run_id, "status": status, "path": str(output.resolve()), "payload": json.dumps(payload, default=str, allow_nan=False)})


def run(config, *, preflight_only=False, engine=None):
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    engine = engine or get_db_engine()
    frame, schedule, splits = load_source(engine)
    records, audit = build_observations(frame, schedule, splits, config.tensor_dir)
    from mstat_project.ml.ltsa_experiments import validation_support
    audit["primary_validation_support"] = validation_support(records)
    if not preflight_only:
        if config.debug_gate is None:
            raise ValueError("Full training requires --debug-gate; run ltsa_experiments debug first")
        protocol = read_json(config.protocol_path)
        hashes = {"source": frame_fingerprint(frame), "schedule": frame_fingerprint(schedule),
                  "splits": frame_fingerprint(splits)}
        if hashes != protocol["snapshot_hashes"]:
            raise ValueError("Current database does not match the locked protocol snapshot")
        if config.cache_path:
            cache_manifest = load_embeddings(config.cache_path, provenance=hashes,
                                             spatial_size=config.spatial_size)[0]["manifest"]
            require_debug_gate(config.debug_gate, protocol, cache_manifest, records=records)
    run_id = str(uuid.uuid4())
    output = config.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    frame.write_parquet(output / "source.parquet")
    schedule.write_csv(output / "schedule.csv")
    splits.write_csv(output / "patient_splits.csv")
    visit_codes = dict(zip(schedule["visit_num"], schedule["visit_code"], strict=True))
    pl.DataFrame([{
        "ptid": r.ptid, "split": r.split, "image_id": r.image_ids[-1], "visit_num": r.visit_nums[-1],
        "visit_code": visit_codes[r.visit_nums[-1]], "acquisition_months": r.acquisition_months[-1],
        "outcome_months": r.outcome_months, "event": r.event,
    } for r in records]).write_csv(output / "training_landmarks.csv")
    code_hashes = implementation_hashes()
    manifest = {"run_id": run_id, "config": asdict(config), "audit": audit,
                "code_sha256": code_hashes, "torch_version": torch.__version__,
                "hashes": {"source": frame_fingerprint(frame), "schedule": frame_fingerprint(schedule), "splits": frame_fingerprint(splits)},
                "protocol": {"objective": "continuous_conditional_nll", "selection": "patient_mean_validation_nll",
                             "augmentation": "none (shared)", "sampling": "one MRI per patient per epoch",
                             "outcome_source": "clinical examinations; months=days*12/365",
                             "risk_metric": "Harrell C-index truncated at each reported horizon",
                             "step_ahead_weight": config.auxiliary_weight, "encoder": "orthogonal center slices / ResNet101",
                             "comparison": "paired patient bootstrap within each landmark; candidate minus reference"}}
    write_json(output / "manifest.json", manifest)
    LOGGER.info("Run %s; audit: %s", run_id, audit)
    LOGGER.info("Artifacts: %s", output.resolve())
    handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    try:
        write_json(output / "status.json", {"status": "preflight"})
        if config.cache_path:
            checked = EmbeddingDataset(records, cache_path=config.cache_path, history=True,
                                       spatial_size=config.spatial_size)
            audit.update(embedding_observations_checked=len(checked), selection="image_id", tensor_outcome_labels_used=False)
        else:
            LOGGER.info("Checking tensor image IDs for every eligible patient")
            audit.update(preflight_tensors(records))
        write_json(output / "manifest.json", manifest)
        if preflight_only:
            write_json(output / "status.json", {"status": "preflight_passed"})
            return output
        _persist_run(engine, run_id, output, "running", manifest)
        summaries = []
        predictions_by_variant = {}
        for variant in config.variants:
            experiment_id = config.experiment_id or variant
            write_json(output / "status.json", {"status": "running", "variant": variant})
            summaries.append(train_variant(config, records, schedule, variant, output / experiment_id))
            predictions_by_variant[experiment_id] = read_json(output / experiment_id / "validation_predictions.json")
            write_json(output / "results.json", summaries)
            _persist_run(engine, run_id, output, "running", {**manifest, "results": summaries})
        LOGGER.info("Computing paired comparisons of the saved validation predictions")
        comparisons = compare_experiments(
            predictions_by_variant, config.horizons, bootstrap_samples=config.bootstrap_samples, seed=config.seed,
            expected_split="validation",
        )
        write_json(output / "comparisons.json", comparisons)
        _persist_run(engine, run_id, output, "completed", {**manifest, "results": summaries, "comparisons": comparisons})
        write_json(output / "status.json", {"status": "completed"})
        LOGGER.info("Completed: %s", output)
    except Exception as exc:
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json(output / "status.json", failure)
        LOGGER.exception("Run failed")
        try:
            _persist_run(engine, run_id, output, "failed", {**manifest, **failure})
        except Exception as persistence_exc:
            exc.add_note(f"Additionally failed to record failure in Postgres: {persistence_exc}")
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    return output


def main(argv=None, *, default_variants=("a", "b")):
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="The baseline/ltsa/compare entry points accept --legacy for historical Cox/discrete experiments.",
    )
    p.add_argument("--variants", nargs="+", choices=("a", "b", "history"), default=list(default_variants))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip-norm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--spatial-size", type=int, nargs=3, default=(96,112,96))
    p.add_argument("--quadrature-points", type=int, default=32)
    p.add_argument("--bootstrap-samples", type=int, default=1000, help="paired patient resamples; 0 saves point estimates only")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--diagnostics", action="store_true",
                   help="trace four fixed train/validation MRIs before training and after each epoch")
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True,
                   help="freeze image encoder weights and statistics; train the projection and survival components")
    p.add_argument("--horizons", type=float, nargs="+", default=(12,24,60))
    p.add_argument("--tensor-dir", type=Path, default=default_tensor_dir())
    p.add_argument("--output-dir", type=Path, default=default_checkpoint_dir("continuous"))
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--cache-path", type=Path)
    p.add_argument("--baseline-init", type=Path)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--auxiliary-weight", type=float, default=0.)
    p.add_argument("--anomaly-detection", action="store_true")
    p.add_argument("--fine-tune-final-block", action="store_true")
    p.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    p.add_argument("--experiment-id")
    p.add_argument("--debug-gate", type=Path)
    p.add_argument("--baseline-gate", type=Path)
    p.add_argument("--protocol-path", type=Path, default=PROTOCOL_PATH)
    args = vars(p.parse_args(argv))
    args["pretrained"] = not args.pop("no_pretrained")
    preflight = args.pop("preflight_only")
    for k in ("variants", "spatial_size", "horizons"):
        args[k] = tuple(args[k])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run(SurvivalConfig(**args), preflight_only=preflight)


if __name__ == "__main__":
    main()
