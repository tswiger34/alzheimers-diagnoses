"""Train matched continuous-survival MRI baselines A/B (and optional history).

Example: python -m mstat_project.ml.survival --variants a b --epochs 2
The default is two single-image baselines; no test outcome selects a checkpoint.
"""

import argparse
import json
import logging
import math
import random
import uuid
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sqlalchemy import text
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights

from mstat_project.data_prep.data_splitting import frame_fingerprint
from mstat_project.ml.continuous_survival import ContinuousMRIModel
from mstat_project.ml.ltsa import resolve_device
from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.ml.survival_comparison import compare_predictions
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
    spatial_size: tuple[int, int, int] = (96, 112, 96)
    quadrature_points: int = 32
    bootstrap_samples: int = 1000
    cpu_threads: int = 2
    diagnostics: bool = False
    freeze_encoder: bool = False
    horizons: tuple[float, ...] = (12.0, 24.0, 60.0)
    tensor_dir: Path = field(default_factory=default_tensor_dir)
    output_dir: Path = field(default_factory=lambda: default_checkpoint_dir("continuous"))

    def validate(self):
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


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(records, config, variant, n_slots):
    return DataLoader(
        SurvivalDataset(records, history=variant == "history", n_slots=n_slots, spatial_size=config.spatial_size),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        collate_fn=collate_survival, pin_memory=config.device.startswith("cuda"),
    )


def _device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device, clip):
    model.train()
    total, count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        batch = _device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training likelihood")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else math.inf, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(batch["records"])
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
    return numerator / denominator


@torch.no_grad()
def predict(model, loader, device, horizons):
    model.eval()
    predictions = []
    for batch in loader:
        batch = _device_batch(batch, device)
        z = model.encode(batch["images"], batch["times"], batch["observed"])
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
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_variant(config, records, schedule, variant, output, *, encoder_factory=None):
    """Train one dynamic model, select by validation NLL, test once at each canonical t."""
    output.mkdir(parents=True, exist_ok=False)
    _seed(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("Variant %s on %s; output: %s", variant, device, output.resolve())
    encoder = encoder_factory() if encoder_factory else OrthogonalSliceResNet101Encoder(
        weights=ResNet101_Weights.IMAGENET1K_V2 if config.pretrained else None
    )
    model = ContinuousMRIModel(encoder, variant, quadrature_points=config.quadrature_points,
                               freeze_encoder=config.freeze_encoder).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info("Encoder %s; trainable parameters: %s",
                "frozen (evaluation mode)" if config.freeze_encoder else "trainable",
                f"{sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    # Same scheduler in every arm; stepping depends only on validation likelihood.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    train = [r for r in records if r.split == "train"]
    val = _loader([r for r in records if r.split == "validation"], config, variant, schedule.height)
    best, best_epoch, stale = math.inf, 0, 0
    epochs = []
    panel = select_diagnostic_panel(records, config.seed) if config.diagnostics else []

    def save_diagnostics(model, epoch):
        report = trace_diagnostic_panel(model, panel, spatial_size=config.spatial_size,
                                        device=device, horizons=config.horizons)
        path = output / f"diagnostics_epoch_{epoch:03d}.json"
        write_json(path, {"epoch": epoch, "variant": variant, "freeze_encoder": config.freeze_encoder, **report})
        LOGGER.info("  Diagnostics epoch %d: %s", epoch, path)
        for warning in report["warnings"]:
            LOGGER.warning("  Diagnostics: %s", warning)

    if config.diagnostics:
        save_diagnostics(model, 0)
    for epoch in range(1, config.epochs + 1):
        LOGGER.info("Baseline/model %s epoch %d/%d", variant, epoch, config.epochs)
        sampled = sample_epoch(train, config.seed, epoch)
        pl.DataFrame({"ptid": [r.ptid for r in sampled], "image_id": [r.image_ids[-1] for r in sampled],
                      "acquisition_months": [r.acquisition_months[-1] for r in sampled]}).write_csv(output / f"sampled_epoch_{epoch:03d}.csv")
        training = train_epoch(model, _loader(sampled, config, variant, schedule.height), optimizer, device, config.gradient_clip_norm)
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
                 "best_validation_nll": best, "epochs": epochs}
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
    for t in schedule["follow_up_months"]:
        landmark = float(t)
        rows = at_landmark(records, landmark, "test")
        LOGGER.info("  test landmark %.1f months: %d patients", landmark, len(rows))
        predictions = predict(model, _loader(rows, config, variant, schedule.height), device, config.horizons) if rows else []
        metrics[str(landmark)] = summarize_predictions(predictions, config.horizons)
        all_predictions.extend(predictions)
    summary = {"variant": variant, "best_epoch": best_epoch, "validation_nll": best,
               "horizons_months": config.horizons, "test_by_landmark": metrics}
    write_json(output / "summary.json", summary)
    write_json(output / "test_predictions.json", all_predictions)
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
    code_files = ("survival.py", "survival_data.py", "continuous_survival.py", "survival_comparison.py",
                  "models.py", "landmarks.py", "metrics.py", "utils.py", "survival_diagnostics.py")
    code_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in code_files
    }
    manifest = {"run_id": run_id, "config": asdict(config), "audit": audit,
                "code_sha256": code_hashes, "torch_version": torch.__version__,
                "hashes": {"source": frame_fingerprint(frame), "schedule": frame_fingerprint(schedule), "splits": frame_fingerprint(splits)},
                "protocol": {"objective": "continuous_conditional_nll", "selection": "patient_mean_validation_nll",
                             "augmentation": "none (shared)", "sampling": "one MRI per patient per epoch",
                             "outcome_source": "clinical examinations; months=days*12/365",
                             "risk_metric": "Harrell C-index truncated at each reported horizon",
                             "step_ahead_loss": "disabled", "encoder": "orthogonal center slices / ResNet101",
                             "comparison": "paired patient bootstrap within each landmark; candidate minus reference"}}
    write_json(output / "manifest.json", manifest)
    LOGGER.info("Run %s; audit: %s", run_id, audit)
    LOGGER.info("Artifacts: %s", output.resolve())
    handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    try:
        write_json(output / "status.json", {"status": "preflight"})
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
            write_json(output / "status.json", {"status": "running", "variant": variant})
            summaries.append(train_variant(config, records, schedule, variant, output / variant))
            predictions_by_variant[variant] = json.loads((output / variant / "test_predictions.json").read_text(encoding="utf-8"))
            write_json(output / "results.json", summaries)
            _persist_run(engine, run_id, output, "running", {**manifest, "results": summaries})
        LOGGER.info("Computing paired comparisons of the saved test predictions")
        comparisons = compare_predictions(
            predictions_by_variant, config.horizons, bootstrap_samples=config.bootstrap_samples, seed=config.seed,
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
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
    p.add_argument("--freeze-encoder", action="store_true",
                   help="freeze image encoder weights and statistics; train the projection and survival components")
    p.add_argument("--horizons", type=float, nargs="+", default=(12,24,60))
    p.add_argument("--tensor-dir", type=Path, default=default_tensor_dir())
    p.add_argument("--output-dir", type=Path, default=default_checkpoint_dir("continuous"))
    p.add_argument("--preflight-only", action="store_true")
    args = vars(p.parse_args(argv))
    args["pretrained"] = not args.pop("no_pretrained")
    preflight = args.pop("preflight_only")
    for k in ("variants", "spatial_size", "horizons"):
        args[k] = tuple(args[k])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run(SurvivalConfig(**args), preflight_only=preflight)


if __name__ == "__main__":
    main()
