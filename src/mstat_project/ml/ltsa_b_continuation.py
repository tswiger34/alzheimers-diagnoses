"""Matched full-cohort B continuation for the exploratory LTSA experiment."""

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
import torch

from mstat_project.ml.continuous_survival import ContinuousMRIModel
from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot, stable_run
from mstat_project.ml.ltsa_exploratory import check_sources, run_name, validation_predictions
from mstat_project.ml.survival import _loader, _save_checkpoint, _seed, model_identity, train_epoch, validation_loss, write_json
from mstat_project.ml.survival_comparison import compare_experiments
from mstat_project.ml.survival_data import sample_epoch
from mstat_project.ml.survival_diagnostics import select_diagnostic_panel, trace_diagnostic_panel
from mstat_project.ml.survival_embeddings import CachedEncoder, fingerprint
from mstat_project.ml.survival_optimization import optimizer_groups, parameter_counts
from mstat_project.ml.survival_protocol import PROTOCOL_PATH, file_sha256, implementation_hashes, read_json

LOGGER = logging.getLogger(__name__)


def matching_samples(frame, path):
    # Image IDs are identifiers, including when every CSV token is numeric.
    # Preserve leading zeroes and match the in-memory identifier schema.
    saved = pl.read_csv(path, schema_overrides={"ptid": pl.String, "image_id": pl.String,
                                               "acquisition_months": pl.Float64})
    return frame.equals(saved)


def verify_history(source, protocol, config, manifest, sources, debug_path):
    spec = read_json(source / "specification.json")
    report = read_json(source / "report.json")
    if (spec.get("runner_sha256") != file_sha256(Path(__file__).with_name("ltsa_exploratory.py"))
        or spec.get("implementation_hashes") != implementation_hashes()
        or spec.get("protocol_sha256") != fingerprint(protocol)
        or spec.get("cache_id") != manifest["cache_id"] or spec.get("sources") != sources
        or spec.get("debug_gate_sha256") != file_sha256(debug_path)
        or spec.get("selection_split") != "validation" or not report.get("exploratory")
        or not report.get("stable") or not stable_run(source / "history-1l")):
        raise ValueError("Control requires a completed stable LTSA run with matching provenance")
    artifacts = report.get("artifacts", {})
    required = ("history-1l/best.pt", "history-1l/epochs.json", "history-1l/validation_predictions.json",
                "b_validation_predictions.json", "specification.json")
    # Path.as_posix isn't used in the source report; Windows paths are native.
    if any(str(Path(name)) not in artifacts for name in required):
        raise ValueError("LTSA report lacks required artifact fingerprints")
    for relative, digest in artifacts.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or file_sha256(path) != digest:
            raise ValueError("LTSA artifact changed or escapes its run directory")
    for key in ("seed", "batch_size", "epochs", "patience", "learning_rate", "weight_decay",
                "gradient_clip_norm", "quadrature_points", "horizons", "num_workers", "anomaly_detection"):
        if fingerprint(spec["config"][key]) != fingerprint(asdict(config)[key]):
            raise ValueError(f"Control and LTSA training settings differ: {key}")
    return spec, report


def train_control(config, records, schedule, manifest, source_checkpoint, output, spec_sha, history_source):
    output.mkdir()
    _seed(config.seed)
    model = ContinuousMRIModel(CachedEncoder(manifest), "b", freeze_encoder=True,
                               quadrature_points=config.quadrature_points).to(config.device)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("identity") != model_identity(model, records, config) or checkpoint.get("variant") != "b":
        raise ValueError("Continuation initialization identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    expected = checkpoint["best_validation_nll"]
    del checkpoint  # Deliberately do not restore optimizer/scheduler state.
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.learning_rate), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3)
    train = [r for r in records if r.split == "train"]
    val = _loader([r for r in records if r.split == "validation"], config, "b", schedule.height)
    best = validation_loss(model, val, config.device)
    if abs(best - expected) > 1e-5:
        raise ValueError("Continuation epoch zero does not reproduce source B NLL")
    best_epoch, stale = 0, 0
    epochs = [{"epoch": 0, "train_nll": None, "validation_nll": best, "learning_rate": config.learning_rate}]
    panel = select_diagnostic_panel(records, config.seed, latest=True)
    identity = model_identity(model, records, config)
    write_json(output / "parameter_counts.json", parameter_counts(model))

    def payload(epoch):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": asdict(config), "variant": "b", "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_nll": best, "epochs": epochs, "identity": identity,
            "implementation_hashes": implementation_hashes(), "initialization_sha256": file_sha256(source_checkpoint),
            "specification_sha256": spec_sha, "exploratory": True, "production_promoted": False}

    def diagnostics(epoch):
        report = trace_diagnostic_panel(model, panel, spatial_size=config.spatial_size, device=config.device,
                                        horizons=config.horizons, cache_path=config.cache_path)
        write_json(output / f"diagnostics_epoch_{epoch:03d}.json", {"epoch": epoch, "variant": "b", **report})

    diagnostics(0)
    _save_checkpoint(output / "epoch_zero.pt", payload(0))
    _save_checkpoint(output / "best.pt", payload(0))
    write_json(output / "epoch_zero_predictions.json", validation_predictions(model, records, schedule, config))
    # Match the trainer's sampling seed and random stream reset before updates.
    _seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        sampled = sample_epoch(train, config.seed, epoch)
        frame = pl.DataFrame({"ptid": [r.ptid for r in sampled], "image_id": [r.image_ids[-1] for r in sampled],
                              "acquisition_months": [r.acquisition_months[-1] for r in sampled]})
        history_samples = history_source / "history-1l" / f"sampled_epoch_{epoch:03d}.csv"
        matched = None
        if history_samples.exists():
            matched = matching_samples(frame, history_samples)
            if not matched:
                raise ValueError("Continuation patient sampling differs from LTSA")
        frame.write_csv(output / f"sampled_epoch_{epoch:03d}.csv")
        optimization = []
        training = train_epoch(model, _loader(sampled, config, "b", schedule.height), optimizer, config.device,
                               config.gradient_clip_norm, diagnostics=optimization, anomaly=config.anomaly_detection)
        value = validation_loss(model, val, config.device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(value)
        epochs.append({"epoch": epoch, "train_nll": training, "validation_nll": value,
                       "learning_rate": lr, "matched_history_samples": matched})
        improved = value < best
        if improved:
            best, best_epoch, stale = value, epoch, 0
        else:
            stale += 1
        _save_checkpoint(output / "last.pt", payload(epoch))
        if improved:
            _save_checkpoint(output / "best.pt", payload(epoch))
        write_json(output / "epochs.json", epochs)
        write_json(output / f"optimization_epoch_{epoch:03d}.json", optimization)
        diagnostics(epoch)
        LOGGER.info("B continuation epoch %d: train %.6f, validation %.6f; selected %d", epoch, training, value, best_epoch)
        if stale >= config.patience:
            break
    del optimizer, scheduler
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    write_json(output / "validation_predictions.json", validation_predictions(model, records, schedule, config))
    summary = {"variant": "b", "experiment_id": "b-continuation", "validation_nll": best,
        "best_epoch": best_epoch, "epoch_zero_nll": epochs[0]["validation_nll"], "identity": identity,
        "exploratory": True, "production_promoted": False}
    write_json(output / "summary.json", summary)
    return summary


def run_control(workspace, records, schedule, protocol, protocol_path, device, debug_run, history_run, name):
    workspace = Path(workspace)
    if read_json(workspace / "protocol.json") != protocol:
        raise ValueError("Workspace protocol differs")
    config, manifest, gate, sources, debug_path, gate_path = check_sources(
        workspace, records, protocol, protocol_path, device, debug_run)
    source = workspace / "investigations" / run_name(history_run)
    history_spec, history_report = verify_history(source, protocol, config, manifest, sources, debug_path)
    output = workspace / "investigations" / run_name(name)
    initialization = config.baseline_init
    config = replace(config, variants=("b",), experiment_id="b-continuation", baseline_init=None, output_dir=output)
    config.validate()
    specification = {"control": "full-cohort-b-continuation", "selection_split": "validation",
        "exploratory": True, "production_promoted": False, "sources": sources,
        "history_report_sha256": file_sha256(source / "report.json"),
        "history_specification_sha256": file_sha256(source / "specification.json"),
        "debug_gate_sha256": file_sha256(debug_path), "baseline_gate_sha256": file_sha256(gate_path),
        "cache_id": manifest["cache_id"], "protocol_sha256": fingerprint(protocol),
        "implementation_hashes": implementation_hashes(), "runner_sha256": file_sha256(Path(__file__)),
        "exploratory_runner_sha256": history_spec["runner_sha256"],
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "initialization_checkpoint": str(initialization.resolve()),
        "optimizer_policy": "Fresh AdamW and ReduceLROnPlateau; no optimizer state transfer",
        "selection_policy": "Same early stopping and maximum epochs; epoch zero is eligible"}
    if output.exists():
        if not (output / "specification.json").exists() or read_json(output / "specification.json") != specification:
            raise ValueError("Control provenance changed; use a new --run-name")
        if not (output / "report.json").exists():
            raise ValueError("Incomplete control; preserve it and use a new --run-name")
        report = read_json(output / "report.json")
        if any(file_sha256(output / p) != sha for p, sha in report["artifacts"].items()):
            raise ValueError("Control artifacts changed")
        return output
    output.mkdir(parents=True)
    write_json(output / "specification.json", specification)
    summary = train_control(config, records, schedule, manifest, initialization, output / "b-continuation",
                            file_sha256(output / "specification.json"), source)
    comparisons = compare_experiments({
        "b": read_json(source / "b_validation_predictions.json"),
        "b-continuation": read_json(output / "b-continuation" / "validation_predictions.json"),
        "history-1l": read_json(source / "history-1l" / "validation_predictions.json")},
        config.horizons, bootstrap_samples=protocol["bootstrap_samples"], seed=protocol["bootstrap_seed"],
        expected_split="validation")
    for row in comparisons:
        row["primary"] = (row["reference"] == "b-continuation" and row["candidate"] == "history-1l"
            and row["landmark_months"] == 24 and row["horizon_months"] == 60 and row["mri_count_stratum"] == "all")
    write_json(output / "paired_validation.json", comparisons)
    history_epochs = {row["epoch"]: row for row in read_json(source / "history-1l" / "epochs.json")}
    trajectories = [{"epoch": row["epoch"], "b_nll": row["validation_nll"],
                     "history_nll": history_epochs[row["epoch"]]["validation_nll"]}
                    for row in read_json(output / "b-continuation" / "epochs.json") if row["epoch"] in history_epochs]
    report = {"exploratory": True, "production_promoted": False, "selection_split": "validation",
        "best_epoch": summary["best_epoch"], "epoch_zero_nll": summary["epoch_zero_nll"],
        "continuation_nll": summary["validation_nll"], "history_nll": history_report["validation_nll"],
        "history_minus_continuation_nll": history_report["validation_nll"] - summary["validation_nll"],
        "continuation_minus_initial_b_nll": summary["validation_nll"] - summary["epoch_zero_nll"],
        "stable": stable_run(output / "b-continuation"), "matched_epoch_trajectory": trajectories,
        "primary_comparison": [r for r in comparisons if r["primary"]],
        "interpretation": "Negative history-minus-continuation NLL favors LTSA; single-seed exploratory evidence.",
        "artifacts": {str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output / "report.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--history-run", type=run_name, default="exploratory_history_1l_seed_42")
    parser.add_argument("--run-name", type=run_name, default="exploratory_b_continuation_seed_42")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    records, schedule, _ = load_snapshot(protocol)
    output = run_control(args.workspace, records, schedule, protocol, args.protocol, args.device,
                         args.debug_run, args.history_run, args.run_name)
    report = read_json(output / "report.json")
    print(json.dumps({"report": str(output / "report.json"),
                      **{k: v for k, v in report.items() if k != "artifacts"}}, indent=2))
    return output


if __name__ == "__main__":
    main()
