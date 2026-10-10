"""Explicit validation-only LTSA amendment; never promotes production baselines."""

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, replace
from pathlib import Path

import torch

from mstat_project.ml.continuous_survival import ContinuousMRIModel, initial_hazard
from mstat_project.ml.ltsa_architecture_ablation import AblationModel, EXPERIMENTS, load_saved_weights
from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot, stable_run, validation_support
from mstat_project.ml.survival import (
    SurvivalConfig, _loader, _save_checkpoint, _seed, model_identity, predict,
    train_variant, transfer_baseline, transfer_prediction_check, validation_loss, write_json,
)
from mstat_project.ml.survival_comparison import compare_experiments
from mstat_project.ml.survival_data import at_landmark
from mstat_project.ml.survival_embeddings import CachedEncoder, fingerprint, load_embeddings
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, cohort_fingerprint, file_sha256, implementation_hashes, read_json, require_debug_gate,
)


def run_name(value):
    if not re.fullmatch(r"exploratory_[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Use a single directory name beginning exploratory_")
    return value


def check_sources(workspace, records, protocol, protocol_path, device, debug_run):
    """Waive only historical NLL promotion; retain all empirical/provenance checks."""
    cache = workspace / "embeddings.pt"
    manifest = load_embeddings(cache, provenance=protocol["snapshot_hashes"],
                               spatial_size=protocol["reference_config"]["spatial_size"])[0]["manifest"]
    debug_path = workspace / debug_run_name(debug_run) / "gate.json"
    require_debug_gate(debug_path, protocol, manifest, records=records)
    source = workspace / "seed_42"
    gate_path = source / "baseline_gate.json"
    gate = read_json(gate_path)
    code = implementation_hashes()
    if gate.get("implementation_hashes") != code or gate.get("protocol_sha256") != fingerprint(protocol):
        raise ValueError("Baseline gate code or protocol differs")
    sources, b_config = {}, None
    for arm in ("a", "b"):
        directory = source / arm
        path = directory / "best.pt"
        result = gate["arms"][arm]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        summary = read_json(directory / "summary.json")
        value = result["validation_nll"]
        if (Path(result["checkpoint"]).resolve() != path.resolve()
            or file_sha256(path) != result["checkpoint_sha256"]
            or checkpoint.get("variant") != arm or checkpoint.get("implementation_hashes") != code
            or checkpoint.get("identity") != gate["identity"] or summary.get("identity") != gate["identity"]
            or gate["identity"]["cohort_sha256"] != cohort_fingerprint(records)
            or gate["identity"]["encoder_sha256"] != manifest["encoder_sha256"]
            or checkpoint.get("best_validation_nll") != value or summary.get("validation_nll") != value
            or not math.isfinite(value)):
            raise ValueError(f"Baseline {arm} checkpoint/summary provenance mismatch")
        if not result.get("stable") or not stable_run(directory):
            raise ValueError(f"Exploratory initialization still requires stable baseline {arm}")
        config = SurvivalConfig(**checkpoint["config"])
        if (config.seed != 42 or not config.freeze_encoder or not config.cache_path or config.baseline_init
            or config.auxiliary_weight or config.fine_tune_final_block or config.evaluate_test):
            raise ValueError("Exploratory LTSA requires the original seed-42 frozen cached baselines")
        for key in ("epochs", "patience", "batch_size", "learning_rate", "weight_decay",
                    "gradient_clip_norm", "quadrature_points"):
            if getattr(config, key) != protocol["reference_config"][key]:
                raise ValueError(f"Baseline configuration differs from locked setting {key}")
        sources[arm] = {"checkpoint": str(path.resolve()), "checkpoint_sha256": result["checkpoint_sha256"],
            "validation_nll": value, "historical_threshold": protocol["reference_baselines"][arm]["validation_nll"],
            "meets_historical_threshold": value <= protocol["reference_baselines"][arm]["validation_nll"]
                                           + protocol["baseline_nll_tolerance"],
            "stable": True}
        if arm == "b":
            b_config = config
    # Gate enforcement is delegated to this entry point. The low-level trainer's
    # debug_gate=None interface avoids invoking its *production promotion* policy.
    # This is not exposed as a user flag, and the passed gate remains pinned below.
    config = replace(b_config, variants=("history",), experiment_id="exploratory-history-1l", device=device,
        cache_path=cache, baseline_init=source / "b" / "best.pt", debug_gate=None, baseline_gate=None,
        protocol_path=protocol_path, diagnostics=True, n_layers=1, d_model=128, dropout=.1,
        auxiliary_weight=0., fine_tune_final_block=False, evaluate_test=False,
        horizons=tuple(protocol["reference_config"]["horizons"]))
    config.validate()
    return config, manifest, gate, sources, debug_path, gate_path


def validation_predictions(model, records, schedule, config):
    predictions = []
    for landmark in sorted({24., *(float(t) for t in schedule["follow_up_months"])}):
        rows = at_landmark(records, landmark, "validation")
        if rows:
            predictions.extend(predict(model, _loader(rows, config, model.variant, schedule.height),
                                       config.device, config.horizons))
    return predictions


def run_exploratory(workspace, records, schedule, protocol, protocol_path, device, debug_run, name):
    workspace = Path(workspace)
    if read_json(workspace / "protocol.json") != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    config, manifest, gate, sources, debug_path, gate_path = check_sources(
        workspace, records, protocol, protocol_path, device, debug_run)
    output = workspace / "investigations" / run_name(name)
    config = replace(config, output_dir=output)
    reference = protocol["reference_baselines"]["b"]
    if file_sha256(reference["checkpoint"]) != reference["checkpoint_sha256"]:
        raise ValueError("Historical B checkpoint changed")
    support = validation_support(records)
    if 24. not in [float(t) for t in schedule["follow_up_months"]] or 60. not in config.horizons:
        raise ValueError("Exploratory comparison requires the month-24 / 60-month primary endpoint")
    if not support["patients"] or not support["comparable_pairs"]:
        raise ValueError("Exploratory primary validation endpoint is not estimable")
    specification = {
        "amendment": "exploratory-ltsa-from-stable-unpromoted-b-v1",
        "waived_requirement": "A/B historical validation NLL promotion thresholds only",
        "purpose": "Does history improve the stabilized B model on validation?",
        "selection_split": "validation", "production_promoted": False,
        "source_baseline_gate_passed": gate["passed"], "sources": sources,
        "reference_b": reference, "primary_validation_support": support,
        "protocol_sha256": fingerprint(protocol), "cohort_sha256": cohort_fingerprint(records),
        "debug_gate": str(debug_path.resolve()), "debug_gate_sha256": file_sha256(debug_path),
        "baseline_gate_sha256": file_sha256(gate_path), "cache_id": manifest["cache_id"],
        "implementation_hashes": implementation_hashes(),
        "runner_sha256": file_sha256(Path(__file__)),
        "reference_adapter_sha256": file_sha256(Path(__file__).with_name("ltsa_architecture_ablation.py")),
        "config": json.loads(json.dumps(asdict(config), default=str)),
    }
    spec_path = output / "specification.json"
    if output.exists():
        if not spec_path.exists() or read_json(spec_path) != specification:
            raise ValueError("Exploratory provenance changed; preserve this run and choose a new --run-name")
        if (output / "report.json").exists():
            report = read_json(output / "report.json")
            if any(file_sha256(output / path) != digest for path, digest in report["artifacts"].items()):
                raise ValueError("Completed exploratory artifacts changed")
            return output
        raise ValueError("Incomplete exploratory run; preserve it and choose a new --run-name")
    output.mkdir(parents=True)
    write_json(spec_path, specification)
    _seed(config.seed)
    model = ContinuousMRIModel(CachedEncoder(manifest), "history", freeze_encoder=True,
        baseline_hazard=initial_hazard(records), quadrature_points=config.quadrature_points,
        n_layers=1, d_model=128, dropout=.1).to(device)
    identity = model_identity(model, records, config)
    checkpoint = torch.load(config.baseline_init, map_location="cpu", weights_only=False)
    transfer_baseline(model, checkpoint, identity)
    baseline = ContinuousMRIModel(CachedEncoder(manifest), "b", freeze_encoder=True,
        quadrature_points=config.quadrature_points).to(device)
    baseline.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    val_records = [r for r in records if r.split == "validation"]
    val = _loader(val_records, config, "history", schedule.height)
    # Check all validation observations, including equality to the actual B model.
    check = transfer_prediction_check(model, val, device, config.horizons)
    alpha = model.alpha.detach().clone()
    with torch.no_grad():
        model.alpha.zero_()
    zero_predictions = validation_predictions(model, records, schedule, config)
    b_predictions = validation_predictions(baseline, records, schedule, config)
    b_nll = validation_loss(baseline, _loader(val_records, config, "b", schedule.height), device)
    with torch.no_grad():
        model.alpha.copy_(alpha)
    max_gap = max(abs(a - b) for left, right in zip(zero_predictions, b_predictions, strict=True)
                  for a, b in zip(left["survival"], right["survival"], strict=True))
    check.update(alpha_zero_max_survival_difference=max_gap, baseline_validation_nll=b_nll)
    write_json(output / "initialization_check.json", check)
    if not check["passed"] or max_gap > 1e-6 or abs(b_nll - sources["b"]["validation_nll"]) > 1e-5:
        raise ValueError("Exploratory baseline reproduction or initialization check failed")
    del baseline
    epoch_zero_nll = validation_loss(model, val, device)
    _save_checkpoint(output / "epoch_zero.pt", {"model": model.state_dict(), "identity": identity,
        "variant": "history", "config": asdict(config), "epoch": 0,
        "best_validation_nll": epoch_zero_nll, "exploratory": True, "production_promoted": False,
        "specification_sha256": file_sha256(spec_path)})
    write_json(output / "epoch_zero_predictions.json", validation_predictions(model, records, schedule, config))
    write_json(output / "b_validation_predictions.json", b_predictions)
    del model
    # Same training implementation, fresh optimizer/scheduler, loss, sampling,
    # diagnostics and epoch-zero checkpoint selection as the production route.
    summary = train_variant(config, records, schedule, "history", output / "history-1l")
    summary.update(exploratory=True, production_promoted=False, specification_sha256=file_sha256(spec_path))
    write_json(output / "history-1l" / "summary.json", summary)
    for filename in ("best.pt", "last.pt"):
        path = output / "history-1l" / filename
        state = torch.load(path, map_location="cpu", weights_only=False)
        state.update(exploratory=True, production_promoted=False, specification_sha256=file_sha256(spec_path))
        _save_checkpoint(path, state)
        del state
    epochs = read_json(output / "history-1l" / "epochs.json")
    if abs(epochs[0]["validation_nll"] - epoch_zero_nll) > 1e-5:
        raise ValueError("Trainer epoch zero differs from preserved initialization")
    history_predictions = read_json(output / "history-1l" / "validation_predictions.json")
    legacy = AblationModel(manifest, "b", EXPERIMENTS["legacy_all"], initial_hazard(records),
                           config.quadrature_points).to(device)
    checkpoint = torch.load(reference["checkpoint"], map_location="cpu", weights_only=False)
    load_saved_weights(legacy, checkpoint, "reference", manifest)
    del checkpoint
    legacy_nll = validation_loss(legacy, _loader(val_records, config, "b", schedule.height), device)
    legacy_predictions = validation_predictions(legacy, records, schedule, config)
    del legacy
    write_json(output / "legacy_b_validation_predictions.json", legacy_predictions)
    comparisons = compare_experiments({"b": b_predictions, "history-1l": history_predictions,
        "legacy-b": legacy_predictions}, config.horizons, bootstrap_samples=protocol["bootstrap_samples"],
        seed=protocol["bootstrap_seed"], expected_split="validation")
    write_json(output / "paired_validation.json", comparisons)
    report = {"exploratory": True, "production_promoted": False, "selection_split": "validation",
        "source_baseline_gate_passed": gate["passed"], "best_epoch": summary["best_epoch"],
        "epoch_zero_nll": epoch_zero_nll, "validation_nll": summary["validation_nll"],
        "stabilized_b_nll": b_nll, "difference_from_stabilized_b": summary["validation_nll"] - b_nll,
        "historical_b_saved_nll": reference["validation_nll"], "historical_b_cached_nll": legacy_nll,
        "difference_from_historical_b_cached": summary["validation_nll"] - legacy_nll,
        "stable": stable_run(output / "history-1l"),
        "primary_comparison": [r for r in comparisons if r["primary"]],
        "interpretation": "Improvement over stabilized B alone does not establish improvement over historical B.",
        "artifacts": {str(path.relative_to(output)): file_sha256(path)
                      for path in output.rglob("*") if path.is_file()}}
    write_json(output / "report.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--run-name", type=run_name, default="exploratory_history_1l_seed_42")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    records, schedule, _ = load_snapshot(protocol)
    output = run_exploratory(args.workspace, records, schedule, protocol, args.protocol,
                              args.device, args.debug_run, args.run_name)
    report = read_json(output / "report.json")
    print(json.dumps({"report": str(output / "report.json"),
                      **{k: v for k, v in report.items() if k != "artifacts"}}, indent=2))
    return output


if __name__ == "__main__":
    main()
