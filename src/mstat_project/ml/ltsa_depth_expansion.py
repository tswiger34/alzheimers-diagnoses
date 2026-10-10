"""Exploratory two-layer LTSA, paired with retained one-layer and continued B."""

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
import torch

from mstat_project.ml.continuous_survival import ContinuousMRIModel, initial_hazard
from mstat_project.ml.ltsa_b_continuation import matching_samples, verify_history
from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot, stable_run
from mstat_project.ml.ltsa_exploratory import check_sources, run_name
from mstat_project.ml.survival import (
    _loader, _save_checkpoint, _seed, model_identity, train_variant, transfer_baseline,
    transfer_prediction_check, validation_loss, write_json,
)
from mstat_project.ml.survival_comparison import compare_experiments
from mstat_project.ml.survival_protocol import PROTOCOL_PATH, file_sha256, fingerprint, implementation_hashes, read_json
from mstat_project.ml.survival_embeddings import CachedEncoder


def verify_continuation(directory, history, protocol, manifest, sources):
    spec, report = read_json(directory / "specification.json"), read_json(directory / "report.json")
    if (spec.get("history_report_sha256") != file_sha256(history / "report.json")
        or spec.get("history_specification_sha256") != file_sha256(history / "specification.json")
        or spec.get("protocol_sha256") != fingerprint(protocol) or spec.get("cache_id") != manifest["cache_id"]
        or spec.get("sources") != sources or spec.get("implementation_hashes") != implementation_hashes()
        or spec.get("runner_sha256") != file_sha256(Path(__file__).with_name("ltsa_b_continuation.py"))
        or not report.get("exploratory") or not report.get("stable")
        or report.get("selection_split") != "validation" or not stable_run(directory / "b-continuation")):
        raise ValueError("Depth expansion requires the matching completed stable B continuation control")
    artifacts = report.get("artifacts", {})
    for required in ("specification.json", "b-continuation/best.pt", "b-continuation/validation_predictions.json"):
        if str(Path(required)) not in artifacts:
            raise ValueError("Continuation report lacks artifact fingerprints")
    for relative, digest in artifacts.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
            raise ValueError("Continuation artifact changed")
    return report


def run_depth(workspace, records, schedule, protocol, protocol_path, device, debug_run, history_run, control_run, name):
    workspace = Path(workspace)
    if read_json(workspace / "protocol.json") != protocol:
        raise ValueError("Workspace protocol differs")
    config, manifest, gate, sources, debug_path, gate_path = check_sources(
        workspace, records, protocol, protocol_path, device, debug_run)
    history = workspace / "investigations" / run_name(history_run)
    continuation = workspace / "investigations" / run_name(control_run)
    _, history_report = verify_history(history, protocol, config, manifest, sources, debug_path)
    control_report = verify_continuation(continuation, history, protocol, manifest, sources)
    output = workspace / "investigations" / run_name(name)
    config = replace(config, n_layers=2, output_dir=output, experiment_id="exploratory-history-2l")
    config.validate()
    specification = {"experiment": "exploratory-depth-2", "selection_split": "validation",
        "production_promoted": False, "change": {"n_layers": {"control": 1, "candidate": 2}},
        "sources": sources, "source_baseline_gate_passed": gate["passed"],
        "protocol_sha256": fingerprint(protocol), "cache_id": manifest["cache_id"],
        "debug_gate_sha256": file_sha256(debug_path), "baseline_gate_sha256": file_sha256(gate_path),
        "history_report_sha256": file_sha256(history / "report.json"),
        "continuation_report_sha256": file_sha256(continuation / "report.json"),
        "implementation_hashes": implementation_hashes(),
        "runner_hashes": {name: file_sha256(Path(__file__).with_name(name)) for name in
                          ("ltsa_depth_expansion.py", "ltsa_exploratory.py", "ltsa_b_continuation.py")},
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "interpretation": "Seed-42 depth screening only; no cross-seed promotion or held-out evaluation."}
    if output.exists():
        if not (output / "specification.json").exists() or read_json(output / "specification.json") != specification:
            raise ValueError("Depth experiment provenance changed; choose a new --run-name")
        if not (output / "report.json").exists():
            raise ValueError("Incomplete depth experiment; preserve it and choose a new --run-name")
        report = read_json(output / "report.json")
        if any(file_sha256(output / p) != sha for p, sha in report["artifacts"].items()):
            raise ValueError("Completed depth experiment artifacts changed")
        return output
    output.mkdir(parents=True)
    spec_path = output / "specification.json"
    write_json(spec_path, specification)
    _seed(config.seed)
    model = ContinuousMRIModel(CachedEncoder(manifest), "history", freeze_encoder=True,
        baseline_hazard=initial_hazard(records), quadrature_points=config.quadrature_points,
        n_layers=2, d_model=config.d_model, dropout=config.dropout).to(device)
    identity = model_identity(model, records, config)
    checkpoint = torch.load(config.baseline_init, map_location="cpu", weights_only=False)
    transfer_baseline(model, checkpoint, identity)
    del checkpoint
    val = _loader([r for r in records if r.split == "validation"], config, "history", schedule.height)
    check = transfer_prediction_check(model, val, device, config.horizons)
    write_json(output / "initialization_check.json", check)
    if not check["passed"]:
        raise ValueError("Two-layer initialization exceeded the survival-probability tolerance")
    epoch_zero = validation_loss(model, val, device)
    _save_checkpoint(output / "epoch_zero.pt", {"model": model.state_dict(), "config": asdict(config),
        "identity": identity, "variant": "history", "epoch": 0, "best_validation_nll": epoch_zero,
        "initialization_sha256": file_sha256(config.baseline_init), "exploratory": True,
        "production_promoted": False, "specification_sha256": file_sha256(spec_path)})
    del model
    candidate = output / "history-2l"
    summary = train_variant(config, records, schedule, "history", candidate)
    summary.update(exploratory=True, production_promoted=False, specification_sha256=file_sha256(spec_path))
    write_json(candidate / "summary.json", summary)
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(candidate / name, map_location="cpu", weights_only=False)
        checkpoint.update(exploratory=True, production_promoted=False, specification_sha256=file_sha256(spec_path))
        _save_checkpoint(candidate / name, checkpoint)
        del checkpoint
    epochs = read_json(candidate / "epochs.json")
    if abs(epochs[0]["validation_nll"] - epoch_zero) > 1e-5:
        raise ValueError("Trainer epoch zero differs from preserved two-layer initialization")
    matched = []
    for path in candidate.glob("sampled_epoch_*.csv"):
        parent_path = history / "history-1l" / path.name
        if parent_path.exists():
            frame = pl.read_csv(path, schema_overrides={"ptid": pl.String, "image_id": pl.String,
                                                       "acquisition_months": pl.Float64})
            if not matching_samples(frame, parent_path):
                raise ValueError("Depth experiment sampling differs from the one-layer control")
            matched.append(path.name)
    predictions = {"b-continuation": read_json(continuation / "b-continuation" / "validation_predictions.json"),
        "history-1l": read_json(history / "history-1l" / "validation_predictions.json"),
        "history-2l": read_json(candidate / "validation_predictions.json")}
    comparisons = compare_experiments(predictions, config.horizons, bootstrap_samples=protocol["bootstrap_samples"],
                                      seed=protocol["bootstrap_seed"], expected_split="validation")
    for row in comparisons:
        row["primary"] = (row["reference"] == "history-1l" and row["candidate"] == "history-2l"
            and row["landmark_months"] == 24 and row["horizon_months"] == 60 and row["mri_count_stratum"] == "all")
    primary = [r for r in comparisons if r["primary"]]
    write_json(output / "paired_validation.json", comparisons)
    stable = stable_run(candidate)
    nll_delta = summary["validation_nll"] - history_report["validation_nll"]
    favorable = (stable and nll_delta < 0 and len(primary) == 1
                 and primary[0]["difference"] is not None and primary[0]["difference"] >= 0)
    report = {"exploratory": True, "production_promoted": False, "selection_split": "validation", "seed": config.seed,
        "best_epoch": summary["best_epoch"], "epoch_zero_nll": epoch_zero, "validation_nll": summary["validation_nll"],
        "one_layer_nll": history_report["validation_nll"], "difference_from_one_layer": nll_delta,
        "b_continuation_nll": control_report["continuation_nll"],
        "difference_from_b_continuation": summary["validation_nll"] - control_report["continuation_nll"],
        "stable": stable, "primary_comparison": primary, "matched_sample_files": sorted(matched),
        "parameter_counts": read_json(candidate / "parameter_counts.json"),
        "screening_decision": "candidate_for_multiseed_validation" if favorable else "no_depth_benefit_established",
        "interpretation": "Depth promotion requires matched seeds 42/43/44; this run is exploratory screening only.",
        "artifacts": {str(p.relative_to(output)): file_sha256(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output / "report.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--history-run", type=run_name, default="exploratory_history_1l_seed_42")
    parser.add_argument("--control-run", type=run_name, default="exploratory_b_continuation_seed_42_retry1")
    parser.add_argument("--run-name", type=run_name, default="exploratory_history_2l_seed_42")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    records, schedule, _ = load_snapshot(protocol)
    output = run_depth(args.workspace, records, schedule, protocol, args.protocol, args.device, args.debug_run,
                       args.history_run, args.control_run, args.run_name)
    report = read_json(output / "report.json")
    print(json.dumps({"report": str(output / "report.json"),
                      **{k: v for k, v in report.items() if k != "artifacts"}}, indent=2))
    return output


if __name__ == "__main__":
    main()
