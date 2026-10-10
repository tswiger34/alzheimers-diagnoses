"""Continue debug Baseline B with fresh AdamW to diagnose the LTSA overfit failure.

This training-only control uses the saved fixed panel and does not change or
replace a production gate. Run only after reviewing its automated tests.
"""

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from mstat_project.ml.continuous_survival import ContinuousMRIModel
from mstat_project.ml.ltsa_debug_inspection import inspect_debug_run
from mstat_project.ml.ltsa_experiments import debug_metrics, debug_run_name, load_snapshot, select_debug_patients
from mstat_project.ml.survival import SurvivalConfig, _loader, _save_checkpoint, _seed, train_epoch, write_json
from mstat_project.ml.survival_embeddings import CachedEncoder, load_embeddings
from mstat_project.ml.survival_protocol import PROTOCOL_PATH, file_sha256, read_json


def run_b_continuation(workspace, records, protocol, device, *, debug_run, max_updates=None):
    """Follow B for the entire matched update budget, retaining diagnostic weights."""
    budget = protocol["debug"]["max_updates"] if max_updates is None else max_updates
    if not 1 <= budget <= protocol["debug"]["max_updates"]:
        raise ValueError("Invalid continuation update budget")
    source = Path(workspace) / debug_run_name(debug_run)
    # Reuse the inspector's protocol, fixed-panel, cache and checkpoint checks.
    evidence = inspect_debug_run(workspace, records, protocol, device, debug_run=debug_run)
    if not evidence["implementation_matches_training"]:
        raise ValueError("Matched continuation requires the same training implementation as the source run")
    if not evidence["arms"].get("b", {}).get("saved_gate", {}).get("passed"):
        raise ValueError("Continuation requires a passed debug B checkpoint")
    if "metrics" not in evidence["arms"].get("history", {}):
        raise ValueError("Continuation requires a saved history arm for comparison")
    gate = read_json(source / "gate.json")
    if gate.get("c_index_risk_score") != "cumulative_hazard_60_months":
        raise ValueError("Use a source debug run with cumulative-hazard ranking")
    checkpoint = torch.load(source / "b.pt", map_location="cpu", weights_only=False)
    config = SurvivalConfig(**checkpoint["config"])
    config = replace(config, device=device, cache_path=Path(workspace) / "debug_embeddings.pt",
                     spatial_size=tuple(config.spatial_size), horizons=tuple(config.horizons))
    if (config.learning_rate != 1e-3 or config.weight_decay != 0 or config.dropout != 0
        or config.batch_size != 16 or config.auxiliary_weight != 0 or config.gradient_clip_norm != 5):
        raise ValueError("Source configuration does not match the fixed debug optimizer and sampling settings")
    subset = select_debug_patients(records, config.seed)
    manifest = load_embeddings(config.cache_path)[0]["manifest"]
    _seed(config.seed)
    model = ContinuousMRIModel(CachedEncoder(manifest), "b", freeze_encoder=True,
                               quadrature_points=config.quadrature_points, dropout=0.).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = _loader(subset, config, "b", 1)
    initial = debug_metrics(model, loader, device)
    # Confirm the control actually starts at the saved B predictions/loss.
    saved = gate["arms"]["b"]["final"]
    if any(abs(initial[key] - saved[key]) > 1e-5 for key in ("nll", "c_index", "risk_variance")):
        raise ValueError("Continuation initialization does not reproduce the saved B metrics")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    history_trace = read_json(source / "history_trace.json")
    history_by_update = {row["update"]: row for row in history_trace}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = source / f"b_continuation_{stamp}"
    output.mkdir(exist_ok=False)
    report = {
        "diagnostic_only": True, "status": "running", "source_debug_run": debug_run,
        "source_gate_sha256": evidence["gate_sha256"],
        "source_b_sha256": evidence["arms"]["b"]["checkpoint_sha256"],
        "source_history_sha256": evidence["arms"]["history"]["checkpoint_sha256"],
        "source_history_trace_sha256": file_sha256(source / "history_trace.json"),
        "protocol_sha256": evidence["protocol_sha256"], "cache_id": evidence["cache_id"],
        "implementation_hashes": evidence["inspection_implementation_hashes"],
        "control_sha256": file_sha256(Path(__file__)),
        "inspector_sha256": evidence["inspector_sha256"],
        "config": asdict(config), "update_budget": budget, "initial": initial,
        "required_nll": initial["nll"] - protocol["debug"]["minimum_nll_drop"],
        "criteria_met_at": None, "best_rank": None, "best_nll": None,
        "source_history_gate": gate["arms"]["history"],
    }
    write_json(output / "report.json", report)
    print(f"B continuation: {output}; initial NLL {initial['nll']:.5f}, C(H) {initial['c_index']:.4f}", flush=True)
    trace = []

    def save(name, update, metrics):
        _save_checkpoint(output / name, {"model": model.state_dict(), "variant": "b",
            "identity": checkpoint["identity"], "config": asdict(config), "update": update,
            "metrics": metrics, "diagnostic_only": True, "initialization_sha256": report["source_b_sha256"]})

    try:
        for update in range(1, budget + 1):
            diagnostic = [] if update == 1 or update % 25 == 0 or update == budget else None
            train_epoch(model, loader, optimizer, device, config.gradient_clip_norm,
                        diagnostics=diagnostic, anomaly=config.anomaly_detection)
            if diagnostic is None:
                continue
            metrics = debug_metrics(model, loader, device)
            rank_ok = metrics["c_index"] >= protocol["debug"]["minimum_c_index"]
            meets = (metrics["nll"] <= report["required_nll"] and rank_ok and metrics["risk_variance"] > 1e-8)
            if meets and report["criteria_met_at"] is None:
                report["criteria_met_at"] = update
                save("first_joint_criteria.pt", update, metrics)
            for name, eligible in (("best_nll", True), ("best_rank", rank_ok)):
                if eligible and (report[name] is None or metrics["nll"] < report[name]["nll"]):
                    report[name] = {"update": update, **metrics}
                    save(f"{name}.pt", update, metrics)
            comparison = history_by_update.get(update)
            row = {"update": update, **metrics, "optimization": diagnostic,
                   "history": ({key: comparison[key] for key in ("nll", "c_index", "probability_c_index")}
                               if comparison is not None else None)}
            trace.append(row)
            report["final"] = {"update": update, **metrics}
            write_json(output / "trace.json", trace)
            write_json(output / "report.json", report)
            if update == 1 or update % 100 == 0 or update == budget:
                print(f"B continuation: update {update}, NLL {metrics['nll']:.5f}, "
                      f"C(H) {metrics['c_index']:.4f}, C(prob) {metrics['probability_c_index']:.4f}", flush=True)
        save("last.pt", budget, metrics)
        report["status"] = "completed"
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "report.json", report)
        raise
    write_json(output / "report.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    if read_json(args.workspace / "protocol.json") != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    records, _, _ = load_snapshot(protocol, args.tensor_dir)
    output = run_b_continuation(args.workspace, records, protocol, args.device, debug_run=args.debug_run)
    report = read_json(output / "report.json")
    print(json.dumps({"report": str(output / "report.json"), **{key: report[key] for key in
        ("diagnostic_only", "status", "required_nll", "criteria_met_at", "best_rank", "best_nll", "final")}}, indent=2))
    return output


if __name__ == "__main__":
    main()
