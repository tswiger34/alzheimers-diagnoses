"""Validation-only patience-8 A/B investigation; never promotes production models."""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch

from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot, stable_run, train_if_missing
from mstat_project.ml.survival import SurvivalConfig, write_json
from mstat_project.ml.survival_embeddings import fingerprint, load_embeddings
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, cohort_fingerprint, file_sha256, implementation_hashes, read_json, require_debug_gate,
)


def run_investigation(workspace, records, schedule, protocol, protocol_path, device, debug_run):
    workspace = Path(workspace)
    source = workspace / "seed_42"
    output = workspace / "investigations" / "patience_8_seed_42"
    cache = workspace / "embeddings.pt"
    manifest = load_embeddings(cache, provenance=protocol["snapshot_hashes"],
                               spatial_size=protocol["reference_config"]["spatial_size"])[0]["manifest"]
    debug_gate = workspace / debug_run_name(debug_run) / "gate.json"
    require_debug_gate(debug_gate, protocol, manifest, records=records)
    original_gate = read_json(source / "baseline_gate.json")
    current_code = implementation_hashes()
    if (original_gate.get("implementation_hashes") != current_code
        or original_gate.get("protocol_sha256") != fingerprint(protocol)):
        raise ValueError("Source baseline experiment has incompatible code or protocol")
    configs, sources = {}, {}
    for arm in ("a", "b"):
        checkpoint_path = source / arm / "best.pt"
        digest = file_sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (digest != original_gate["arms"][arm]["checkpoint_sha256"]
            or checkpoint.get("variant") != arm or checkpoint.get("implementation_hashes") != current_code
            or checkpoint.get("identity") != original_gate["identity"]
            or checkpoint["identity"]["cohort_sha256"] != cohort_fingerprint(records)
            or checkpoint["identity"]["encoder_sha256"] != manifest["encoder_sha256"]):
            raise ValueError(f"Source {arm} checkpoint provenance differs from the baseline experiment")
        config = SurvivalConfig(**checkpoint["config"])
        if config.patience != 4 or config.epochs != 30 or config.seed != 42 or config.baseline_init:
            raise ValueError("This investigation requires the original seed-42, 30-epoch, patience-4 baselines")
        # Reinitialize from the same seed, not from the trained source weights.
        # Only patience changes scientifically; paths/device select local resources.
        configs[arm] = replace(config, patience=8, device=device, cache_path=cache,
            debug_gate=debug_gate, protocol_path=protocol_path, output_dir=output)
        configs[arm].validate()
        sources[arm] = {"checkpoint_sha256": digest,
                        "validation_nll": original_gate["arms"][arm]["validation_nll"]}
    specification = {
        "investigation": "early_stopping_patience_8", "selection_split": "validation",
        "change": {"patience": {"control": 4, "candidate": 8}}, "source_baselines": sources,
        "protocol_sha256": fingerprint(protocol), "debug_gate_sha256": file_sha256(debug_gate),
        "source_baseline_gate_sha256": file_sha256(source / "baseline_gate.json"),
        "cache_id": manifest["cache_id"], "implementation_hashes": current_code,
        "investigation_sha256": file_sha256(Path(__file__)), "production_promoted": False,
    }
    specification_path = output / "specification.json"
    if output.exists():
        if not specification_path.exists() or read_json(specification_path) != specification:
            raise ValueError("Existing investigation has incompatible provenance; preserve it separately")
    else:
        output.mkdir(parents=True)
        write_json(specification_path, specification)
    report = {"production_promoted": False, "selection_split": "validation", "arms": {}}
    for arm in ("a", "b"):
        summary = train_if_missing(configs[arm], records, schedule, arm, output / arm)
        reference = protocol["reference_baselines"][arm]["validation_nll"]
        report["arms"][arm] = {
            "best_epoch": summary["best_epoch"], "validation_nll": summary["validation_nll"],
            "patience_4_validation_nll": sources[arm]["validation_nll"],
            "difference_from_patience_4": summary["validation_nll"] - sources[arm]["validation_nll"],
            "reference_nll": reference,
            "meets_reference_nll": summary["validation_nll"] <= reference + protocol["baseline_nll_tolerance"],
            "stable": stable_run(output / arm), "checkpoint_sha256": file_sha256(output / arm / "best.pt"),
        }
        write_json(output / "report.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    if read_json(args.workspace / "protocol.json") != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    records, schedule, _ = load_snapshot(protocol)
    output = run_investigation(args.workspace, records, schedule, protocol, args.protocol,
                               args.device, args.debug_run)
    print(json.dumps({"report": str(output / "report.json"), **read_json(output / "report.json")}, indent=2))
    return output


if __name__ == "__main__":
    main()
