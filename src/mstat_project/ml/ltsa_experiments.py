"""Gated LTSA experiments on the immutable reference snapshot.

Run ``python -m mstat_project.ml.ltsa_experiments --help`` for stage commands.
No stage evaluates held-out test predictions except the explicit report stage.
"""

import argparse
import hashlib
import json
import logging
import math
import re
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
import torch
from torchvision.models import ResNet101_Weights

from mstat_project.data_prep.data_splitting import frame_fingerprint
from mstat_project.ml.continuous_survival import ContinuousMRIModel, initial_hazard
from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.ml.survival import (
    SurvivalConfig,
    _device_batch,
    _loader,
    _save_checkpoint,
    _seed,
    model_identity,
    predict,
    train_epoch,
    train_variant,
    transfer_baseline,
    transfer_prediction_check,
    write_json,
)
from mstat_project.ml.survival_comparison import compare_experiments
from mstat_project.ml.survival_data import at_landmark, build_observations
from mstat_project.ml.survival_embeddings import CachedEncoder, export_embeddings, fingerprint, load_embeddings
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH,
    cohort_fingerprint,
    file_sha256,
    implementation_hashes,
    read_json,
    require_debug_gate,
)
from mstat_project.ml.utils import concordance_index


def read_reference_schedule(path):
    """Restore the NUMERIC(10,2) boundaries declared by core_follow_up_buckets.

    CSV inference otherwise converts Decimal values to Float64 (9.00 -> 9.0),
    changing frame_fingerprint's canonical CSV despite unchanged data. Read the
    boundary tokens as strings first so no precision is lost before validation.
    """
    boundaries = ("follow_up_start", "follow_up_end")
    schedule = pl.read_csv(path, schema_overrides={name: pl.String for name in boundaries})
    for name in boundaries:
        values = schedule[name]
        if values.null_count() or not values.str.contains(r"^[0-9]+\.[0-9]{2}$").all():
            raise ValueError(f"Reference schedule {name} must contain non-null, two-decimal boundaries")
    return schedule.with_columns(pl.col(name).cast(pl.Decimal(precision=10, scale=2)) for name in boundaries)


def load_snapshot(protocol, tensor_dir=None):
    for arm, reference_arm in protocol["reference_baselines"].items():
        if file_sha256(reference_arm["checkpoint"]) != reference_arm["checkpoint_sha256"]:
            raise ValueError(f"Reference {arm} checkpoint fingerprint changed")
    reference = Path(protocol["reference_directory"])
    frame = pl.read_parquet(reference / "source.parquet")
    schedule = read_reference_schedule(reference / "schedule.csv")
    splits = pl.read_csv(reference / "patient_splits.csv")
    hashes = {
        "source": frame_fingerprint(frame),
        "schedule": frame_fingerprint(schedule),
        "splits": frame_fingerprint(splits),
    }
    mismatches = [
        f"{name}: expected {protocol['snapshot_hashes'].get(name)}, got {actual}"
        for name, actual in hashes.items()
        if actual != protocol["snapshot_hashes"].get(name)
    ]
    if mismatches:
        raise ValueError("Reference snapshot fingerprints changed: " + "; ".join(mismatches))
    records, audit = build_observations(
        frame, schedule, splits, Path(tensor_dir or protocol["reference_config"]["tensor_dir"])
    )
    return records, schedule, audit


def select_debug_patients(records, seed=42):
    latest = {}
    for r in records:
        if r.split == "train" and (r.ptid not in latest or r.prediction_months > latest[r.ptid].prediction_months):
            latest[r.ptid] = r
    selected = []
    for event in (True, False):
        candidates = sorted(
            (r for r in latest.values() if r.event == event),
            key=lambda r: hashlib.sha256(f"{seed}:overfit:{r.ptid}".encode()).hexdigest(),
        )
        if len(candidates) < 8:
            raise ValueError("Debug gate requires eight event and eight censored training patients")
        selected.extend(candidates[:8])
    if not any(len(r.image_ids) > 1 for r in selected) or not any(
        any(b - a > 1 for a, b in zip(r.visit_nums, r.visit_nums[1:], strict=False)) for r in selected
    ):
        # Deterministically include a gap-bearing history without changing event balance.
        candidate = next(
            (
                r
                for r in sorted(latest.values(), key=lambda r: r.ptid)
                if any(b - a > 1 for a, b in zip(r.visit_nums, r.visit_nums[1:], strict=False))
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Debug cohort has no history with missing nominal visits")
        selected[next(i for i, r in enumerate(selected) if r.event == candidate.event)] = candidate
    return sorted(selected, key=lambda r: r.ptid)


def validation_support(records):
    rows = at_landmark(records, 24.0, "validation")
    events = sum(r.event and r.outcome_months <= 84 for r in rows)
    pairs = sum(
        a.event and a.outcome_months <= 84 and a.outcome_months < min(b.outcome_months, 84)
        for a in rows
        for b in rows
    )
    return {
        "patients": len(rows),
        "events_within_horizon": events,
        "comparable_pairs": pairs,
        "confirmatory_support": len(rows) >= 30 and events >= 5 and pairs > 0,
    }


@torch.no_grad()
def debug_metrics(model, loader, device):
    """Rank by 60-month cumulative hazard to avoid rounded probability ties."""
    model.eval()
    losses, risks, times, events = [], [], [], []
    for batch in loader:
        batch = _device_batch(batch, device)
        z = model.encode_batch(batch)
        losses.append(model.head.nll(z, batch["entry"], batch["outcome"], batch["event"]))
        risks.append(model.head.cumulative_hazard(z, batch["entry"], batch["entry"] + 60))
        times.append(batch["outcome"] - batch["entry"])
        events.append(batch["event"])
    risk = torch.cat(risks)
    if not torch.isfinite(risk).all():
        raise FloatingPointError("Debug cumulative hazards are nonfinite")
    probability = 1 - torch.exp(-risk)
    result = {
        "nll": float(torch.cat(losses).mean()),
        "c_index": concordance_index(risk, torch.cat(times), torch.cat(events)),
        "risk_variance": float(probability.var(unbiased=False)),
        "probability_c_index": concordance_index(probability, torch.cat(times), torch.cat(events)),
        "probabilities_equal_one": int((probability == 1).sum()),
        "cumulative_hazard_max": float(risk.max()),
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("Debug metrics are nonfinite or have no comparable pairs")
    return result


def run_debug(config, records, output, protocol, *, max_updates=2000):
    """One full-batch update per step; balanced fixed observations, no test data."""
    if not 1 <= max_updates <= protocol["debug"]["max_updates"]:
        raise ValueError("Invalid debug update budget")
    output.mkdir(parents=True, exist_ok=False)
    subset = select_debug_patients(records, config.seed)
    config = replace(
        config,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        dropout=0.0,
        auxiliary_weight=0.0,
        diagnostics=True,
    )
    cache = load_embeddings(config.cache_path)[0]["manifest"]
    report = {
        "passed": False,
        "protocol_sha256": fingerprint(protocol),
        "c_index_risk_score": "cumulative_hazard_60_months",
        "c_index_followup": "full_followup",
        "risk_variance_score": "probability_60_months",
        "gate_evaluation": {
            "a": "update 1, every 25 updates, final update",
            "b": "update 1, every 25 updates, final update",
            "history": "every update",
        },
        "snapshot_hashes": protocol["snapshot_hashes"],
        "encoder_sha256": cache["encoder_sha256"],
        "implementation_hashes": implementation_hashes(),
        "config": asdict(config),
        "full_cohort_sha256": cohort_fingerprint(records),
        "patients": [r.ptid for r in subset],
        "image_ids": [r.image_ids[-1] for r in subset],
        "arms": {},
    }
    write_json(output / "gate.json", report)
    baseline = None
    for variant in ("a", "b", "history"):
        _seed(config.seed)
        model = ContinuousMRIModel(
            CachedEncoder(cache), variant, freeze_encoder=True, baseline_hazard=initial_hazard(subset), dropout=0.0
        ).to(config.device)
        identity = model_identity(model, subset, config)
        loader = _loader(subset, config, variant, 1)
        check, history, passed, best_rank = None, [], False, None
        try:
            if variant == "history":
                transfer_baseline(model, baseline, identity)
                check = transfer_prediction_check(model, loader, config.device, config.horizons)
            initial = debug_metrics(model, loader, config.device)
            final = initial
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0
            )
            if check and not check["passed"]:
                raise ValueError("Tiny LTSA initialization differs from B by more than .01")
            for update in range(1, max_updates + 1):
                diagnostic = [] if update == 1 or update % 25 == 0 or update == max_updates else None
                train_epoch(
                    model,
                    loader,
                    optimizer,
                    config.device,
                    5.0,
                    diagnostics=diagnostic,
                    anomaly=config.anomaly_detection,
                )
                if diagnostic is not None or variant == "history":
                    # Extra cached-panel evaluations must not advance the CPU
                    # generator used for DataLoader iterator seeds. Dropout is
                    # zero, and eval-mode model inference draws no CUDA randomness.
                    with torch.random.fork_rng(devices=[]) if diagnostic is None else nullcontext():
                        final = debug_metrics(model, loader, config.device)
                    rank_passed = final["c_index"] >= protocol["debug"]["minimum_c_index"]
                    passed = (
                        initial["nll"] - final["nll"] >= protocol["debug"]["minimum_nll_drop"]
                        and rank_passed
                        and final["risk_variance"] > 1e-8
                    )
                    if rank_passed and (best_rank is None or final["nll"] < best_rank["nll"]):
                        best_rank = {"update": update, **final, "checkpoint": f"{variant}_best_rank.pt"}
                        _save_checkpoint(
                            output / best_rank["checkpoint"],
                            {
                                "model": model.state_dict(),
                                "variant": variant,
                                "identity": identity,
                                "config": asdict(config),
                                "update": update,
                                "metrics": final,
                            },
                        )
                    history.append({"update": update, **final, "optimization": diagnostic or []})
                    if diagnostic is not None or passed:
                        write_json(output / f"{variant}_trace.json", history)
                    if update % 100 == 0 or passed:
                        print(
                            f"Debug {variant}: update {update}, NLL {final['nll']:.5f}, "
                            f"C(H) {final['c_index']:.4f}, C(prob) {final['probability_c_index']:.4f}",
                            flush=True,
                        )
                    if passed:
                        break
            checkpoint = {
                "model": model.state_dict(),
                "variant": variant,
                "identity": identity,
                "config": asdict(config),
                "update": update,
            }
            _save_checkpoint(output / f"{variant}.pt", checkpoint)
            if variant == "b":
                baseline = checkpoint
            report["arms"][variant] = {
                "initial": initial,
                "final": final,
                "passed": passed,
                "best_rank": best_rank,
                "initialization_check": check,
                "updates": update,
            }
            write_json(output / f"{variant}_trace.json", history)
        except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
            write_json(output / f"{variant}_trace.json", history)
            report["arms"][variant] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        write_json(output / "gate.json", report)
        if not report["arms"][variant]["passed"]:
            break
    report["passed"] = len(report["arms"]) == 3 and all(v["passed"] for v in report["arms"].values())
    write_json(output / "gate.json", report)
    return report


def stable_run(path):
    if not (path / "summary.json").is_file():
        return False
    summary = read_json(path / "summary.json")
    reports = sorted(path.glob("diagnostics_epoch_*.json"))
    if not reports:
        return False
    for report_path in reports:
        report = read_json(report_path)
        hazard = report.get("panel_variation", {}).get("hazard", {})
        risk = report.get("panel_variation", {}).get("risk", {})
        if (
            hazard.get("near_constant") is not False
            or risk.get("near_constant") is not False
            or any("Nonfinite" in w for w in report["warnings"])
        ):
            return False
    return isinstance(summary["validation_nll"], (int, float)) and math.isfinite(summary["validation_nll"])


def baseline_gate(root, protocol):
    results = {}
    identities = []
    for arm in ("a", "b"):
        directory = root / arm
        if not (directory / "best.pt").is_file() or not (directory / "summary.json").is_file():
            raise ValueError(f"Missing completed {arm} baseline at {directory}")
        summary = read_json(directory / "summary.json")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        if (
            checkpoint.get("variant") != arm
            or checkpoint.get("implementation_hashes") != implementation_hashes()
            or checkpoint.get("identity") != summary.get("identity")
            or checkpoint.get("best_validation_nll") != summary["validation_nll"]
        ):
            raise ValueError("Baseline summary/checkpoint provenance mismatch")
        identities.append(checkpoint["identity"])
        threshold = protocol["reference_baselines"][arm]["validation_nll"]
        results[arm] = {
            "validation_nll": summary["validation_nll"],
            "reference_nll": threshold,
            "stable": stable_run(directory),
            "checkpoint": str((directory / "best.pt").resolve()),
            "checkpoint_sha256": file_sha256(directory / "best.pt"),
        }
        results[arm]["passed"] = results[arm]["stable"] and summary["validation_nll"] <= threshold + 1e-6
    if identities[0] != identities[1]:
        raise ValueError("A/B baseline identities differ")
    result = {
        "passed": all(r["passed"] for r in results.values()),
        "arms": results,
        "identity": identities[0],
        "implementation_hashes": implementation_hashes(),
        "protocol_sha256": fingerprint(protocol),
    }
    write_json(root / "baseline_gate.json", result)
    return result


def primary_c(directory):
    summary = read_json(directory / "summary.json")
    value = summary["validation_by_landmark"]["24.0"]["c_index"][list(summary["horizons_months"]).index(60)]
    if value is None or not math.isfinite(value):
        raise ValueError("Primary validation C-index is not estimable; cannot promote this experiment")
    return value


def train_if_missing(config, records, schedule, variant, path):
    config = replace(config, variants=(variant,), experiment_id=path.name)
    if path.exists():
        if not (path / "summary.json").exists():
            raise ValueError(f"Incomplete experiment at {path}; use a new workspace for a fresh run")
        checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        ignored = {
            "device",
            "cpu_threads",
            "num_workers",
            "output_dir",
            "tensor_dir",
            "cache_path",
            "debug_gate",
            "baseline_gate",
            "protocol_path",
        }
        for key, value in asdict(config).items():
            if key not in ignored and fingerprint(saved.get(key)) != fingerprint(value):
                raise ValueError(f"Existing experiment configuration differs: {key}")
        if (
            checkpoint.get("implementation_hashes") != implementation_hashes()
            or checkpoint["identity"]["cohort_sha256"] != cohort_fingerprint(records)
            or checkpoint.get("variant") != variant
        ):
            raise ValueError("Existing experiment code, cohort or architecture differs")
        if config.baseline_init and checkpoint.get("initialization_sha256") != file_sha256(config.baseline_init):
            raise ValueError("Existing experiment initialization checkpoint changed")
        if config.cache_path:
            manifest = load_embeddings(config.cache_path, spatial_size=config.spatial_size)[0]["manifest"]
            if checkpoint["identity"]["encoder_sha256"] != manifest["encoder_sha256"]:
                raise ValueError("Existing experiment encoder differs")
        return read_json(path / "summary.json")
    return train_variant(config, records, schedule, variant, path)


def save_decision(path, decision, protocol, directories):
    decision = {
        **decision,
        "protocol_sha256": fingerprint(protocol),
        "implementation_hashes": implementation_hashes(),
        "checkpoints": {
            str((directory / "best.pt").resolve()): file_sha256(directory / "best.pt") for directory in directories
        },
    }
    write_json(path, decision)
    return decision


def read_decision(path, protocol):
    decision = read_json(path)
    if (
        decision.get("protocol_sha256") != fingerprint(protocol)
        or decision.get("implementation_hashes") != implementation_hashes()
        or not decision.get("checkpoints")
    ):
        raise ValueError("Optional experiment decision has stale or missing provenance")
    if any(file_sha256(p) != digest for p, digest in decision["checkpoints"].items()):
        raise ValueError("Optional experiment checkpoint changed after selection")
    return decision


def debug_run_name(value):
    """Keep named debug attempts separate from caches and production artifacts."""
    if not re.fullmatch(r"debug(?:_[A-Za-z0-9_-]+)?", value):
        raise argparse.ArgumentTypeError("Use debug or debug_<name> as a single directory name")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "cache-debug",
            "debug",
            "cache-full",
            "baselines",
            "history",
            "depth",
            "width",
            "step-ahead",
            "fine-tune",
            "report",
        ),
    )
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument(
        "--debug-run",
        type=debug_run_name,
        default="debug",
        help="Debug attempt directory, also selecting the gate for later stages",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-dir", type=Path)
    parser.add_argument("--anomaly-detection", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("cache-debug", "debug", "report", "fine-tune") and args.seed != 42:
        raise ValueError("The shared debug gate and final reporting use seed 42")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    records, schedule, audit = load_snapshot(protocol, args.tensor_dir)
    root = args.workspace
    root.mkdir(parents=True, exist_ok=True)
    protocol_file = root / "protocol.json"
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    write_json(protocol_file, protocol)
    support = validation_support(records)
    write_json(root / "audit.json", {"cohort": audit, "primary_validation_support": support})
    if args.stage == "audit":
        print(json.dumps(support, indent=2))
        return root
    if not support["confirmatory_support"]:
        raise ValueError("Insufficient validation support for the locked primary endpoint; inspect audit.json")
    reference_config = protocol["reference_config"]
    config = SurvivalConfig(
        seed=args.seed,
        device=args.device,
        diagnostics=True,
        epochs=reference_config["epochs"],
        patience=reference_config["patience"],
        batch_size=reference_config["batch_size"],
        learning_rate=reference_config["learning_rate"],
        weight_decay=reference_config["weight_decay"],
        gradient_clip_norm=reference_config["gradient_clip_norm"],
        spatial_size=tuple(reference_config["spatial_size"]),
        quadrature_points=reference_config["quadrature_points"],
        horizons=tuple(float(h) for h in reference_config["horizons"]),
        tensor_dir=Path(args.tensor_dir or protocol["reference_config"]["tensor_dir"]),
        anomaly_detection=args.anomaly_detection,
        protocol_path=args.protocol,
        debug_gate=root / args.debug_run / "gate.json",
        baseline_gate=root / f"seed_{args.seed}" / "baseline_gate.json",
    )
    cache = root / ("debug_embeddings.pt" if args.stage in ("cache-debug", "debug") else "embeddings.pt")
    config = replace(config, cache_path=cache)
    seed_root = root / f"seed_{args.seed}"
    seed_root.mkdir(exist_ok=True)
    if args.stage.startswith("cache-"):
        if args.stage == "cache-full":
            debug_cache = load_embeddings(root / "debug_embeddings.pt")[0]["manifest"]
            require_debug_gate(config.debug_gate, protocol, debug_cache, records=records)
        chosen = select_debug_patients(records) if args.stage == "cache-debug" else records
        encoder = OrthogonalSliceResNet101Encoder(weights=ResNet101_Weights.IMAGENET1K_V2)
        manifest = export_embeddings(
            cache,
            chosen,
            encoder,
            spatial_size=config.spatial_size,
            provenance=protocol["snapshot_hashes"],
            device=args.device,
        )
        write_json(cache.with_suffix(".json"), manifest)
        return cache
    manifest = load_embeddings(cache, provenance=protocol["snapshot_hashes"], spatial_size=config.spatial_size)[0][
        "manifest"
    ]
    if args.stage == "debug":
        result = run_debug(config, records, root / args.debug_run, protocol)
        if not result["passed"]:
            raise RuntimeError(f"Tiny-subset gate failed; inspect {config.debug_gate}")
        return result
    require_debug_gate(config.debug_gate, protocol, manifest, records=records)
    if args.stage == "baselines":
        for arm in ("a", "b"):
            train_if_missing(config, records, schedule, arm, seed_root / arm)
        result = baseline_gate(seed_root, protocol)
        if not result["passed"]:
            raise RuntimeError(f"Baseline promotion failed; inspect {seed_root / 'baseline_gate.json'}")
        return result
    if not baseline_gate(seed_root, protocol)["passed"]:
        raise ValueError("Production baseline gate failed; full LTSA training is blocked")
    config = replace(config, variants=("history",), baseline_init=seed_root / "b" / "best.pt")
    primary = seed_root / "history-1l"
    if args.stage == "history":
        return train_if_missing(config, records, schedule, "history", primary)
    if not stable_run(primary):
        raise ValueError("Optional experiments and reporting require a stable one-layer LTSA run")
    if args.stage in ("depth", "width", "step-ahead"):
        if args.stage == "width" and not (root / "depth_decision.json").exists():
            raise ValueError("Evaluate depth before width")
        changes = {"depth": {"n_layers": 2}, "width": {"d_model": 256}, "step-ahead": {"auxiliary_weight": 0.01}}[
            args.stage
        ]
        if args.stage == "width" and read_decision(root / "depth_decision.json", protocol)["retained"]:
            changes["n_layers"] = 2
        results = []
        for seed in protocol["optional_seeds"]:
            directory = root / f"seed_{seed}"
            # Corresponding baselines and 1L controls must pass before a depth/width/auxiliary trial.
            if not baseline_gate(directory, protocol)["passed"] or not stable_run(directory / "history-1l"):
                raise ValueError(f"Run passed baselines and history for seed {seed} first")
            candidate_config = replace(
                config,
                seed=seed,
                baseline_init=directory / "b" / "best.pt",
                baseline_gate=directory / "baseline_gate.json",
                **changes,
            )
            candidate = directory / f"history-{args.stage}"
            candidate_summary = train_if_missing(candidate_config, records, schedule, "history", candidate)
            control = directory / (
                "history-depth" if args.stage == "width" and changes.get("n_layers") == 2 else "history-1l"
            )
            results.append(
                {
                    "seed": seed,
                    "candidate": str(candidate),
                    "control": str(control),
                    "nll_difference": candidate_summary["validation_nll"]
                    - read_json(control / "summary.json")["validation_nll"],
                    "c_difference": primary_c(candidate) - primary_c(control),
                    "stable": stable_run(candidate),
                }
            )
        retained = (
            all(r["stable"] for r in results)
            and sum(r["nll_difference"] for r in results) < 0
            and sum(r["c_difference"] for r in results) >= 0
        )
        decision = {"retained": retained, "results": results, "selection_split": "validation"}
        return save_decision(
            root / f"{args.stage}_decision.json",
            decision,
            protocol,
            [Path(r[key]) for r in results for key in ("candidate", "control")],
        )
    if args.stage == "fine-tune":
        candidate_config = replace(
            config, cache_path=None, fine_tune_final_block=True, baseline_init=primary / "best.pt"
        )
        candidate = seed_root / "history-fine-tune"
        summary = train_if_missing(candidate_config, records, schedule, "history", candidate)
        decision = {
            "retained": stable_run(candidate)
            and summary["validation_nll"] < read_json(primary / "summary.json")["validation_nll"]
            and primary_c(candidate) >= primary_c(primary),
            "candidate": str(candidate),
            "selection_split": "validation",
        }
        return save_decision(root / "fine-tune_decision.json", decision, protocol, [primary, candidate])
    return final_report(root, seed_root, config, records, schedule, protocol)


def final_report(root, seed_root, config, records, schedule, protocol):
    """Explicit held-out evaluation after configurations and retention are locked."""
    output = root / "final_report"
    retained = {"a": seed_root / "a", "b": seed_root / "b", "history-1l": seed_root / "history-1l"}
    decisions = {}
    for name in ("depth", "width", "step-ahead", "fine-tune"):
        decision_path = root / f"{name}_decision.json"
        if not decision_path.exists() and any(root.glob(f"seed_*/history-{name}")):
            raise ValueError(f"Finish the validation decision for {name} before held-out reporting")
        decisions[name] = (
            read_decision(decision_path, protocol) if decision_path.exists() else {"status": "not_attempted"}
        )
        if decisions[name].get("retained"):
            retained[f"history-{name}"] = seed_root / f"history-{name}"
    # Preflight every retained artifact before touching the held-out prediction loader.
    manifests = {}
    cache_manifest = load_embeddings(config.cache_path, provenance=protocol["snapshot_hashes"])[0]["manifest"]
    for name, directory in retained.items():
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        identity = checkpoint.get("identity", {})
        if (
            checkpoint.get("implementation_hashes") != implementation_hashes()
            or identity.get("cohort_sha256") != cohort_fingerprint(records)
            or identity.get("encoder_sha256") != cache_manifest["encoder_sha256"]
            or not stable_run(directory)
        ):
            raise ValueError(f"Retained experiment {name} lacks matching, stable provenance")
        manifests[name] = {
            "checkpoint": str((directory / "best.pt").resolve()),
            "checkpoint_sha256": file_sha256(directory / "best.pt"),
            "config": checkpoint["config"],
            "identity": identity,
        }
        del checkpoint
    output.mkdir(exist_ok=False)
    write_json(
        output / "locked_experiments.json",
        {"protocol": protocol, "experiments": manifests, "decisions": decisions},
    )
    predictions, details = {}, {}
    for name, directory in retained.items():
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        selected_config = replace(
            config,
            n_layers=saved["n_layers"],
            d_model=saved["d_model"],
            dropout=saved["dropout"],
            auxiliary_weight=saved["auxiliary_weight"],
            cache_path=None if saved["fine_tune_final_block"] else config.cache_path,
        )
        encoder = (
            OrthogonalSliceResNet101Encoder(weights=None)
            if selected_config.cache_path is None
            else CachedEncoder(load_embeddings(selected_config.cache_path)[0]["manifest"])
        )
        model = ContinuousMRIModel(
            encoder,
            checkpoint["variant"],
            freeze_encoder=True,
            n_layers=selected_config.n_layers,
            d_model=selected_config.d_model,
            dropout=selected_config.dropout,
            step_ahead=selected_config.auxiliary_weight > 0,
        ).to(config.device)
        model.load_state_dict(checkpoint["model"])
        rows = []
        for landmark in schedule["follow_up_months"]:
            eligible = at_landmark(records, float(landmark), "test")
            if eligible:
                rows.extend(
                    predict(
                        model,
                        _loader(eligible, selected_config, model.variant, schedule.height),
                        config.device,
                        config.horizons,
                    )
                )
        predictions[name] = rows
        details[name] = {
            "checkpoint": str(directory / "best.pt"),
            "identity": checkpoint["identity"],
            "checkpoint_sha256": manifests[name]["checkpoint_sha256"],
            "parameters": read_json(directory / "parameter_counts.json"),
            "validation_nll": checkpoint["best_validation_nll"],
        }
        write_json(output / f"{name}_test_predictions.json", rows)
        del model
    comparisons = compare_experiments(predictions, config.horizons, bootstrap_samples=1000, seed=42)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "models.json", details)
    primary = next((r for r in comparisons if r["primary"]), None)
    conclusion = "inconclusive"
    if primary is not None and primary["status"] == "estimated":
        if primary["confidence_interval_low"] > 0:
            conclusion = "evidence of improved discrimination"
        elif primary["confidence_interval_high"] < 0:
            conclusion = "evidence of worse discrimination"
    primary = primary or {"difference": None, "confidence_interval_low": None, "confidence_interval_high": None}
    rows = ["| Model | Validation NLL | Trainable parameters |", "|---|---:|---:|"]
    rows.extend(
        f"| {name} | {item['validation_nll']:.6f} | {sum(p['trainable'] for p in item['parameters'].values()):,} |"
        for name, item in details.items()
    )
    (output / "report.md").write_text(
        f"# Locked LTSA comparison\n\nPrimary: LTSA-1L minus B at month 24, next 60 months.\n\n"
        f"Result: {conclusion}. C-index difference: {primary['difference']}; "
        f"95% interval: [{primary['confidence_interval_low']}, {primary['confidence_interval_high']}].\n\n"
        + "\n".join(rows)
        + "\n\n"
        "See comparisons.json for the full grid, test NLL, individual and paired intervals, and MRI-count strata; models.json for provenance and counts; "
        "and locked_experiments.json for validation-only retention decisions.\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    main()
