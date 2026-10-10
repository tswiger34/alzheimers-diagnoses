"""Inspect saved tiny-subset checkpoints without training or changing the gate."""

import argparse
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from mstat_project.ml.continuous_survival import ContinuousMRIModel
from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot, select_debug_patients
from mstat_project.ml.survival import SurvivalConfig, _device_batch, _loader, model_identity, write_json
from mstat_project.ml.survival_embeddings import CachedEncoder, fingerprint, load_embeddings
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, cohort_fingerprint, file_sha256, implementation_hashes, read_json,
)
from mstat_project.ml.utils import concordance_index


def risk_diagnostics(cumulative, duration, event, horizon=60.):
    """Compare rank-preserving hazards with the original probability risk score.

    Full-follow-up scores reproduce the debug gate's pair eligibility. Truncated
    scores are separate diagnostics, not replacements for the locked gate.
    Casting an already rounded probability to double cannot recover lost ranks.
    """
    cumulative, duration, event = (v.detach().reshape(-1).cpu() for v in (cumulative, duration, event))
    event = event.bool()
    if not (cumulative.numel() == duration.numel() == event.numel()) or not cumulative.numel():
        raise ValueError("Risk diagnostics require nonempty aligned observations")
    if (not torch.isfinite(cumulative).all() or not torch.isfinite(duration).all()
        or (cumulative < 0).any() or (duration <= 0).any() or not math.isfinite(horizon) or horizon <= 0):
        raise ValueError("Risk diagnostics require finite nonnegative hazards and positive durations/horizon")
    probability = 1 - torch.exp(-cumulative)
    probability64 = -torch.expm1(-cumulative.double())

    def scores(time, observed):
        pairs = observed[:, None] & (time[:, None] < time[None, :])
        probability_ties = probability[:, None] == probability[None, :]
        hazard_ties = cumulative[:, None] == cumulative[None, :]

        def c_index(risk):
            value = concordance_index(risk, time, observed)
            return value if math.isfinite(value) else None

        return {
            "comparable_pairs": int(pairs.sum()),
            "probability_c_index": c_index(probability),
            "probability_float64_c_index": c_index(probability64),
            "cumulative_hazard_c_index": c_index(cumulative),
            "probability_tied_pairs": int((pairs & probability_ties).sum()),
            "cumulative_hazard_tied_pairs": int((pairs & hazard_ties).sum()),
            "pairs_tied_only_after_probability_conversion": int((pairs & probability_ties & ~hazard_ties).sum()),
        }

    return {
        "horizon_months": horizon,
        "observations": cumulative.numel(),
        "probabilities_equal_one": int((probability == 1).sum()),
        "probabilities_equal_zero": int((probability == 0).sum()),
        "probability_variance": float(probability.var(unbiased=False)),
        "cumulative_hazard_min": float(cumulative.min()),
        "cumulative_hazard_max": float(cumulative.max()),
        "full_followup": scores(duration, event),
        "horizon_truncated": scores(duration.clamp(max=horizon), event & (duration <= horizon)),
        "values": {"cumulative_hazard": cumulative.tolist(), "probability": probability.tolist(),
                   "probability_float64": probability64.tolist(),
                   "duration_months": duration.tolist(), "event": event.tolist()},
    }


@torch.inference_mode()
def inspect_predictions(model, loader, device):
    """Evaluate the fixed training panel, restoring all module training flags."""
    modes = [(module, module.training) for module in model.modules()]
    cumulative, durations, events, integrals, event_logs, patients = [], [], [], [], [], []
    try:
        model.eval()
        for batch in loader:
            batch = _device_batch(batch, device)
            z = model.encode_batch(batch)
            integral, event_log = model.head.nll_components(z, batch["entry"], batch["outcome"], batch["event"])
            integrals.append(integral.cpu())
            event_logs.append(event_log.cpu())
            cumulative.append(model.head.cumulative_hazard(z, batch["entry"], batch["entry"] + 60).cpu())
            durations.append((batch["outcome"] - batch["entry"]).cpu())
            events.append(batch["event"].cpu())
            patients.extend(r.ptid for r in batch["records"])
        result = risk_diagnostics(torch.cat(cumulative), torch.cat(durations), torch.cat(events))
        integral, event_log = torch.cat(integrals), torch.cat(event_logs)
        if not torch.isfinite(integral + event_log).all():
            raise FloatingPointError("Nonfinite saved-checkpoint likelihood")
        result.update(nll=float((integral + event_log).mean()),
                      nll_cumulative_mean=float(integral.mean()), nll_event_log_mean=float(event_log.mean()))
        result["values"]["patients"] = patients
        return result
    finally:
        for module, training in modes:
            module.training = training


def inspect_debug_run(workspace, records, protocol, device, *, debug_run="debug"):
    """Return evidence from saved arms; never promote or rewrite experiment files."""
    directory = Path(workspace) / debug_run_name(debug_run)
    gate_path = directory / "gate.json"
    gate = read_json(gate_path)
    if (gate.get("protocol_sha256") != fingerprint(protocol)
        or gate.get("snapshot_hashes") != protocol["snapshot_hashes"]
        or gate.get("full_cohort_sha256") != cohort_fingerprint(records)):
        raise ValueError("Saved debug gate has incompatible protocol or cohort provenance")
    config = SurvivalConfig(**gate["config"])
    config = replace(config, device=device, cache_path=Path(workspace) / "debug_embeddings.pt",
                     spatial_size=tuple(config.spatial_size), horizons=tuple(config.horizons))
    cache = load_embeddings(config.cache_path, provenance=protocol["snapshot_hashes"],
                            spatial_size=config.spatial_size)[0]["manifest"]
    if gate.get("encoder_sha256") != cache["encoder_sha256"]:
        raise ValueError("Saved debug encoder differs from the embedding cache")
    subset = select_debug_patients(records, config.seed)
    if gate.get("patients") != [r.ptid for r in subset] or gate.get("image_ids") != [r.image_ids[-1] for r in subset]:
        raise ValueError("Saved debug panel differs from the selected training observations")
    current_code = implementation_hashes()
    result = {
        "diagnostic_only": True, "saved_gate_passed": gate["passed"],
        "saved_c_index_risk_score": gate.get("c_index_risk_score", "probability_60_months"),
        "gate_sha256": file_sha256(gate_path), "protocol_sha256": fingerprint(protocol),
        "cache_id": cache.get("cache_id"), "inspector_sha256": file_sha256(Path(__file__)),
        "training_implementation_hashes": gate.get("implementation_hashes"),
        "inspection_implementation_hashes": current_code,
        "implementation_matches_training": gate.get("implementation_hashes") == current_code,
        "arms": {},
    }
    for variant, saved in gate["arms"].items():
        if variant not in ("a", "b", "history"):
            raise ValueError(f"Unrecognized debug arm: {variant}")
        path = directory / f"{variant}.pt"
        if not path.exists():
            result["arms"][variant] = {"unavailable": "No saved checkpoint", "saved_gate": saved}
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("variant") != variant
            or fingerprint(checkpoint.get("config")) != fingerprint(gate["config"])):
            raise ValueError(f"Saved {variant} checkpoint/configuration differs from the gate")
        model = ContinuousMRIModel(CachedEncoder(cache), variant, freeze_encoder=True,
            quadrature_points=config.quadrature_points, n_layers=config.n_layers,
            d_model=config.d_model, dropout=config.dropout).to(device)
        if checkpoint.get("identity") != model_identity(model, subset, config):
            raise ValueError(f"Saved {variant} checkpoint has incompatible model/cache/cohort identity")
        model.load_state_dict(checkpoint["model"])
        metrics = inspect_predictions(model, _loader(subset, config, variant, 1), device)
        arm = {"checkpoint_sha256": file_sha256(path), "update": checkpoint["update"],
               "saved_gate": saved, "metrics": metrics}
        initial_nll = saved.get("initial", {}).get("nll")
        if initial_nll is not None:
            arm["required_nll"] = initial_nll - protocol["debug"]["minimum_nll_drop"]
        trace_path = directory / f"{variant}_trace.json"
        if trace_path.exists():
            trace = read_json(trace_path)
            eligible = [row for row in trace if row["c_index"] >= protocol["debug"]["minimum_c_index"]]
            arm["trace_sha256"] = file_sha256(trace_path)
            arm["lowest_nll_with_passing_recorded_c_index"] = (
                {key: min(eligible, key=lambda row: row["nll"])[key] for key in ("update", "nll", "c_index")}
                if eligible else None)
        result["arms"][variant] = arm
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--debug-run", type=debug_run_name, default="debug")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    if read_json(args.workspace / "protocol.json") != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    records, _, _ = load_snapshot(protocol, args.tensor_dir)
    result = inspect_debug_run(args.workspace, records, protocol, args.device, debug_run=args.debug_run)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.workspace / args.debug_run / f"inspection_{stamp}.json"
    write_json(output, result)
    summary = {"report": str(output), "saved_gate_passed": result["saved_gate_passed"],
               "implementation_matches_training": result["implementation_matches_training"], "arms": {}}
    for variant, arm in result["arms"].items():
        if "metrics" not in arm:
            summary["arms"][variant] = {"unavailable": arm["unavailable"]}
            continue
        metrics = arm["metrics"]
        summary["arms"][variant] = {
            "nll": metrics["nll"], "required_nll": arm.get("required_nll"),
            "probabilities_equal_one": metrics["probabilities_equal_one"],
            "full_followup": metrics["full_followup"], "horizon_truncated": metrics["horizon_truncated"],
        }
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    main()
