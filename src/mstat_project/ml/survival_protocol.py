"""Shared provenance and training gates for both continuous CLI entry points."""

import hashlib
import json
import math
from pathlib import Path

from mstat_project.ml.survival_embeddings import fingerprint

PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "docs" / "ltsa_protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_hashes():
    directory = Path(__file__).parent
    names = {p.name for p in directory.glob("*survival*.py")}
    names.update(("ltsa_experiments.py", "models.py", "landmarks.py", "metrics.py", "utils.py"))
    return {name: file_sha256(directory / name) for name in sorted(names)}


def cohort_fingerprint(records):
    return fingerprint(sorted((r.ptid, r.split, r.image_ids, r.acquisition_months,
                               r.visit_nums, r.prediction_months, r.outcome_months, r.event) for r in records))


def require_debug_gate(path, protocol, encoder_manifest, *, records=None):
    if path is None:
        raise ValueError("Full training requires --debug-gate from a passed ltsa_experiments debug run")
    report = read_json(path)
    if (not report.get("passed") or set(report.get("arms", {})) != {"a", "b", "history"}
        or report.get("protocol_sha256") != fingerprint(protocol)
        or report.get("snapshot_hashes") != protocol["snapshot_hashes"]
        or report.get("encoder_sha256") != encoder_manifest["encoder_sha256"]
        or report.get("implementation_hashes") != implementation_hashes()):
        raise ValueError("Full training requires a passed debug gate for this protocol, code and encoder")
    if records is not None and report.get("full_cohort_sha256") != cohort_fingerprint(records):
        raise ValueError("Debug gate cohort differs from the current cohort")
    for arm, result in report["arms"].items():
        initial, final = result.get("initial", {}), result.get("final", {})
        numbers = (initial.get("nll"), final.get("nll"), final.get("c_index"), final.get("risk_variance"))
        if (not result.get("passed") or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in numbers)
            or numbers[0] - numbers[1] < protocol["debug"]["minimum_nll_drop"]
            or numbers[2] < protocol["debug"]["minimum_c_index"] or numbers[3] <= 1e-8):
            raise ValueError(f"Debug gate lacks passing empirical evidence for {arm}")
        if arm == "history":
            check = result.get("initialization_check") or {}
            if not check.get("passed") or not 0 <= check.get("max_survival_difference", math.inf) <= .01:
                raise ValueError("Debug gate lacks a passed LTSA initialization check")
    return report


def require_baseline_gate(path, protocol, identity, checkpoint_path):
    """Bind promotion to the exact selected A/B files and their cohort identity."""
    if path is None:
        raise ValueError("History training requires --baseline-gate from promoted A/B experiments")
    report = read_json(path)
    if not report.get("passed") or report.get("protocol_sha256") != fingerprint(protocol):
        raise ValueError("Production baseline gate failed or belongs to another protocol")
    if report.get("identity") != identity or report.get("implementation_hashes") != implementation_hashes():
        raise ValueError("Production baseline gate has incompatible cohort, encoder or code")
    for arm in ("a", "b"):
        result = report.get("arms", {}).get(arm, {})
        value = result.get("validation_nll", math.inf)
        threshold = protocol["reference_baselines"][arm]["validation_nll"]
        if (not result.get("passed") or not result.get("stable") or not math.isfinite(value)
            or value > threshold + protocol["baseline_nll_tolerance"]):
            raise ValueError(f"Production baseline gate lacks passing evidence for {arm}")
        if file_sha256(result["checkpoint"]) != result.get("checkpoint_sha256"):
            raise ValueError("Promoted baseline checkpoint changed")
    if Path(checkpoint_path).resolve() != Path(report["arms"]["b"]["checkpoint"]).resolve():
        raise ValueError("History initialization must use the promoted B checkpoint")
    return report
