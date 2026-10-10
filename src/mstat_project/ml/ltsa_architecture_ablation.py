"""Validation-only architecture isolation; no production promotion or test inference."""

import argparse
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mstat_project.ml.continuous_survival import ContinuousHazardHead, bounded_time, initial_hazard
from mstat_project.ml.ltsa_experiments import debug_run_name, load_snapshot
from mstat_project.ml.survival import (
    SurvivalConfig, _device_batch, _loader, _save_checkpoint, _seed, train_epoch, validation_loss, write_json,
)
from mstat_project.ml.survival_data import at_landmark, sample_epoch
from mstat_project.ml.survival_embeddings import CachedEncoder, fingerprint, load_embeddings
from mstat_project.ml.survival_optimization import optimizer_groups, parameter_counts
from mstat_project.ml.survival_protocol import (
    PROTOCOL_PATH, cohort_fingerprint, file_sha256, implementation_hashes, read_json, require_debug_gate,
)
from mstat_project.ml.utils import concordance_index


@dataclass(frozen=True)
class Architecture:
    projection_norm: bool = True
    head_norm: bool = True
    activation: str = "gelu"
    hazard_time: str = "bounded"
    acquisition_time: str = "bounded"
    small_output_weights: bool = True
    calibrated_output_bias: bool = True


STABILIZED = Architecture()
EXPERIMENTS = {
    "stabilized": STABILIZED,
    "no_projection_norm": replace(STABILIZED, projection_norm=False),
    "no_head_norm": replace(STABILIZED, head_norm=False),
    "tanh": replace(STABILIZED, activation="tanh"),
    "legacy_hazard_time": replace(STABILIZED, hazard_time="months_over_12"),
    "legacy_acquisition_time": replace(STABILIZED, acquisition_time="months_over_12"),
    "default_output_weights": replace(STABILIZED, small_output_weights=False),
    "default_output_bias": replace(STABILIZED, calibrated_output_bias=False),
    "legacy_initialization": replace(STABILIZED, small_output_weights=False, calibrated_output_bias=False),
    "legacy_all": Architecture(False, False, "tanh", "months_over_12", "months_over_12", False, False),
}
TERMS = ("cumulative_hazard", "event_log_hazard", "nll")
REPRODUCTION_TOLERANCE = 1e-5
LOGGER = logging.getLogger(__name__)


def model_time(time, transform):
    if transform == "bounded":
        return bounded_time(time)
    if transform != "months_over_12" or not torch.isfinite(time).all() or (time < 0).any():
        raise ValueError("Invalid time transform or acquisition/query times")
    return time / 12.


class AblationHead(ContinuousHazardHead):
    """Reuse the production likelihood/integration; explicitly vary head inputs."""

    def __init__(self, features, quadrature_points, baseline_hazard, architecture):
        nn.Module.__init__(self)
        self.architecture = architecture
        if quadrature_points < 8 or not math.isfinite(baseline_hazard) or baseline_hazard <= 1e-7 / 12:
            raise ValueError("Invalid quadrature or initial monthly hazard")
        self.input_norm = nn.LayerNorm(features + 1)
        self.network = nn.Sequential(nn.Linear(features + 1, 64), nn.GELU(), nn.Linear(64, 1))
        default_weight = self.output_linear.weight.detach().clone()
        default_bias = self.output_linear.bias.detach().clone()
        # Draw the same random numbers in every arm. Only the nominated values
        # are replaced; projection/time/hidden weights are exactly matched.
        nn.init.normal_(self.output_linear.weight, std=1e-3)
        rate = baseline_hazard * 12 - 1e-7
        nn.init.constant_(self.output_linear.bias, rate + math.log(-math.expm1(-rate)))
        with torch.no_grad():
            if not architecture.small_output_weights:
                self.output_linear.weight.copy_(default_weight)
            if not architecture.calibrated_output_bias:
                self.output_linear.bias.copy_(default_bias)
        if not architecture.head_norm:
            self.input_norm = nn.Identity()
        if architecture.activation == "tanh":
            self.network[1] = nn.Tanh()
        elif architecture.activation != "gelu":
            raise ValueError("Invalid ablation activation")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        self.register_buffer("nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def hazard(self, z, time):
        if time.ndim == 1:
            time = time[:, None]
        expanded = z[:, None, :].expand(-1, time.shape[1], -1)
        inputs = torch.cat((expanded, model_time(time, self.architecture.hazard_time)[..., None]), -1)
        return (F.softplus(self.network(self.input_norm(inputs))).squeeze(-1) + 1e-7) / 12


class AblationModel(nn.Module):
    """Embedding-only A/B; deliberately incompatible with production checkpoints."""

    def __init__(self, manifest, variant, architecture, baseline_hazard, quadrature_points=32):
        super().__init__()
        if variant not in ("a", "b") or architecture not in EXPERIMENTS.values():
            raise ValueError("Only declared baseline ablations are supported")
        self.register_buffer("_ablation_format_version", torch.tensor(1))
        self.variant, self.architecture = variant, architecture
        self.encoder = CachedEncoder(manifest)
        self.projection = nn.Linear(self.encoder.n_features, 128)
        self.projection_norm = nn.LayerNorm(128)
        self.time_embedding = nn.Linear(1, 128)
        if variant == "a":
            self.time_embedding.requires_grad_(False)
        self.head = AblationHead(128, quadrature_points, baseline_hazard, architecture)
        if not architecture.projection_norm:
            self.projection_norm = nn.Identity()
        self.transformer = None

    def encode_batch(self, batch):
        if "embeddings" not in batch:
            raise ValueError("Architecture ablations require frozen cached inputs")
        raw, times, observed = batch["embeddings"], batch["times"], batch["observed"]
        if observed.dtype != torch.bool or not observed.any(dim=1).all():
            raise ValueError("Each observation requires an observed MRI")
        positions = torch.arange(observed.shape[1], device=raw.device)[None].expand_as(observed)
        last = positions.masked_fill(~observed, -1).max(dim=1).values
        rows = torch.arange(raw.shape[0], device=raw.device)
        z = self.projection_norm(self.projection(raw[rows, last]))
        if self.variant == "b":
            z = z + self.time_embedding(model_time(times[rows, last, None], self.architecture.acquisition_time))
        return z

    def forward(self, batch):
        return self.head.nll(self.encode_batch(batch), batch["entry"], batch["outcome"], batch["event"])


def encoder_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_saved_weights(model, checkpoint, kind, manifest):
    """Explicit inspection adapter, not a general production compatibility path."""
    if checkpoint.get("variant") != model.variant:
        raise ValueError("Checkpoint arm differs from the requested model")
    state = dict(checkpoint["model"])
    if kind == "reference":
        if model.architecture != EXPERIMENTS["legacy_all"] or "_architecture_version" in state:
            raise ValueError("Reference inspection requires the original unversioned A/B architecture")
        encoder = {name.removeprefix("encoder."): value for name, value in state.items()
                   if name.startswith("encoder.")}
        if not encoder or encoder_state_hash(encoder) != manifest["encoder_sha256"]:
            raise ValueError("Reference encoder weights differ from the frozen embedding cache")
        state = {name: value for name, value in state.items() if not name.startswith("encoder.")}
    elif kind == "stabilized":
        if model.architecture != STABILIZED or int(state.pop("_architecture_version", -1)) != 2:
            raise ValueError("Stabilized inspection requires architecture version 2")
        if checkpoint.get("identity", {}).get("encoder_sha256") != manifest["encoder_sha256"]:
            raise ValueError("Stabilized checkpoint encoder differs from cache")
    else:
        raise ValueError("Unknown saved-checkpoint format")
    state["_ablation_format_version"] = model._ablation_format_version
    model.load_state_dict(state, strict=True)


def aggregate_components(rows):
    """Equal weight per patient, then per eligible observation within patient."""
    grouped = defaultdict(list)
    keys = set()
    for row in rows:
        key = (row["ptid"], row["entry"], tuple(row["image_ids"]))
        if key in keys or not all(math.isfinite(row[term]) for term in TERMS):
            raise ValueError("Duplicate observation or nonfinite validation loss")
        keys.add(key)
        if abs(row["nll"] - row["cumulative_hazard"] - row["event_log_hazard"]) > 1e-5:
            raise ValueError("Validation components do not reconstruct NLL")
        grouped[row["ptid"]].append(row)
    if not grouped:
        raise ValueError("Validation cohort is empty")
    patients = []
    for ptid, observations in sorted(grouped.items()):
        if len({(r["event"], r["outcome"]) for r in observations}) != 1:
            raise ValueError("Patient outcomes are inconsistent across observations")
        patients.append({"ptid": ptid, "event": observations[0]["event"],
            "outcome": observations[0]["outcome"], "observations": len(observations),
            **{term: sum(r[term] for r in observations) / len(observations) for term in TERMS}})

    def summarize(items):
        return {"patients": len(items), **{term: sum(r[term] for r in items) / len(items) if items else None
                                          for term in TERMS}}

    overall = {**summarize(patients), "observations": len(rows)}
    by_event = {}
    for event, label in ((True, "event"), (False, "censored")):
        group = [p for p in patients if p["event"] == event]
        by_event[label] = {**summarize(group), "contribution_to_overall": {
            term: sum(p[term] for p in group) / len(patients) for term in TERMS}}
    by_entry = {}
    for label, low, high in (("0_to_24", 0., 24.), ("24_to_60", 24., 60.), ("60_plus", 60., math.inf)):
        selected = [r for r in rows if low <= r["entry"] < high]
        # Contributions, not pooled independent-patient scores: groups can
        # contain different observations from the same patient.
        by_entry[label] = {"observations": len(selected), "patients": len({r["ptid"] for r in selected}),
            "contribution_to_overall": {term: sum(r[term] / len(grouped[r["ptid"]]) for r in selected)
                                        / len(patients) for term in TERMS}}
    return {"overall": overall, "by_event": by_event, "by_entry_months": by_entry, "patients": patients}


@torch.no_grad()
def validation_components(model, loader, device):
    modes = [(module, module.training) for module in model.modules()]
    rows = []
    try:
        model.eval()
        for batch in loader:
            if any(r.split != "validation" for r in batch["records"]):
                raise ValueError("Loss-component inspection accepts validation observations only")
            batch = _device_batch(batch, device)
            z = model.encode_batch(batch)
            cumulative, event_log = model.head.nll_components(z, batch["entry"], batch["outcome"], batch["event"])
            endpoint_hazard = model.head.hazard(z, batch["outcome"]).squeeze(-1)
            for record, h, e, hazard in zip(batch["records"], cumulative.cpu().tolist(),
                                           event_log.cpu().tolist(), endpoint_hazard.cpu().tolist(), strict=True):
                if not math.isfinite(hazard) or hazard <= 0:
                    raise ValueError("Nonpositive or nonfinite validation hazard")
                rows.append({"ptid": record.ptid, "entry": record.prediction_months,
                    "outcome": record.outcome_months, "event": record.event, "image_ids": list(record.image_ids),
                    "cumulative_hazard": h, "event_log_hazard": e, "nll": h + e, "endpoint_hazard": hazard})
        return {**aggregate_components(rows), "rows": rows}
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def primary_metrics(model, records, config):
    rows = at_landmark(records, 24., "validation")
    risks, times, events = [], [], []
    model.eval()
    for batch in _loader(rows, config, model.variant, 1):
        batch = _device_batch(batch, config.device)
        risks.append(model.head.cumulative_hazard(model.encode_batch(batch), batch["entry"], batch["entry"] + 60).cpu())
        duration = batch["outcome"] - batch["entry"]
        times.append(duration.clamp(max=60).cpu())
        events.append((batch["event"].bool() & (duration <= 60)).cpu())
    if not risks:
        return {"patients": 0, "events": 0, "c_index_cumulative_hazard": None}
    risk, time, event = torch.cat(risks), torch.cat(times), torch.cat(events)
    if not torch.isfinite(risk).all():
        raise ValueError("Nonfinite primary-endpoint cumulative hazards")
    c = concordance_index(risk, time, event)
    return {"patients": len(rows), "events": int(event.sum()),
        "comparable_pairs": int((event[:, None] & (time[:, None] < time[None, :])).sum()),
        "c_index_cumulative_hazard": c if math.isfinite(c) else None,
        "cumulative_hazard_variance": float(risk.double().var(unbiased=False)),
        "probabilities_equal_one": int(((1 - torch.exp(-risk)) == 1).sum())}


def paired_components(control, candidate):
    def observations(report):
        return sorted((r["ptid"], r["entry"], r["outcome"], r["event"], tuple(r["image_ids"])) for r in report["rows"])
    if observations(control) != observations(candidate):
        raise ValueError("Component comparison requires identical validation histories and outcomes")
    left = {p["ptid"]: p for p in control["patients"]}
    differences = [{"ptid": p["ptid"], **{term: p[term] - left[p["ptid"]][term] for term in TERMS}}
                   for p in candidate["patients"]]
    return {"direction": "candidate minus control", "patients": len(differences),
        "mean_difference": {term: sum(p[term] for p in differences) / len(differences) for term in TERMS},
        "patient_differences": differences,
        "largest_nll_increases": sorted(differences, key=lambda p: p["nll"], reverse=True)[:10]}


def study_name(value):
    if not re.fullmatch(r"architecture_[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Use a single directory name beginning architecture_")
    return value


def verified_reference_replay(path, specification, batch_size, device):
    """Accept measured cache scores only after raw historical reproduction."""
    path = Path(path).resolve()
    report = read_json(path)
    original_path = path.parent.parent / "specification.json"
    original = read_json(original_path)
    # The adapter's gate logic changed; the scientific inputs must not change.
    comparable = lambda spec: {k: v for k, v in spec.items()
                               if k not in ("ablation_code_sha256", "reference_replay")}
    if (report.get("study_fingerprint") != fingerprint(original)
        or comparable(original) != comparable(specification)
        or report.get("replay_code_sha256") != file_sha256(Path(__file__).with_name("ltsa_reference_replay.py"))
        or report.get("batch_size") != batch_size or report.get("device") != device
        or report.get("torch_version") != str(torch.__version__)
        or report.get("cuda_version") != torch.version.cuda):
        raise ValueError("Reference replay provenance or runtime differs from this study")
    if report.get("feature_differences", {}).get("singleton_minus_cache") != {"max_abs": 0., "mean_image_rms": 0.}:
        raise ValueError("Reference replay did not establish exact singleton/cache agreement")
    expected = {}
    for arm in ("a", "b"):
        result = report["arms"][arm]
        historical = specification["sources"][arm]["reference_nll"]
        raw = result["raw_batch"]["validation_nll_original_reduction"]
        cached = result["cached"]["validation_nll_original_reduction"]
        singleton = result["raw_singleton"]["validation_nll_original_reduction"]
        if (not all(math.isfinite(x) for x in (raw, cached, singleton))
            or result["expected_nll"] != historical
            or abs(raw - historical) > REPRODUCTION_TOLERANCE or cached != singleton
            or result["cached"]["overall"] != result["raw_singleton"]["overall"]):
            raise ValueError("Reference replay failed raw reproduction or singleton/cache likelihood agreement")
        expected[arm] = cached
    return {"path": str(path), "sha256": file_sha256(path),
            "source_specification_sha256": file_sha256(original_path), "cached_reference_nll": expected,
            "interpretation": "Historical raw NLL reproduced; cached reference uses verified singleton encoder execution."}


def prepare_study(workspace, protocol_path, debug_run, device, name, reference_replay=None):
    workspace = Path(workspace)
    protocol = read_json(protocol_path)
    if read_json(workspace / "protocol.json") != protocol:
        raise ValueError("Workspace belongs to a different protocol")
    records, schedule, _ = load_snapshot(protocol)
    cache = workspace / "embeddings.pt"
    manifest = load_embeddings(cache, provenance=protocol["snapshot_hashes"],
                               spatial_size=protocol["reference_config"]["spatial_size"])[0]["manifest"]
    gate_path = workspace / debug_run_name(debug_run) / "gate.json"
    require_debug_gate(gate_path, protocol, manifest, records=records)
    source = workspace / "seed_42"
    baseline_gate = read_json(source / "baseline_gate.json")
    code = implementation_hashes()
    if baseline_gate.get("implementation_hashes") != code or baseline_gate.get("protocol_sha256") != fingerprint(protocol):
        raise ValueError("Source baseline code or protocol differs")
    configs, sources = {}, {}
    output = workspace / "investigations" / study_name(name)
    for arm in ("a", "b"):
        path = source / arm / "best.pt"
        sha = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (sha != baseline_gate["arms"][arm]["checkpoint_sha256"] or checkpoint.get("variant") != arm
            or checkpoint.get("implementation_hashes") != code
            or checkpoint.get("identity") != baseline_gate["identity"]
            or checkpoint["identity"]["cohort_sha256"] != cohort_fingerprint(records)
            or checkpoint["identity"]["encoder_sha256"] != manifest["encoder_sha256"]
            or checkpoint["best_validation_nll"] != baseline_gate["arms"][arm]["validation_nll"]):
            raise ValueError(f"Source {arm} baseline checkpoint provenance mismatch")
        config = SurvivalConfig(**checkpoint["config"])
        if config.seed != 42 or config.baseline_init or not config.freeze_encoder or config.auxiliary_weight:
            raise ValueError("Ablations require fresh seed-42 frozen baselines without auxiliary loss")
        for key in ("epochs", "patience", "learning_rate", "weight_decay", "gradient_clip_norm", "batch_size",
                    "quadrature_points"):
            if getattr(config, key) != protocol["reference_config"][key]:
                raise ValueError(f"Source baseline differs from locked training setting {key}")
        configs[arm] = replace(config, device=device, cache_path=cache, protocol_path=protocol_path,
                               output_dir=output, debug_gate=gate_path)
        configs[arm].validate()
        sources[arm] = {"stabilized_checkpoint": str(path.resolve()), "stabilized_sha256": sha,
            "stabilized_nll": checkpoint["best_validation_nll"],
            "reference_checkpoint": str(Path(protocol["reference_baselines"][arm]["checkpoint"]).resolve()),
            "reference_sha256": protocol["reference_baselines"][arm]["checkpoint_sha256"],
            "reference_nll": protocol["reference_baselines"][arm]["validation_nll"]}
        del checkpoint
    specification = {"study_version": 1, "sources": sources, "protocol_sha256": fingerprint(protocol),
        "cohort_sha256": cohort_fingerprint(records), "cache_id": manifest["cache_id"],
        "debug_gate_sha256": file_sha256(gate_path), "implementation_hashes": code,
        "ablation_code_sha256": file_sha256(Path(__file__)), "production_promoted": False,
        "selection_split": "validation", "experiments": {key: asdict(value) for key, value in EXPERIMENTS.items()},
        "reproduction_tolerance": REPRODUCTION_TOLERANCE}
    if reference_replay is not None:
        specification["reference_replay"] = verified_reference_replay(
            reference_replay, specification, configs["a"].batch_size, device)
    spec_path = output / "specification.json"
    if output.exists():
        if not spec_path.exists() or read_json(spec_path) != specification:
            raise ValueError("Study provenance changed; choose a new --study name and preserve existing artifacts")
    else:
        output.mkdir(parents=True)
        write_json(spec_path, specification)
    return {"output": output, "records": records, "schedule": schedule, "configs": configs,
            "manifest": manifest, "specification": specification, "sources": sources}


def new_model(study, arm, experiment):
    config = study["configs"][arm]
    _seed(config.seed)
    return AblationModel(study["manifest"], arm, EXPERIMENTS[experiment], initial_hazard(study["records"]),
                         config.quadrature_points).to(config.device)


def inspect_saved(study):
    output = study["output"] / "saved_checkpoint_inspection"
    output.mkdir(exist_ok=True)
    result = {"production_promoted": False, "reproduced": True, "arms": {},
              "study_fingerprint": fingerprint(study["specification"])}
    for arm in ("a", "b"):
        config, source = study["configs"][arm], study["sources"][arm]
        loader = _loader([r for r in study["records"] if r.split == "validation"], config, arm, 1)
        reports = {}
        result["arms"][arm] = {}
        for kind, experiment in (("reference", "legacy_all"), ("stabilized", "stabilized")):
            model = new_model(study, arm, experiment)
            if file_sha256(source[f"{kind}_checkpoint"]) != source[f"{kind}_sha256"]:
                raise ValueError("Saved checkpoint changed after the study was prepared")
            checkpoint = torch.load(source[f"{kind}_checkpoint"], map_location="cpu", weights_only=False)
            load_saved_weights(model, checkpoint, kind, study["manifest"])
            del checkpoint
            report = validation_components(model, loader, config.device)
            report["primary_24_60"] = primary_metrics(model, study["records"], config)
            report["checkpoint_sha256"] = source[f"{kind}_sha256"]
            report["expected_nll"] = source[f"{kind}_nll"]
            report["historical_saved_nll"] = source[f"{kind}_nll"]
            evidence = study["specification"].get("reference_replay")
            if kind == "reference" and evidence is not None:
                if file_sha256(evidence["path"]) != evidence["sha256"]:
                    raise ValueError("Verified reference replay changed")
                report["expected_nll"] = evidence["cached_reference_nll"][arm]
                report["reference_replay_sha256"] = evidence["sha256"]
            report["reproduced"] = abs(report["overall"]["nll"] - report["expected_nll"]) <= REPRODUCTION_TOLERANCE
            reports[kind] = report
            write_json(output / f"{arm}_{kind}.json", report)
            result["arms"][arm][kind] = {key: report[key] for key in ("overall", "by_event", "by_entry_months",
                                                                      "primary_24_60", "expected_nll", "historical_saved_nll", "reproduced")}
            result["reproduced"] &= report["reproduced"]
            del model
        delta = paired_components(reports["reference"], reports["stabilized"])
        write_json(output / f"{arm}_paired_differences.json", delta)
        result["arms"][arm]["stabilized_minus_reference"] = delta["mean_difference"]
    write_json(output / "report.json", result)
    return result


def require_inspection(study):
    directory = study["output"] / "saved_checkpoint_inspection"
    # Always recompute from pinned weights before training. This also validates
    # the compatibility adapter and patient-balanced loss on the real cache.
    result = inspect_saved(study)
    if not result["reproduced"]:
        raise ValueError(f"Saved validation NLL reproduction failed; inspect {directory / 'report.json'} before training")


def train_candidate(study, arm, experiment):
    config = study["configs"][arm]
    output = study["output"] / arm / experiment
    identity = {"study_fingerprint": fingerprint(study["specification"]), "arm": arm,
                "experiment": experiment, "architecture": asdict(EXPERIMENTS[experiment])}
    if output.exists():
        summary_path = output / "summary.json"
        if not summary_path.exists():
            raise ValueError(f"Incomplete ablation at {output}; preserve it and choose a new --study name")
        summary = read_json(summary_path)
        if (summary.get("identity") != identity or summary["checkpoint_sha256"] != file_sha256(output / "best.pt")
            or summary.get("components_sha256") != file_sha256(output / "selected_components.json")
            or summary.get("differences_sha256") != file_sha256(output / "paired_differences.json")):
            raise ValueError("Existing ablation checkpoint or identity changed")
        return summary
    output.mkdir(parents=True)
    model = new_model(study, arm, experiment)
    train = [r for r in study["records"] if r.split == "train"]
    val = _loader([r for r in study["records"] if r.split == "validation"], config, arm, 1)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.learning_rate), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3)
    write_json(output / "parameter_counts.json", parameter_counts(model))
    write_json(output / "initial_components.json", validation_components(model, val, config.device))
    initial_weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _save_checkpoint(output / "initial.pt", {"model": initial_weights, "identity": identity,
                                             "config": asdict(config), "diagnostic_only": True})
    del initial_weights
    _seed(config.seed)
    best, best_epoch, stale, epochs = math.inf, 0, 0, []
    for epoch in range(1, config.epochs + 1):
        sampled = sample_epoch(train, config.seed, epoch)
        optimization = []
        learning_rate = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, _loader(sampled, config, arm, 1), optimizer, config.device,
                           config.gradient_clip_norm, diagnostics=optimization)
        value = validation_loss(model, val, config.device)
        with torch.random.fork_rng(devices=[]):
            components = validation_components(model, val, config.device)
        if abs(value - components["overall"]["nll"]) > REPRODUCTION_TOLERANCE:
            raise ValueError("Component aggregation disagrees with patient-balanced validation loss")
        scheduler.step(value)
        if value < best:
            best, best_epoch, stale = value, epoch, 0
        else:
            stale += 1
        epochs.append({"epoch": epoch, "train_nll": loss, "validation_nll": value, "learning_rate": learning_rate,
            "sample_fingerprint": fingerprint([(r.ptid, r.image_ids, r.prediction_months) for r in sampled]),
            "components": {key: components[key] for key in ("overall", "by_event", "by_entry_months")}})
        payload = {"model": model.state_dict(), "identity": identity, "config": asdict(config), "epoch": epoch,
            "best_epoch": best_epoch, "best_validation_nll": best, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "diagnostic_only": True}
        _save_checkpoint(output / "last.pt", payload)
        if best_epoch == epoch:
            _save_checkpoint(output / "best.pt", payload)
        write_json(output / "epochs.json", epochs)
        write_json(output / f"optimization_epoch_{epoch:03d}.json", optimization)
        LOGGER.info("%s/%s epoch %d: train %.6f, validation %.6f = H %.6f + event-log %.6f; selected %d",
            arm, experiment, epoch, loss, value, components["overall"]["cumulative_hazard"],
            components["overall"]["event_log_hazard"], best_epoch)
        if stale >= config.patience:
            break
    del payload, optimizer, scheduler
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    report = validation_components(model, val, config.device)
    report["primary_24_60"] = primary_metrics(model, study["records"], config)
    write_json(output / "selected_components.json", report)
    control = read_json(study["output"] / "saved_checkpoint_inspection" / f"{arm}_stabilized.json")
    reference = read_json(study["output"] / "saved_checkpoint_inspection" / f"{arm}_reference.json")
    differences = {"versus_stabilized": paired_components(control, report),
                   "versus_reference": paired_components(reference, report)}
    write_json(output / "paired_differences.json", differences)
    summary = {"identity": identity, "production_promoted": False, "best_epoch": best_epoch,
        "validation_nll": best, "components": report["overall"], "primary_24_60": report["primary_24_60"],
        "checkpoint_sha256": file_sha256(output / "best.pt"),
        "components_sha256": file_sha256(output / "selected_components.json"),
        "differences_sha256": file_sha256(output / "paired_differences.json"),
        **{key: delta["mean_difference"] for key, delta in differences.items()}}
    if experiment == "stabilized":
        summary["reproduced_saved_control"] = abs(best - study["sources"][arm]["stabilized_nll"]) <= REPRODUCTION_TOLERANCE
    write_json(output / "summary.json", summary)
    del model
    if str(config.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def write_overview(study):
    rows = []
    for arm in ("a", "b"):
        for name in EXPERIMENTS:
            path = study["output"] / arm / name / "summary.json"
            if path.exists():
                summary = read_json(path)
                rows.append({"arm": arm, "experiment": name, **summary})
    result = {"production_promoted": False, "selection_split": "validation", "seed": 42,
        "interpretation": "Single-factor reversals measure conditional effects; joint controls and single-seed results do not establish unique causation.",
        "results": rows}
    write_json(study["output"] / "report.json", result)
    lines = ["# Shared architecture ablations", "", result["interpretation"], "",
        "Loss differences are patient-balanced, candidate minus saved stabilized control. Negative NLL differences favor the candidate.", "",
        "| Arm | Experiment | Epoch | NLL | Delta H | Delta event-log | Delta NLL |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        d = row["versus_stabilized"]
        lines.append(f"| {row['arm']} | {row['experiment']} | {row['best_epoch']} | {row['validation_nll']:.6f} | "
                     f"{d['cumulative_hazard']:+.6f} | {d['event_log_hazard']:+.6f} | {d['nll']:+.6f} |")
    (study["output"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "train", "report"))
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--debug-run", type=debug_run_name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--study", type=study_name, default="architecture_seed_42")
    parser.add_argument("--reference-replay", type=Path,
                        help="Verified raw-batch/singleton/cache replay report; requires a new study name")
    parser.add_argument("--arms", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(2)
    study = prepare_study(args.workspace, args.protocol, args.debug_run, args.device, args.study, args.reference_replay)
    if args.stage == "inspect":
        result = inspect_saved(study)
        print(json.dumps(result, indent=2))
        if not result["reproduced"]:
            raise ValueError("Saved NLL reproduction failed; do not interpret architectural differences yet")
    elif args.stage == "train":
        require_inspection(study)
        for arm in dict.fromkeys(args.arms):
            control = train_candidate(study, arm, "stabilized")
            if not control["reproduced_saved_control"]:
                raise ValueError("Fresh stabilized control does not reproduce saved NLL; stop ablations and inspect its trace")
            for experiment in dict.fromkeys(args.experiments):
                if experiment == "stabilized" or (arm == "a" and experiment == "legacy_acquisition_time"):
                    continue
                train_candidate(study, arm, experiment)
                write_overview(study)
        result = write_overview(study)
        print(f"Ablation report: {study['output'] / 'report.md'} ({len(result['results'])} completed runs)")
    else:
        write_overview(study)
        print(study["output"] / "report.md")
    return study["output"]


if __name__ == "__main__":
    main()
