"""Diagnose reference NLL reproduction without changing ablation gates or caches."""

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from mstat_project.ml.ltsa_architecture_ablation import (
    REPRODUCTION_TOLERANCE, encoder_state_hash, load_saved_weights, new_model,
    paired_components, prepare_study, validation_components,
)
from mstat_project.ml.models import OrthogonalSliceResNet101Encoder
from mstat_project.ml.survival import _loader, _seed, validation_loss, write_json
from mstat_project.ml.survival_embeddings import fingerprint, state_fingerprint
from mstat_project.ml.survival_protocol import PROTOCOL_PATH, file_sha256


@torch.no_grad()
def replay_batches(encoder, raw_loader, cached_loader, device):
    """Hold observations fixed; compare original batches, singleton export, cache."""
    modes = [(module, module.training) for module in encoder.modules()]
    before = state_fingerprint(encoder)
    batches = {name: [] for name in ("raw_batch", "raw_singleton", "cached")}
    differences = []
    try:
        encoder.eval()
        for step, (raw, cached) in enumerate(zip(raw_loader, cached_loader, strict=True), 1):
            if raw["records"] != cached["records"] or any(r.split != "validation" for r in raw["records"]):
                raise ValueError("Replay requires identical validation observations")
            for name in ("times", "observed", "entry", "outcome", "event", "weight"):
                if not torch.equal(raw[name], cached[name]):
                    raise ValueError(f"Replay metadata mismatch: {name}")
            if raw["observed"].shape[1] != 1 or not raw["observed"].all():
                raise ValueError("Replay requires latest-image baseline inputs")
            images = raw["images"][:, 0].to(device)
            direct = encoder(images).detach().cpu()
            # Match export_embeddings' inference-mode, one-MRI encoder calls.
            with torch.inference_mode():
                single = torch.cat([encoder(image[None]).detach().cpu() for image in images])
            cache = cached["embeddings"][:, 0]
            if direct.shape != cache.shape or single.shape != cache.shape or not all(
                torch.isfinite(x).all() for x in (direct, single, cache)
            ):
                raise ValueError("Nonfinite or mismatched replay features")
            for name, values in (("raw_batch", direct), ("raw_singleton", single), ("cached", cache)):
                batches[name].append({**cached, "embeddings": values[:, None].clone()})
            for i, record in enumerate(raw["records"]):
                differences.append({"ptid": record.ptid, "image_id": record.image_ids[-1], **{
                    name: {"max_abs": float((left[i] - right[i]).abs().max()),
                           "rms": float((left[i].double() - right[i].double()).square().mean().sqrt())}
                    for name, left, right in (("batch_minus_cache", direct, cache),
                        ("singleton_minus_cache", single, cache), ("batch_minus_singleton", direct, single))}})
            if step == 1 or step % 25 == 0:
                print(f"Reference replay: {len(differences)} validation MRIs encoded", flush=True)
    finally:
        for module, training in modes:
            module.training = training
    if state_fingerprint(encoder) != before:
        raise RuntimeError("Replay changed encoder state")
    if not differences:
        raise ValueError("Empty validation replay")
    summary = {name: {"max_abs": max(row[name]["max_abs"] for row in differences),
                      "mean_image_rms": sum(row[name]["rms"] for row in differences) / len(differences)}
               for name in ("batch_minus_cache", "singleton_minus_cache", "batch_minus_singleton")}
    return batches, {"observations": len(differences), "summary": summary, "rows": differences}


def run_replay(study):
    output = study["output"] / ("reference_replay_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    config = study["configs"]["a"]
    _seed(config.seed)
    source = study["sources"]["a"]
    if file_sha256(source["reference_checkpoint"]) != source["reference_sha256"]:
        raise ValueError("Reference checkpoint changed")
    checkpoint = torch.load(source["reference_checkpoint"], map_location="cpu", weights_only=False)
    state = {key.removeprefix("encoder."): value for key, value in checkpoint["model"].items()
             if key.startswith("encoder.")}
    if encoder_state_hash(state) != study["manifest"]["encoder_sha256"]:
        raise ValueError("Reference encoder differs from cache")
    encoder = OrthogonalSliceResNet101Encoder(weights=None)
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).to(config.device).eval()
    del checkpoint, state
    records = [r for r in study["records"] if r.split == "validation"]
    raw_config = replace(config, cache_path=None)
    batches, features = replay_batches(encoder, _loader(records, raw_config, "a", 1),
                                       _loader(records, config, "a", 1), config.device)
    del encoder
    write_json(output / "feature_differences.json", features)
    result = {"diagnostic_only": True, "production_promoted": False,
        "study_fingerprint": fingerprint(study["specification"]),
        "replay_code_sha256": file_sha256(Path(__file__)), "batch_size": config.batch_size,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device": config.device, "feature_differences": features["summary"], "arms": {}}
    for arm in ("a", "b"):
        if study["configs"][arm].batch_size != config.batch_size:
            raise ValueError("Reference replay requires matched A/B validation batch sizes")
        source = study["sources"][arm]
        if file_sha256(source["reference_checkpoint"]) != source["reference_sha256"]:
            raise ValueError("Reference checkpoint changed")
        model = new_model(study, arm, "legacy_all")
        checkpoint = torch.load(source["reference_checkpoint"], map_location="cpu", weights_only=False)
        load_saved_weights(model, checkpoint, "reference", study["manifest"])
        del checkpoint
        reports, result["arms"][arm] = {}, {"expected_nll": source["reference_nll"]}
        for name, loader in batches.items():
            report = validation_components(model, loader, config.device)
            value = validation_loss(model, loader, config.device)
            report.update({"validation_nll_original_reduction": value,
                "difference_from_saved_nll": value - source["reference_nll"],
                "reproduced": abs(value - source["reference_nll"]) <= REPRODUCTION_TOLERANCE})
            reports[name] = report
            write_json(output / f"{arm}_{name}.json", report)
            result["arms"][arm][name] = {key: report[key] for key in (
                "overall", "validation_nll_original_reduction", "difference_from_saved_nll", "reproduced")}
        result["arms"][arm]["raw_batch_minus_cached"] = paired_components(
            reports["cached"], reports["raw_batch"])["mean_difference"]
        del model
    result["report"] = str(output / "report.json")
    write_json(output / "report.json", result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/artifacts/ltsa_v2"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--debug-run", required=True)
    parser.add_argument("--study", default="architecture_seed_42")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    study = prepare_study(args.workspace, args.protocol, args.debug_run, args.device, args.study)
    return run_replay(study)


if __name__ == "__main__":
    main()
