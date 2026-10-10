"""Small, fixed-panel forward traces; no gradients, checkpoints or image cache."""

import hashlib

import torch
from torch.nn import functional as F

from mstat_project.ml.survival_data import SurvivalDataset, collate_survival


def select_diagnostic_panel(records, seed, *, latest=False):
    """Choose one distinct patient per train/validation x event/censor stratum.

    Use each patient's earliest eligible MRI (a one-image prefix, including for
    history models). Selection is independent of input order and model variant.
    Missing strata stay absent rather than being filled with test patients.
    """
    earliest = {}
    for record in records:
        if record.split not in ("train", "validation"):
            continue
        key = (record.acquisition_months[-1], record.image_ids[-1])
        previous = earliest.get(record.ptid)
        if previous is None or ((key > (previous.acquisition_months[-1], previous.image_ids[-1])) if latest
                                else (key < (previous.acquisition_months[-1], previous.image_ids[-1]))):
            earliest[record.ptid] = record
    panel = []
    for split in ("train", "validation"):
        for event in (True, False):
            candidates = [r for r in earliest.values() if r.split == split and r.event == event]
            if candidates:
                panel.append(
                    min(
                        candidates,
                        key=lambda r: (
                            hashlib.sha256(f"{seed}:diagnostics:{r.ptid}".encode()).hexdigest(),
                            r.ptid,
                        ),
                    )
                )
    return panel


def _summary(value):
    """Summarize one tensor without retaining it or emitting non-JSON NaNs."""
    value = value.detach().float()
    finite = torch.isfinite(value)
    result = {"shape": list(value.shape), "finite_fraction": float(finite.float().mean())}
    if not bool(finite.all()):
        return {**result, "min": None, "max": None, "mean": None, "std": None, "rms": None}
    low, high = float(value.min()), float(value.max())
    scale = max(abs(low), abs(high)) or 1.0
    scaled = value / scale
    return {
        **result,
        "min": low,
        "max": high,
        "mean": float(scaled.mean()) * scale,
        "std": float(scaled.std(unbiased=False)) * scale,
        "rms": float(scaled.square().mean().sqrt()) * scale,
    }


def _capture(stages, vectors, name, value, *, image=False):
    stages[name] = _summary(value)
    if image:
        # Retain only spatial sketches, never full images. These are proxies for
        # between-patient variation, not proof that full images are identical.
        value = value[-1:]
        value = (
            F.adaptive_avg_pool3d(value.float(), (4, 4, 4))
            if value.ndim == 5
            else F.adaptive_avg_pool2d(value.float(), (8, 8))
        )
    if name in ("encoder_features", "projected_features", "temporal_features"):
        value = value.reshape(-1, value.shape[-1])[-1:]
    vectors[name] = value.detach().flatten().to(device="cpu", dtype=torch.float64).clone()


def _stage_hook(stages, vectors, name, *, image=False, capture_input=False):
    """Bind each hook to one patient's small result dictionaries."""

    def capture(_module, args, result=None):
        value = args[0] if capture_input else result
        _capture(stages, vectors, name, value, image=image)
        if name == "tanh":
            finite = stages[name]["finite_fraction"] == 1
            stages[name]["saturated_fraction_abs_ge_0_999"] = (
                float((value.abs() >= 0.999).float().mean()) if finite else None
            )
            stages[name]["mean_local_derivative"] = float((1 - value.square()).mean()) if finite else None

    return capture


def _panel_variation(vectors, name):
    values = torch.stack([v[name] for v in vectors if name in v])
    image = name in ("loaded_volume", "preprocessed_volume", "normalized_input")
    result = {
        "patients": len(values),
        "coordinates": values.shape[1],
        "comparison_basis": "pooled_image_sketch" if image else "full_feature_vector",
    }
    if not torch.isfinite(values).all() or len(values) < 2:
        return {**result, "near_constant": None, "mean_coordinate_std": None, "max_coordinate_range": None}
    spread = values.max(0).values - values.min(0).values
    tolerance = 1e-8 + 1e-6 * values.abs().max(0).values
    return {
        **result,
        "near_constant": bool((spread <= tolerance).all()),
        "mean_coordinate_std": float(values.std(0, unbiased=False).mean()),
        "max_coordinate_range": float(spread.max()),
    }


@torch.inference_mode()
def trace_diagnostic_panel(model, panel, *, spatial_size, device, horizons, cache_path=None):
    """Trace one MRI at a time through the real forward path and summarize it.

    Restore every module's mode and remove hooks even on failure. Common query
    times separate across-patient embedding variation from within-patient time
    variation. These probes are descriptive, not outcome-performance estimates.
    """
    if not panel:
        return {"patients": [], "warnings": ["No training/validation diagnostic patients available."]}
    modes = [(module, module.training) for module in model.modules()]
    anchor = max(r.acquisition_months[-1] for r in panel)
    query_times = [anchor, *(anchor + h for h in horizons)]
    patients, vectors, warnings = [], [], []
    if len(panel) < 4:
        warnings.append(
            "Panel has fewer than four patients: some train/validation event/censor strata are absent."
        )
    model.eval()
    try:
        for record in panel:
            stages, row_vectors, handles = {}, {}, []
            # Memory-map the package, summarize only the selected raw image,
            # and release it before loading the preprocessed image.
            if cache_path is None:
                package = torch.load(record.tensor_path, map_location="cpu", weights_only=True, mmap=True)
                ids = [str(i) for i in package["img_ids"]]
                index = ids.index(record.image_ids[-1])
                _capture(stages, row_vectors, "loaded_volume", package["images"][index : index + 1], image=True)
                del package
                dataset = SurvivalDataset(
                    [record], history=model.variant == "history", n_slots=1, spatial_size=spatial_size
                )
                batch = collate_survival([dataset[0]])
                _capture(stages, row_vectors, "preprocessed_volume",
                         batch["images"][batch["observed"]][-1:], image=True)
            else:
                from mstat_project.ml.survival_embeddings import EmbeddingDataset, collate_embeddings
                dataset = EmbeddingDataset([record], cache_path=cache_path, history=model.variant == "history",
                                           spatial_size=spatial_size)
                batch = collate_embeddings([dataset[0]])
                _capture(stages, row_vectors, "encoder_features", batch["embeddings"][batch["observed"]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                handles.append(
                    model.encoder.register_forward_hook(_stage_hook(stages, row_vectors, "encoder_features"))
                )
                handles.append(
                    model.projection_norm.register_forward_hook(_stage_hook(stages, row_vectors, "projected_features"))
                )
                if hasattr(model.encoder, "volume_to_resnet_input"):
                    handles.append(
                        model.encoder.model.register_forward_pre_hook(
                            _stage_hook(stages, row_vectors, "normalized_input", image=True, capture_input=True)
                        )
                    )
                if model.transformer is not None:
                    handles.append(model.temporal_norm.register_forward_hook(
                        _stage_hook(stages, row_vectors, "temporal_features")))
                z = model.encode_batch(batch)
            finally:
                for handle in handles:
                    handle.remove()
            _capture(stages, row_vectors, "head_embedding", z)
            times = z.new_tensor([query_times])

            handles = [
                model.head.hidden_linear.register_forward_hook(_stage_hook(stages, row_vectors, "hidden_preactivation")),
                model.head.activation.register_forward_hook(_stage_hook(stages, row_vectors,
                    "tanh" if isinstance(model.head.activation, torch.nn.Tanh) else "hidden_activation")),
            ]
            try:
                hazard = model.head.hazard(z, times)
            finally:
                for handle in handles:
                    handle.remove()
            _capture(stages, row_vectors, "hazard", hazard)
            start = z.new_tensor([anchor])
            survival = torch.stack([model.head.survival(z, start, start + h) for h in horizons], dim=1)
            cumulative = torch.stack([model.head.cumulative_hazard(z, start, start + h) for h in horizons], dim=1)
            _capture(stages, row_vectors, "cumulative_hazard", cumulative)
            _capture(stages, row_vectors, "survival", survival)
            _capture(stages, row_vectors, "risk", 1-survival)
            hazard_values = row_vectors["hazard"]
            temporal_range = float(hazard_values.max() - hazard_values.min())
            temporal_finite = bool(torch.isfinite(hazard_values).all())
            patients.append(
                {
                    "ptid": record.ptid,
                    "split": record.split,
                    "event": record.event,
                    "image_id": record.image_ids[-1],
                    "acquisition_months": record.acquisition_months[-1],
                    "stages": stages,
                    "hazard_per_month": [float(v) if torch.isfinite(v) else None for v in hazard_values],
                    "survival": [float(v) if torch.isfinite(v) else None for v in row_vectors["survival"]],
                    "hazard_range_across_times": temporal_range if temporal_finite else None,
                    "hazard_near_constant_across_times": (
                        temporal_range <= 1e-8 + 1e-6 * float(hazard_values.abs().max())
                        if temporal_finite
                        else None
                    ),
                }
            )
            vectors.append(row_vectors)
            del batch, dataset, z, times, hazard, survival
    finally:
        for module, training in modes:
            module.training = training
    order = (
        "loaded_volume",
        "preprocessed_volume",
        "normalized_input",
        "encoder_features",
        "projected_features",
        "head_embedding",
        "temporal_features",
        "hidden_preactivation",
        "hidden_activation",
        "tanh",
        "hazard",
        "cumulative_hazard",
        "survival",
        "risk",
    )
    variation = {name: _panel_variation(vectors, name) for name in order if all(name in v for v in vectors)}
    first = next((name for name in order if name in variation and variation[name]["near_constant"]), None)
    if first:
        warnings.append(
            f"First near-constant panel stage: {first}; small-panel screening, not a causal diagnosis."
        )
    if any((p["stages"].get("tanh", {}).get("saturated_fraction_abs_ge_0_999") or 0) > 0.95 for p in patients):
        warnings.append("Over 95% of Tanh activations are saturated for at least one panel patient.")
    if any(s["finite_fraction"] < 1 for p in patients for s in p["stages"].values()):
        warnings.append("Nonfinite activations detected in the diagnostic panel.")
    if "normalized_input" not in variation and cache_path is None:
        warnings.append("Normalized input capture unavailable for this custom encoder.")
    risk = torch.stack([v["risk"] for v in vectors])
    risk_variance = risk.var(dim=0, unbiased=False).tolist() if torch.isfinite(risk).all() else [None] * len(horizons)
    if variation.get("risk", {}).get("near_constant"):
        warnings.append("Risk predictions are near constant across the fixed panel.")
    return {
        "patients": patients,
        "panel_variation": variation,
        "first_near_constant_stage": first,
        "hazard_query_months": query_times,
        "survival_conditioning_months": anchor,
        "survival_horizons_months": list(horizons),
        "risk_variance_by_horizon": risk_variance,
        "warnings": warnings,
        "near_constant_tolerance": {"absolute": 1e-8, "relative": 1e-6},
        "notes": [
            "Images are compared using pooled sketches; feature vectors use all coordinates.",
            "Fixed eligible MRI prefixes; history traces preserve observed visits and gaps.",
            "Common query times are diagnostic probes, not clinical evaluation landmarks.",
            "Across-patient differences include acquisition-time embeddings in B/history.",
        ],
    }
