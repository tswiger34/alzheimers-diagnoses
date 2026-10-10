"""Detached optimizer diagnostics and constrained encoder fine-tuning."""

import math
from contextlib import contextmanager

import torch

from mstat_project.ml.survival_diagnostics import _summary


@contextmanager
def capture_training_activations(model, enabled):
    """Observe the actual forward pass without extra stochastic forwards."""
    stages, handles = {}, []
    def hook(name):
        def capture(module, args, result):
            stages[name] = _summary(result)
            if isinstance(module, torch.nn.Tanh):
                stages[name]["saturated_fraction_abs_ge_0_999"] = float((result.detach().abs() >= .999).float().mean())
        return capture
    try:
        if enabled:
            modules = {"normalized_head_input": model.head.input_norm,
                       "hidden_preactivation": model.head.hidden_linear,
                       "hidden_activation": model.head.activation}
            if model.transformer is not None:
                modules["temporal_features"] = model.temporal_norm
            modules.update({f"tanh:{name}": m for name, m in model.named_modules() if isinstance(m, torch.nn.Tanh)})
            handles = [m.register_forward_hook(hook(name)) for name, m in modules.items()]
        yield stages
    finally:
        for handle in handles:
            handle.remove()


def parameter_group(name):
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("projection"):
        return "projection"
    if "self_attn" in name:
        return "attention"
    if "transformer" in name and ("linear1" in name or "linear2" in name):
        return "feed_forward"
    if name.startswith("head."):
        return "survival_head"
    if name == "alpha":
        return "residual_gate"
    if name.startswith("time_embedding"):
        return "time_encoding"
    if name.startswith("step_predictor"):
        return "step_ahead"
    return "temporal_other"


def parameter_counts(model):
    groups = {}
    for name, value in model.named_parameters():
        item = groups.setdefault(parameter_group(name), {"total": 0, "trainable": 0})
        item["total"] += value.numel()
        item["trainable"] += value.numel() if value.requires_grad else 0
    if hasattr(model.encoder, "cached_parameters"):
        groups["encoder"] = {"total": model.encoder.cached_parameters, "trainable": 0}
    return groups


def parameter_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}


def norms(model, *, before=None):
    groups = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        delta = value.detach() - before[name] if before is not None else value.grad
        group = parameter_group(name)
        groups.setdefault(group, 0.)
        if delta is not None:
            groups[group] += float(delta.detach().double().square().sum())
    return {name: math.sqrt(value) for name, value in groups.items()}


def optimization_summary(model, batch, z, cumulative, event_log, auxiliary):
    with torch.no_grad():
        hazard = model.head.hazard(z, batch["outcome"])
        result = {"head_input": _summary(z), "hazard": _summary(hazard),
                  "cumulative_hazard": _summary(cumulative),
                  "nll_cumulative_mean": float(cumulative.mean()),
                  "nll_event_log_mean": float(event_log.mean()),
                  "auxiliary_loss": float(auxiliary),
                  "events": int(batch["event"].sum()), "observations": len(batch["event"]),
                  "event_proportion": float(batch["event"].mean())}
        return result


def configure_final_block(model):
    if not hasattr(model.encoder, "model") or not hasattr(model.encoder.model, "layer4"):
        raise ValueError("Final-block fine-tuning requires the raw ResNet encoder")
    model.freeze_encoder = False
    model.fine_tune_final_block = True
    model.encoder.requires_grad_(False)
    model.encoder.model.layer4.requires_grad_(True)


def optimizer_groups(model, learning_rate, encoder_learning_rate=1e-5):
    regular, encoder = [], []
    for name, value in model.named_parameters():
        if value.requires_grad:
            (encoder if name.startswith("encoder.") else regular).append(value)
    groups = [{"params": regular, "lr": learning_rate}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_learning_rate})
    return groups
