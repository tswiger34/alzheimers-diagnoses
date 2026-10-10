"""Shared neural continuous hazard, conditional likelihood and MRI model variants.

All times are continuous months on the baseline time axis. Integration uses
Gauss-Legendre quadrature, unrelated to the visit-bin schedule.
"""

import math
from collections import Counter

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ARCHITECTURE_VERSION = 2
TIME_TRANSFORM = "t/(t+180)"


def bounded_time(time: Tensor) -> Tensor:
    if not torch.isfinite(time).all() or (time < 0).any():
        raise ValueError("Model times must be finite and nonnegative")
    return time / (time + 180.0)


def initial_hazard(records) -> float:
    """Constant-rate MLE under the patient-balanced conditional objective."""
    train = [r for r in records if r.split == "train"]
    counts = Counter(r.ptid for r in train)
    events = sum(float(r.event) / counts[r.ptid] for r in train)
    exposure = sum((r.outcome_months - r.prediction_months) / counts[r.ptid] for r in train)
    if not train or exposure <= 0 or events <= 0:
        raise ValueError("Hazard initialization requires training events and positive exposure")
    return max(events / exposure, 1e-6)


class ContinuousHazardHead(nn.Module):
    """Positive monthly hazard lambda(u | z)"""

    def __init__(self, features: int, quadrature_points: int = 32, baseline_hazard: float = 0.002):
        super().__init__()
        if quadrature_points < 8:
            raise ValueError("Use at least 8 integration points")
        if not math.isfinite(baseline_hazard) or baseline_hazard <= 1e-7 / 12:
            raise ValueError("baseline_hazard must exceed the positive monthly hazard floor")
        self.input_norm = nn.LayerNorm(features + 1)
        self.network = nn.Sequential(nn.Linear(features + 1, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(self.network[2].weight, std=1e-3)
        rate = 12 * baseline_hazard - 1e-7
        nn.init.constant_(self.network[2].bias, rate + math.log(-math.expm1(-rate)))
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        self.register_buffer("nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def hazard(self, z: Tensor, time: Tensor) -> Tensor:
        if time.ndim == 1:
            time = time[:, None]
        expanded = z[:, None, :].expand(-1, time.shape[1], -1)
        inputs = torch.cat((expanded, bounded_time(time)[..., None]), -1)
        return (F.softplus(self.network(self.input_norm(inputs))).squeeze(-1) + 1e-7) / 12

    @property
    def hidden_linear(self):
        return self.network[0]

    @property
    def activation(self):
        return self.network[1]

    @property
    def output_linear(self):
        return self.network[2]

    def cumulative_hazard(self, z: Tensor, start: Tensor, end: Tensor) -> Tensor:
        if (
            not torch.isfinite(start).all()
            or not torch.isfinite(end).all()
            or (start < 0).any()
            or (end < start).any()
        ):
            raise ValueError("Integration requires finite 0 <= start <= end")
        width = (end - start) / 2
        times = start[:, None] + width[:, None] * (self.nodes[None, :] + 1)
        return width * (self.hazard(z, times) * self.weights).sum(-1)

    def nll_components(self, z: Tensor, entry: Tensor, outcome: Tensor, event: Tensor):
        if (outcome <= entry).any() or not ((event == 0) | (event == 1)).all():
            raise ValueError("Likelihood requires positive future follow-up and binary events")
        cumulative = self.cumulative_hazard(z, entry, outcome)
        event_log = -event * self.hazard(z, outcome).squeeze(-1).log()
        return cumulative, event_log

    def nll(self, z: Tensor, entry: Tensor, outcome: Tensor, event: Tensor) -> Tensor:
        cumulative, event_log = self.nll_components(z, entry, outcome, event)
        return cumulative + event_log

    def survival(self, z: Tensor, entry: Tensor, end: Tensor) -> Tensor:
        return torch.exp(-self.cumulative_hazard(z, entry, end))


class ContinuousMRIModel(nn.Module):
    """A: image only; B: image plus acquisition time; history: matched transformer.

    The hazard query u is the survival time variable in every variant. Baseline
    A receives neither s, nominal visit nor history in its image representation.
    B and history share the same acquisition-time embedding. Padding tokens are
    masked and never encoded as images. Histories contain only the available prefix.
    """

    def __init__(self, encoder, variant="a", features=128, quadrature_points=32, freeze_encoder=False,
                 *, baseline_hazard=0.002, n_layers=1, d_model=128, dropout=0.1, step_ahead=False):
        super().__init__()
        if variant not in ("a", "b", "history"):
            raise ValueError("variant must be a, b or history")
        self.variant = variant
        self.register_buffer("_architecture_version", torch.tensor(ARCHITECTURE_VERSION))
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.projection = nn.Linear(encoder.n_features, features)
        self.projection_norm = nn.LayerNorm(features)
        # Construct common components in identical order for matched initial weights.
        self.time_embedding = nn.Linear(1, features)
        if variant == "a":
            self.time_embedding.requires_grad_(False)
        self.head = ContinuousHazardHead(features, quadrature_points, baseline_hazard)
        self.transformer = None
        self.step_predictor = None
        if variant == "history":
            if d_model % 4 or n_layers < 1 or not 0 <= dropout < 1:
                raise ValueError("Invalid temporal width, depth or dropout")
            self.temporal_input = nn.Linear(features, d_model) if d_model != features else nn.Identity()
            layer = nn.TransformerEncoderLayer(d_model, 4, d_model * 2, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.temporal_output = nn.Linear(d_model, features) if d_model != features else nn.Identity()
            self.temporal_norm = nn.LayerNorm(features)
            self.alpha = nn.Parameter(torch.tensor(0.01))
            if step_ahead:
                self.step_predictor = nn.Sequential(nn.Linear(d_model + 1, 128), nn.GELU(),
                                                    nn.Linear(128, encoder.n_features))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        version = state_dict.get("_architecture_version")
        if version is None or int(version) != ARCHITECTURE_VERSION:
            raise ValueError("Incompatible continuous architecture; retrain a version-2 baseline before transfer")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def train(self, mode=True):
        """Keep frozen encoder parameters and running statistics fixed."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        if getattr(self, "fine_tune_final_block", False):
            self.encoder.eval()
            self.encoder.model.layer4.train(mode)
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def raw_features(self, images, observed):
        values = self.encoder(images[observed])
        raw = values.new_zeros((*observed.shape, values.shape[-1]))
        raw[observed] = values
        return raw

    def representations(self, raw: Tensor, times: Tensor, observed: Tensor):
        """Return baseline tokens and causal states; missing slots never enter attention."""
        if observed.dtype != torch.bool or raw.shape[:2] != observed.shape or times.shape != observed.shape:
            raise ValueError("Feature, time and boolean mask shapes must agree")
        if not observed.any(dim=1).all():
            raise ValueError("Each patient requires at least one observed MRI")
        for row, mask in zip(times, observed, strict=True):
            selected = row[mask]
            if (selected[1:] < selected[:-1]).any():
                raise ValueError("Observed acquisition times must be chronological")
        values = self.projection_norm(self.projection(raw[observed]))
        if self.variant != "a":
            values = values + self.time_embedding(bounded_time(times[observed])[:, None])
        tokens = values.new_zeros((*observed.shape, values.shape[-1]))
        tokens[observed] = values
        if self.transformer is None:
            return tokens, None
        # Compact each row only for attention. Scatter back to original nominal slots.
        # This avoids fully masked queries (including leading gaps) and their NaNs.
        temporal = []
        for row, mask in zip(tokens, observed, strict=True):
            x = self.temporal_input(row[mask])[None]
            causal = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).triu(1)
            temporal.append(self.transformer(x, mask=causal)[0])
        states = values.new_zeros((*observed.shape, temporal[0].shape[-1]))
        states[observed] = torch.cat(temporal)
        return tokens, states

    def encode_features(self, raw: Tensor, times: Tensor, observed: Tensor) -> Tensor:
        tokens, states = self.representations(raw, times, observed)
        last = torch.arange(observed.shape[1], device=observed.device)[None, :].expand_as(observed)
        last = last.masked_fill(~observed, -1).max(dim=1).values
        rows = torch.arange(raw.shape[0], device=raw.device)
        z = tokens[rows, last]
        if states is not None:
            z = z + self.alpha * self.temporal_norm(self.temporal_output(states[rows, last]))
        return z

    def encode(self, images: Tensor, times: Tensor, observed: Tensor) -> Tensor:
        if self.variant != "history":
            if not observed.any(dim=1).all():
                raise ValueError("Each patient requires at least one observed MRI")
            slots = torch.arange(observed.shape[1], device=observed.device).expand_as(observed)
            last = slots.masked_fill(~observed, -1).max(1).values
            rows = torch.arange(len(images), device=images.device)
            values = self.encoder(images[rows, last])[:, None]
            return self.encode_features(values, times[rows, last, None],
                                        torch.ones((len(images), 1), dtype=torch.bool, device=images.device))
        return self.encode_features(self.raw_features(images, observed), times, observed)

    def encode_batch(self, batch):
        if "embeddings" in batch:
            if any(p.requires_grad for p in self.encoder.parameters()):
                raise ValueError("Cached inputs cannot train an image encoder")
            return self.encode_features(batch["embeddings"], batch["times"], batch["observed"])
        return self.encode(batch["images"], batch["times"], batch["observed"])

    def auxiliary_loss(self, batch):
        if self.step_predictor is None:
            return self.projection.weight.sum() * 0
        raw = batch.get("embeddings")
        if raw is None:
            raw = self.raw_features(batch["images"], batch["observed"])
        _, states = self.representations(raw, batch["times"], batch["observed"])
        losses = []
        for i, mask in enumerate(batch["observed"]):
            x, targets, times = states[i, mask], raw[i, mask].detach(), batch["times"][i, mask]
            if len(times) > 1:
                elapsed = bounded_time(times[1:] - times[:-1])[:, None]
                prediction = self.step_predictor(torch.cat((x[:-1], elapsed), -1))
                losses.append((prediction - targets[1:]).square().mean())
        return torch.stack(losses).mean() if losses else states.sum() * 0

    def forward(self, batch):
        z = self.encode_batch(batch)
        return self.head.nll(z, batch["entry"], batch["outcome"], batch["event"])
