"""Versioned raw encoder embeddings with explicit observation provenance."""

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from mstat_project.ml.survival_data import SurvivalDataset, collate_survival

CACHE_VERSION = 2


def preprocessing_fingerprint():
    directory = Path(__file__).parent
    return fingerprint({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                        for name in ("models.py", "landmarks.py")})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def image_metadata(record, index=-1):
    return {"image_id": record.image_ids[index], "ptid": record.ptid, "split": record.split,
            "acquisition_months": record.acquisition_months[index], "visit_num": record.visit_nums[index],
            "observed": True, "outcome_months": record.outcome_months, "event": record.event}


def validate_metadata(row):
    s, t = row["acquisition_months"], row["outcome_months"]
    if not (0 <= s < t < float("inf")) or row["observed"] is not True:
        raise ValueError("Cached MRI must be observed and strictly precede the clinical outcome")
    if row["split"] not in ("train", "validation", "test") or row["visit_num"] < 0:
        raise ValueError("Invalid cached split or nominal visit")
    if type(row["event"]) is not bool or not row["image_id"] or not row["ptid"]:
        raise ValueError("Cached image/patient IDs and boolean event indicators are required")


@torch.inference_mode()
def export_embeddings(path, records, encoder, *, spatial_size, provenance, device="cpu"):
    """Atomically export each eligible MRI exactly once; never mutate encoder state."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not records:
        raise ValueError("Cannot export an empty embedding cache")
    unique = {}
    for record in records:
        for i, image_id in enumerate(record.image_ids):
            row = image_metadata(record, i)
            validate_metadata(row)
            if image_id in unique and unique[image_id] != row:
                raise ValueError("Conflicting cache metadata for image ID")
            unique[image_id] = row
    # The source has one prefix ending at each image. This guarantees identical
    # preprocessing to the latest-image baseline, without duplicating histories.
    terminal = {r.image_ids[-1]: r for r in records}
    if set(terminal) != set(unique):
        from dataclasses import replace
        for r in records:
            for i, image_id in enumerate(r.image_ids):
                terminal.setdefault(image_id, replace(r, image_ids=r.image_ids[:i+1],
                    acquisition_months=r.acquisition_months[:i+1], visit_nums=r.visit_nums[:i+1],
                    prediction_months=r.acquisition_months[i]))
    ordered = [terminal[key] for key in sorted(unique)]
    dataset = SurvivalDataset(ordered, history=False, n_slots=1, spatial_size=spatial_size)
    modes = [(module, module.training) for module in encoder.modules()]
    encoder.to(device).eval()
    encoder_hash = state_fingerprint(encoder)
    values = []
    try:
        for i in range(len(dataset)):
            sample = dataset[i]
            value = encoder(sample["images"].to(device)).detach().cpu().float()
            if value.shape != (1, encoder.n_features) or not torch.isfinite(value).all():
                raise ValueError("Invalid encoder output while building cache")
            values.append(value[0])
            if (i + 1) % 100 == 0:
                print(f"Cached {i + 1}/{len(dataset)} MRIs", flush=True)
    finally:
        for module, mode in modes:
            module.training = mode
    if state_fingerprint(encoder) != encoder_hash:
        raise RuntimeError("Encoder state changed during cache export")
    metadata = [unique[key] for key in sorted(unique)]
    embeddings = torch.stack(values)
    manifest = {"cache_version": CACHE_VERSION, "n_features": encoder.n_features,
                "encoder_sha256": encoder_hash, "spatial_size": list(spatial_size),
                "preprocessing": "per-volume-standardize/orthogonal-center-slices/ImageNet-v1",
                "preprocessing_sha256": preprocessing_fingerprint(),
                "provenance": provenance, "metadata_sha256": fingerprint(metadata),
                "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
                "embeddings_sha256": hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()}
    manifest["cache_id"] = fingerprint(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"manifest": manifest, "metadata": metadata, "embeddings": embeddings}, temporary)
    temporary.replace(path)
    return manifest


@lru_cache(maxsize=2)
def _read_cache(path, modified, size):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    manifest, rows, values = payload["manifest"], payload["metadata"], payload["embeddings"]
    if manifest["cache_version"] != CACHE_VERSION:
        raise ValueError("Unsupported embedding cache version; regenerate cache")
    if manifest.get("preprocessing_sha256") != preprocessing_fingerprint():
        raise ValueError("Cache preprocessing implementation changed; regenerate cache")
    if fingerprint({k: v for k, v in manifest.items() if k != "cache_id"}) != manifest["cache_id"]:
        raise ValueError("Cache manifest fingerprint mismatch")
    if fingerprint(rows) != manifest["metadata_sha256"]:
        raise ValueError("Cache metadata fingerprint mismatch")
    if values.shape != (len(rows), manifest["n_features"]) or not torch.isfinite(values).all():
        raise ValueError("Cache contains missing/nonfinite/wrong-shape features")
    if hashlib.sha256(values.numpy().tobytes()).hexdigest() != manifest["embeddings_sha256"]:
        raise ValueError("Cache embedding fingerprint mismatch")
    index = {row["image_id"]: i for i, row in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError("Duplicate image IDs in embedding cache")
    for row in rows:
        validate_metadata(row)
    return payload, index


def load_embeddings(path, *, provenance=None, spatial_size=None):
    path = Path(path).resolve()
    stat = path.stat()
    payload, index = _read_cache(str(path), stat.st_mtime_ns, stat.st_size)
    manifest = payload["manifest"]
    if provenance is not None and manifest["provenance"] != provenance:
        raise ValueError("Cache source/split/schedule fingerprint mismatch")
    if spatial_size is not None and manifest["spatial_size"] != list(spatial_size):
        raise ValueError("Cache preprocessing spatial size mismatch")
    return payload, index


class CachedEncoder(nn.Module):
    """Dimension-only encoder: attempting image inference is an error."""

    def __init__(self, manifest):
        super().__init__()
        self.n_features = manifest["n_features"]
        self.cached_parameters = manifest["encoder_parameters"]

    def forward(self, images):
        raise RuntimeError("CachedEncoder requires embeddings, not raw images")


class EmbeddingDataset(Dataset):
    def __init__(self, records, *, cache_path, history, spatial_size):
        self.records, self.history = records, history
        self.payload, self.index = load_embeddings(cache_path, spatial_size=spatial_size)
        self.counts = Counter(r.ptid for r in records)
        for r in records:
            if (not r.image_ids or not (len(r.image_ids) == len(r.acquisition_months) == len(r.visit_nums))
                or not r.acquisition_months[-1] <= r.prediction_months < r.outcome_months
                or tuple(sorted(set(r.visit_nums))) != r.visit_nums
                or tuple(sorted(r.acquisition_months)) != r.acquisition_months):
                raise ValueError("Invalid prediction prefix, chronology or nominal visit alignment")
            for i, image_id in enumerate(r.image_ids):
                if image_id not in self.index or self.payload["metadata"][self.index[image_id]] != image_metadata(r, i):
                    raise ValueError(f"Cache missing or inconsistent MRI {image_id}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        ids = r.image_ids if self.history else r.image_ids[-1:]
        return {"images": self.payload["embeddings"][[self.index[key] for key in ids]],
                "slots": r.visit_nums if self.history else (0,),
                "times": r.acquisition_months if self.history else r.acquisition_months[-1:],
                "entry": r.prediction_months, "outcome": r.outcome_months, "event": r.event,
                "weight": 1 / self.counts[r.ptid], "record": r}


def collate_embeddings(samples):
    batch = collate_survival(samples)
    batch["embeddings"] = batch.pop("images")
    return batch
