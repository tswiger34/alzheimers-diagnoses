"""Auditable MRI landmarks shared by continuous single-image and history models.

Database outcomes are authoritative; labels stored in older tensor packages are
never used. Missing visits stay absent, and tensor selection is by image ID.
"""

import hashlib
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import torch
from sqlalchemy import Engine
from torch.utils.data import Dataset

from mstat_project.ml.landmarks import LandmarkPatientRecord, LandmarkSequenceDataset

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Observation:
    ptid: str
    split: str
    image_ids: tuple[str, ...]
    acquisition_months: tuple[float, ...]
    visit_nums: tuple[int, ...]
    prediction_months: float
    outcome_months: float
    event: bool
    tensor_path: Path


def load_source(engine: Engine) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the clinical landmark table, canonical bins and existing splits atomically."""
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as c:
        frame = pl.read_database("SELECT * FROM _core.core_survival_landmarks ORDER BY ptid, visit_num", c)
        schedule = pl.read_database("SELECT * FROM _core.core_follow_up_buckets ORDER BY visit_num", c)
        splits = pl.read_database("SELECT ptid, split::text AS split FROM _raw.train_test_split ORDER BY ptid", c)
    return frame, schedule, splits


def build_observations(
    frame: pl.DataFrame, schedule: pl.DataFrame, splits: pl.DataFrame, tensor_dir: Path
) -> tuple[list[Observation], dict]:
    """Build one chronological prefix per eligible MRI; never compact visit codes."""
    required = {
        "ptid", "image_id", "visit_num", "visit_code", "binning_months", "acquisition_months",
        "outcome_months", "event_observed", "exclusion_reason",
    }
    if frame.is_empty() or not required.issubset(frame.columns):
        raise ValueError("Empty or incomplete clinical landmark source")
    if splits["ptid"].is_duplicated().any() or splits.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError("Patient split assignments must be unique and non-null")
    if set(splits["split"].to_list()) != set(SPLITS):
        raise ValueError("Expected train, validation and test splits")
    if schedule["visit_num"].to_list() != list(range(schedule.height)) or schedule["visit_code"].is_duplicated().any():
        raise ValueError("Canonical schedule must have unique codes and consecutive slots starting at BL=0")
    if schedule.row(0, named=True)["visit_code"] != "BL":
        raise ValueError("First canonical slot must be BL")
    targets = [float(v) for v in schedule["follow_up_months"]]
    if any(not math.isfinite(v) or v < 0 for v in targets) or targets != sorted(set(targets)):
        raise ValueError("Canonical target months must be finite, unique and increasing")
    if frame.select("ptid", "visit_num").is_duplicated().any() or frame["image_id"].is_duplicated().any():
        raise ValueError("Expected one image per patient/bin and unique image IDs")
    bins = {int(r["visit_num"]): r for r in schedule.iter_rows(named=True)}
    for row in frame.iter_rows(named=True):
        b = bins.get(row["visit_num"])
        s = float(row["acquisition_months"])
        rounded = float(row["binning_months"])
        if b is None or row["visit_code"] != b["visit_code"] or not math.isfinite(s) or s < 0:
            raise ValueError("Image has invalid canonical bin or acquisition time")
        # dbt assigns bins with rounded months, while eligibility uses unrounded days.
        if (row["visit_num"] == 0 and s != 0) or (
            row["visit_num"] != 0 and not (s > 0 and float(b["follow_up_start"]) <= rounded < float(b["follow_up_end"]))
        ):
            raise ValueError("Image does not belong to its canonical bin")
    assigned = frame.join(splits, on="ptid", how="inner", validate="m:1")
    eligible = assigned.filter(pl.col("exclusion_reason").is_null()).sort("ptid", "acquisition_months", "image_id")
    records = []
    for group in eligible.partition_by("ptid", maintain_order=True):
        rows = group.to_dicts()
        if group["outcome_months"].n_unique() != 1 or group["event_observed"].n_unique() != 1:
            raise ValueError("Inconsistent clinical outcomes within patient")
        outcome = float(rows[0]["outcome_months"])
        if not math.isfinite(outcome) or outcome <= 0 or rows[0]["event_observed"] is None:
            raise ValueError("Missing or invalid clinical outcome")
        for end, row in enumerate(rows, start=1):
            prefix = rows[:end]
            s = float(row["acquisition_months"])
            if not 0 <= s < outcome:
                raise ValueError("MRI must strictly precede the event/censoring time")
            records.append(Observation(
                ptid=row["ptid"], split=row["split"], image_ids=tuple(str(r["image_id"]) for r in prefix),
                acquisition_months=tuple(float(r["acquisition_months"]) for r in prefix),
                visit_nums=tuple(int(r["visit_num"]) for r in prefix), prediction_months=s,
                outcome_months=outcome, event=bool(row["event_observed"]), tensor_path=tensor_dir / f"{row['ptid']}.pt",
            ))
    if any(not any(r.split == split for r in records) for split in SPLITS):
        raise ValueError("Every split must contain eligible MRI landmarks")
    return records, {
        "source_images": frame.height,
        "unassigned_images": frame.height - assigned.height,
        "exclusions": assigned.filter(pl.col("exclusion_reason").is_not_null()).group_by("exclusion_reason").len().to_dicts(),
        "eligible_images": len(records),
        "eligible_patients": len({r.ptid for r in records}),
        "assigned_patients_without_eligible_images": splits.height - len({r.ptid for r in records}),
        "splits": {s: {"patients": len({r.ptid for r in records if r.split == s}),
                       "images": sum(r.split == s for r in records),
                       "events": len({r.ptid for r in records if r.split == s and r.event})} for s in SPLITS},
    }


def preflight_tensors(records: list[Observation]) -> dict:
    """Inspect every package before training; reject stale/missing image mappings."""
    latest = {r.ptid: r for r in records}
    errors = []
    for r in latest.values():
        try:
            p = torch.load(r.tensor_path, map_location="cpu", weights_only=True, mmap=True)
            ids = [str(i) for i in p["img_ids"]]
            x = p["images"]
            if str(p["ptid"]) != r.ptid or len(ids) != len(set(ids)):
                raise ValueError("wrong patient or duplicate image IDs")
            if x.ndim != 5 or x.shape[1] != 1 or x.shape[0] != len(ids):
                raise ValueError("image tensor shape/ID count mismatch")
            missing = set(r.image_ids) - set(ids)
            if missing:
                raise ValueError(f"missing {len(missing)} selected image IDs")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            errors.append(f"{r.ptid}: {exc}")
    if errors:
        raise ValueError(f"Tensor preflight failed for {len(errors)} patients; rebuild their packages. " + "; ".join(errors[:8]))
    return {"tensor_patients_checked": len(latest), "selection": "image_id", "tensor_outcome_labels_used": False}


def at_landmark(records: list[Observation], landmark: float, split: str) -> list[Observation]:
    """Latest actual MRI <= t, with T > t; never include a later scan in the same bin."""
    if not math.isfinite(landmark) or landmark < 0:
        raise ValueError("Landmark must be finite and nonnegative")
    latest = {}
    for r in records:
        if r.split == split and r.prediction_months <= landmark < r.outcome_months:
            if r.ptid not in latest or r.acquisition_months[-1] > latest[r.ptid].acquisition_months[-1]:
                latest[r.ptid] = replace(r, prediction_months=landmark)
    return [latest[p] for p in sorted(latest)]


def sample_epoch(records: list[Observation], seed: int, epoch: int) -> list[Observation]:
    """Uniformly sample one eligible MRI per patient, independent of model variant."""
    groups = {}
    for r in records:
        groups.setdefault(r.ptid, []).append(r)
    sampled = []
    for ptid in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{epoch}:{ptid}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        sampled.append(rng.choice(sorted(groups[ptid], key=lambda r: r.prediction_months)))
    random.Random(seed + epoch).shuffle(sampled)
    return sampled


class SurvivalDataset(Dataset):
    """Reuse MRI preprocessing; load only one image for either baseline."""

    def __init__(self, records, *, history, n_slots, spatial_size):
        self.records, self.history, self.n_slots = records, history, n_slots
        legacy = [LandmarkPatientRecord(
            ptid=r.ptid, split=r.split, landmark_months=r.prediction_months,
            observed_time_months=r.outcome_months-r.prediction_months, event_observed=r.event,
            image_ids=r.image_ids if history else r.image_ids[-1:],
            relative_times=r.acquisition_months if history else r.acquisition_months[-1:], tensor_path=r.tensor_path,
        ) for r in records]
        self.images = LandmarkSequenceDataset(legacy, spatial_size=spatial_size)
        self.counts = {}
        for r in records:
            self.counts[r.ptid] = self.counts.get(r.ptid, 0) + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r, sample = self.records[index], self.images[index]
        if not torch.isfinite(sample.images).all():
            raise ValueError(f"Non-finite MRI voxels for {r.ptid}")
        return {"images": sample.images, "slots": r.visit_nums if self.history else (0,),
                "times": r.acquisition_months if self.history else r.acquisition_months[-1:],
                "entry": r.prediction_months, "outcome": r.outcome_months,
                "event": r.event, "weight": 1 / self.counts[r.ptid], "record": r}


def collate_survival(samples):
    # Keep explicit empty canonical slots. No fabricated MRI or time enters the encoder.
    slots = max(max(s["slots"]) for s in samples) + 1
    shape = samples[0]["images"].shape[1:]
    images = torch.zeros((len(samples), slots, *shape))
    times = torch.zeros((len(samples), slots))
    observed = torch.zeros((len(samples), slots), dtype=torch.bool)
    for i, s in enumerate(samples):
        indices = list(s["slots"])
        images[i, indices] = s["images"]
        times[i, indices] = torch.tensor(s["times"])
        observed[i, indices] = True
    return {"images": images, "times": times, "observed": observed,
            **{k: torch.tensor([s[k] for s in samples], dtype=torch.float32) for k in ("entry", "outcome", "event", "weight")},
            "records": [s["record"] for s in samples]}
