import os
import random
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from sqlalchemy import Engine, inspect, text

from mstat_project.utils import get_db_engine

PROPORTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def load_binned_cohort(engine: Engine) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read retained images and the authoritative dbt schedule in one snapshot."""
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        images = pl.read_database(
            """
            SELECT i.ptid, i.image_id, i.image_date, i.visit_num, i.visit_code,
                i.days_since_baseline_image, i.months_since_baseline_image,
                i.months_since_prior_image, i.n_ptid_imgs,
                i.baseline_diagnosis, i.diagnosis_at_visit, i.final_diagnosis,
                i.change_from_baseline, i.is_censored, i.mri_is_valid,
                i.months_to_ad_from_baseline, i.age_at_baseline,
                i.pt_education_years, i.pt_marital_status, d.gender_code AS pt_gender_code,
                i.mri_manufacturer, i.slice_thickness, i.magnetic_field_strength,
                i.is_accelerated
            FROM _core.core_image_set AS i
            LEFT JOIN _core.core_ptdemog AS d USING (ptid)
            ORDER BY i.ptid, i.visit_num, i.image_id
            """,
            conn,
            infer_schema_length=None,
        )
        buckets = pl.read_database("SELECT * FROM _core.core_follow_up_buckets ORDER BY visit_num", conn)
    validate_binned_cohort(images, buckets)
    return images, buckets


def validate_binned_cohort(images: pl.DataFrame, buckets: pl.DataFrame) -> None:
    """Reject duplicate, mis-binned, or inconsistent data; skipped bins are valid."""
    required = [
        "ptid",
        "image_id",
        "visit_num",
        "visit_code",
        "days_since_baseline_image",
        "months_since_baseline_image",
        "baseline_diagnosis",
        "is_censored",
    ]
    if images.is_empty() or any(images[c].null_count() for c in required):
        raise ValueError("The image cohort is empty or has null required values")
    if images.select("ptid", "visit_num").is_duplicated().any() or images["image_id"].is_duplicated().any():
        raise ValueError("Expected one unique image per patient and scheduled visit bin")
    if buckets["visit_num"].is_duplicated().any() or buckets["visit_code"].is_duplicated().any():
        raise ValueError("Follow-up schedule has duplicate identifiers")
    checked = images.join(buckets, on=["visit_num", "visit_code"], how="left", validate="m:1")
    valid = (
        (pl.col("visit_code") == "BL")
        & (pl.col("days_since_baseline_image") == 0)
        & (pl.col("months_since_baseline_image") == 0)
    ) | (
        (pl.col("visit_code") != "BL")
        & (pl.col("days_since_baseline_image") > 0)
        & (pl.col("months_since_baseline_image") >= pl.col("follow_up_start"))
        & (pl.col("months_since_baseline_image") < pl.col("follow_up_end"))
    )
    if checked["follow_up_months"].null_count() or checked.filter(~valid.fill_null(False)).height:
        raise ValueError("Images do not match the baseline-relative follow-up schedule")
    patients = images.group_by("ptid").agg(
        (pl.col("visit_code") == "BL").sum().alias("baseline_count"),
        pl.col("baseline_diagnosis").n_unique().alias("diagnoses"),
        pl.col("is_censored").n_unique().alias("outcomes"),
    )
    if patients.filter(
        (pl.col("baseline_count") != 1) | (pl.col("diagnoses") != 1) | (pl.col("outcomes") != 1)
    ).height:
        raise ValueError("Each patient must have one baseline and consistent stratification fields")


def tensor_eligible_images(images: pl.DataFrame, tensor_dir: Path | None = None) -> pl.DataFrame:
    """Preserve the existing patient tensor-file eligibility rule (not a content audit)."""
    tensor_dir = tensor_dir or Path(os.getenv("IMAGES_PATH", "./images")) / "tensors"
    if not tensor_dir.is_dir():
        raise FileNotFoundError(f"Tensor directory does not exist: {tensor_dir}")
    ptids = sorted(path.stem for path in tensor_dir.glob("*.pt"))
    return images.filter(pl.col("ptid").is_in(ptids))


def patient_split_frame(images: pl.DataFrame) -> pl.DataFrame:
    """Count retained patient/bin rows, never the pre-filter n_ptid_imgs field."""
    return (
        images.group_by("ptid")
        .agg(
            pl.len().alias("n_images"),
            pl.col("baseline_diagnosis").first(),
            pl.col("is_censored").first(),
        )
        .sort("ptid")
    )


def get_core_img_set() -> pl.DataFrame:
    images, _ = load_binned_cohort(get_db_engine())
    return patient_split_frame(tensor_eligible_images(images))


def weighted_patient_split(
    patients_df: pl.DataFrame,
    proportions: dict[str, float],
    seed: int = 42,
) -> pl.DataFrame:

    if (
        not proportions
        or any(not math.isfinite(p) or p <= 0 for p in proportions.values())
        or not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError("Split proportions must be positive, finite, and sum to 1")
    required = ["ptid", "n_images", "baseline_diagnosis", "is_censored"]
    if patients_df.is_empty() or any(patients_df[c].null_count() for c in required):
        raise ValueError("Patient data is empty or has null required values")
    if patients_df["ptid"].is_duplicated().any():
        raise ValueError("Each patient must appear exactly once")
    if patients_df.filter(pl.col("n_images") <= 0).height:
        raise ValueError("Each patient must have at least one retained image")

    rng = random.Random(seed)
    assignments = []

    # Canonical order makes seed reproducibility independent of database/hash-group ordering.
    strata = patients_df.sort("baseline_diagnosis", "is_censored", "ptid").partition_by(
        ["baseline_diagnosis", "is_censored"],
        as_dict=False,
    )

    for stratum in strata:
        rows = stratum.to_dicts()

        rng.shuffle(rows)
        rows.sort(key=lambda row: row["n_images"], reverse=True)

        total_images = sum(row["n_images"] for row in rows)
        assigned_images = {split: 0 for split in proportions}

        for row in rows:
            # Select the split with the lowest projected target utilization.
            split = min(
                proportions,
                key=lambda name: (assigned_images[name] + row["n_images"]) / (total_images * proportions[name]),
            )

            assigned_images[split] += row["n_images"]
            assignments.append(
                {
                    "ptid": row["ptid"],
                    "split": split,
                }
            )

    return pl.DataFrame(assignments).sort("ptid")


def validate_assignments(patients: pl.DataFrame, assignments: pl.DataFrame) -> None:
    if assignments["ptid"].is_duplicated().any() or assignments.null_count().sum_horizontal().item():
        raise ValueError("Assignments contain duplicate patients or nulls")
    if set(assignments["ptid"]) != set(patients["ptid"]):
        raise ValueError("Assignments must cover exactly the eligible patients")
    if set(assignments["split"]) != set(PROPORTIONS):
        raise ValueError("All three splits must be present, with no unknown split labels")


def frame_fingerprint(frame: pl.DataFrame) -> str:
    """Stable content hash used to detect source/split drift in the read-only notebook."""
    canonical = frame.select(sorted(frame.columns)).sort(sorted(frame.columns)).write_csv()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def persist_splits(
    engine: Engine,
    images: pl.DataFrame,
    buckets: pl.DataFrame,
    assignments: pl.DataFrame,
    artifact_dir: Path,
    *,
    seed: int = 42,
    proportions: dict[str, float] | None = None,
) -> dict:
    """Archive provenance/previous rows, then atomically update the training split table."""
    validate_binned_cohort(images, buckets)
    validate_assignments(patient_split_frame(images), assignments)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = artifact_dir / stamp
    run_dir.mkdir()
    images.write_parquet(run_dir / "cohort.parquet")
    buckets.write_csv(run_dir / "follow_up_buckets.csv")
    assignments.write_csv(run_dir / "assignments.csv")
    manifest = {
        "created_at_utc": stamp,
        "source": "_core.core_image_set",
        "schedule": "_core.core_follow_up_buckets",
        "destination": "_raw.train_test_split",
        "seed": seed,
        "proportions": proportions or PROPORTIONS,
        "eligibility": "Patient has a .pt file; tensor contents are not validated here",
        "weight": "Retained patient/visit-bin rows, including event images",
        "strata": ["baseline_diagnosis", "is_censored"],
        "cohort_sha256": frame_fingerprint(images),
        "buckets_sha256": frame_fingerprint(buckets),
        "assignments_sha256": frame_fingerprint(assignments),
        "patients": assignments.height,
        "images": images.height,
        "snapshot_directory": str(run_dir.resolve()),
    }
    with engine.begin() as conn:
        if inspect(conn).has_table("train_test_split", schema="_raw"):
            previous = pl.read_database("SELECT ptid, split FROM _raw.train_test_split", conn)
            previous.write_csv(run_dir / "previous_assignments.csv")
            conn.execute(text("DELETE FROM _raw.train_test_split"))
        assignments.to_pandas().to_sql("train_test_split", conn, schema="_raw", if_exists="append", index=False)
        stored = pl.read_database("SELECT ptid, split FROM _raw.train_test_split", conn)
        if frame_fingerprint(stored) != frame_fingerprint(assignments):
            raise RuntimeError("Stored assignments failed verification; transaction rolled back")
    serialized = json.dumps(manifest, indent=2)
    (run_dir / "manifest.json").write_text(serialized, encoding="utf-8")
    (artifact_dir / "latest.json").write_text(serialized, encoding="utf-8")
    return manifest


if __name__ == "__main__":
    engine = get_db_engine()
    images, buckets = load_binned_cohort(engine)
    images = tensor_eligible_images(images)
    patients_df = patient_split_frame(images)
    split_df = weighted_patient_split(patients_df, PROPORTIONS)
    persist_splits(engine, images, buckets, split_df, Path("data/artifacts/data_splits"))
