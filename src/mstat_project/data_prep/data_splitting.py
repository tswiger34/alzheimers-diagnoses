import os
import random

import polars as pl

from mstat_project.utils import get_db_engine

PROPORTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def get_core_img_set() -> pl.DataFrame:
    img_base_path = os.getenv("IMAGES_PATH", "./images")
    tensor_files = [f for f in os.listdir(f"{img_base_path}/tensors") if f.endswith(".pt")]
    ptid_tensors = [f.split(".")[0] for f in tensor_files]

    with get_db_engine().connect() as conn:
        base_df = pl.read_database(
            "SELECT * FROM core_image_set",
            conn,
        )
    patients = (
        base_df.group_by("ptid")
        .agg(
            pl.len().alias("n_images"),
            pl.col("baseline_diagnosis").first(),
            pl.col("is_censored").first(),
        )
        .filter(pl.col("ptid").is_in(ptid_tensors))
    )
    return patients


def weighted_patient_split(
    patients_df: pl.DataFrame,
    proportions: dict[str, float],
    seed: int = 42,
) -> pl.DataFrame:

    if not abs(sum(proportions.values()) - 1.0) < 1e-9:
        raise ValueError("Split proportions must sum to 1")

    rng = random.Random(seed)
    assignments = []

    strata = patients_df.partition_by(
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

    return pl.DataFrame(assignments)


if __name__ == "__main__":
    patients_df = get_core_img_set()
    split_df = weighted_patient_split(patients_df, PROPORTIONS)
    with get_db_engine().connect() as conn:
        split_df.write_database("_raw.train_test_split", conn, if_table_exists="replace")
