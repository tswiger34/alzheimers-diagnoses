from pathlib import Path

import polars as pl
import pytest

from mstat_project.data_prep.data_splitting import (
    PROPORTIONS,
    frame_fingerprint,
    patient_split_frame,
    tensor_eligible_images,
    validate_assignments,
    validate_binned_cohort,
    weighted_patient_split,
)


def patients():
    return pl.DataFrame(
        [
            {
                "ptid": f"p{i:03}",
                "n_images": i % 7 + 1,
                "baseline_diagnosis": "CN" if i < 40 else "MCI",
                "is_censored": i % 2 == 0,
            }
            for i in range(80)
        ]
    )


def cohort():
    # A skipped scheduled bin must remain a gap, not become the second visit code.
    images = pl.DataFrame(
        {
            "ptid": ["p1", "p1"],
            "image_id": [1, 2],
            "visit_num": [0, 2],
            "visit_code": ["BL", "v2"],
            "days_since_baseline_image": [0, 365],
            "months_since_baseline_image": [0.0, 12.0],
            "n_ptid_imgs": [9, 9],
            "baseline_diagnosis": ["CN", "CN"],
            "is_censored": [True, True],
        }
    )
    buckets = pl.DataFrame(
        {
            "visit_num": [0, 1, 2],
            "visit_code": ["BL", "v1", "v2"],
            "follow_up_months": [0, 6, 12],
            "follow_up_start": [0.0, 0.0, 9.0],
            "follow_up_end": [0.0, 9.0, 18.0],
        }
    )
    return images, buckets


def test_split_is_patient_disjoint_and_independent_of_input_order():
    frame = patients()
    assigned = weighted_patient_split(frame, PROPORTIONS)
    assert assigned.equals(weighted_patient_split(frame.sample(fraction=1, shuffle=True, seed=123), PROPORTIONS))
    validate_assignments(frame, assigned)
    strata = frame.join(assigned, on="ptid").group_by("baseline_diagnosis", "is_censored")
    assert strata.agg(pl.col("split").n_unique())["split"].to_list() == [3] * 4
    assert not assigned.equals(weighted_patient_split(frame, PROPORTIONS, seed=7))


@pytest.mark.parametrize(
    "proportions", [{}, {"train": 1, "test": 0}, {"train": -1, "test": 2}, {"train": float("nan")}, {"train": 0.7}]
)
def test_invalid_proportions_fail(proportions):
    with pytest.raises(ValueError, match="proportions"):
        weighted_patient_split(patients(), proportions)


def test_duplicate_patient_rejected_before_assignment():
    frame = patients()
    with pytest.raises(ValueError, match="exactly once"):
        weighted_patient_split(pl.concat([frame, frame.head(1)]), PROPORTIONS)


def test_retained_counts_and_schedule_gaps():
    images, buckets = cohort()
    validate_binned_cohort(images, buckets)
    assert patient_split_frame(images)["n_images"].to_list() == [2]


@pytest.mark.parametrize(
    "months,code,number,days,valid",
    [
        (0.0, "BL", 0, 0, True),
        (0.07, "v1", 1, 2, True),
        (9.0, "v1", 1, 274, False),
        (9.0, "v2", 2, 274, True),
        (18.0, "v2", 2, 548, False),
        (0.0, "v1", 1, 0, False),
    ],
)
def test_bin_boundaries(months, code, number, days, valid):
    images, buckets = cohort()
    if code != "BL":
        row = images.tail(1).with_columns(
            pl.lit(months).alias("months_since_baseline_image"),
            pl.lit(code).alias("visit_code"),
            pl.lit(number).cast(pl.Int64).alias("visit_num"),
            pl.lit(days).cast(pl.Int64).alias("days_since_baseline_image"),
        )
        images = pl.concat([images.head(1), row])
    if valid:
        validate_binned_cohort(images, buckets)
    else:
        with pytest.raises(ValueError, match="schedule"):
            validate_binned_cohort(images, buckets)


def test_duplicate_patient_bin_rejected():
    images, buckets = cohort()
    with pytest.raises(ValueError, match="unique image"):
        validate_binned_cohort(pl.concat([images, images.tail(1)]), buckets)


def test_tensor_eligibility_and_fingerprint(tmp_path: Path):
    images, _ = cohort()
    assert tensor_eligible_images(images, tmp_path).is_empty()
    (tmp_path / "p1.pt").touch()
    assert tensor_eligible_images(images, tmp_path).equals(images)
    with pytest.raises(FileNotFoundError):
        tensor_eligible_images(images, tmp_path / "absent")
    assert frame_fingerprint(images) == frame_fingerprint(images.reverse())
    changed = images.with_columns((pl.col("image_id") + 1).alias("image_id"))
    assert frame_fingerprint(images) != frame_fingerprint(changed)


def test_missing_or_overlapping_assignments_rejected():
    frame = patients()
    assigned = weighted_patient_split(frame, PROPORTIONS)
    with pytest.raises(ValueError, match="exactly the eligible"):
        validate_assignments(frame, assigned.head(10))
    with pytest.raises(ValueError, match="duplicate"):
        validate_assignments(frame, pl.concat([assigned, assigned.head(1)]))
