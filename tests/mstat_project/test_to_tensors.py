import asyncio
import threading
import time
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import polars as pl
import pytest
import torch

from mstat_project.data_prep import to_tensors
from mstat_project.utils import ADNISubjectTensorDictMin


def _minimal_metadata(subject_id: str, image_ids: list[str]) -> ADNISubjectTensorDictMin:
    return cast(
        ADNISubjectTensorDictMin,
        {
            "ptid": subject_id,
            "img_ids": image_ids,
            "images": None,
        },
    )


def _write_archive(path: Path, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w") as zip_file:
        for member in members:
            zip_file.writestr(member, b"nifti")


def test_get_subject_images_supports_multiple_cohorts_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "preprocessed"
    _write_archive(input_path / "cohort_01.zip", ["102/scan.nii.gz", "101/scan.nii.gz"])
    _write_archive(input_path / "cohort_02.zip", ["103/scan.nii.gz"])

    def fake_nifti_to_tensor(path: Path) -> torch.Tensor:
        return torch.full((1, 1, 1, 1), float(path.parent.name))

    monkeypatch.setattr(to_tensors, "INPUT_PATH", input_path)
    monkeypatch.setattr(to_tensors, "nifti_to_tensor", fake_nifti_to_tensor)

    images = to_tensors.get_subject_images(
        _minimal_metadata("subject", ["101", "103", "102"]),
        input_cohorts=[1, 2, 1],
    )

    assert images.shape == (3, 1, 1, 1, 1)
    assert images[:, 0, 0, 0, 0].tolist() == [101.0, 103.0, 102.0]


def test_get_subject_images_rejects_empty_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="has no image IDs"):
        to_tensors.get_subject_images(_minimal_metadata("subject", []), input_cohorts=[])

    with pytest.raises(ValueError, match="2 image IDs but 1 cohort values"):
        to_tensors.get_subject_images(
            _minimal_metadata("subject", ["101", "102"]),
            input_cohorts=[1],
        )


def test_get_subject_images_reports_missing_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "preprocessed"
    _write_archive(input_path / "cohort_01.zip", ["999/scan.nii.gz"])
    monkeypatch.setattr(to_tensors, "INPUT_PATH", input_path)

    with pytest.raises(FileNotFoundError, match="image 101"):
        to_tensors.get_subject_images(_minimal_metadata("subject", ["101"]), input_cohorts=[1])


def test_get_subject_images_reports_duplicate_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "preprocessed"
    _write_archive(
        input_path / "cohort_01.zip",
        ["101/scan.nii.gz", "101/repeat.nii.gz"],
    )
    monkeypatch.setattr(to_tensors, "INPUT_PATH", input_path)

    with pytest.raises(ValueError, match="Multiple NIfTI files found for image 101"):
        to_tensors.get_subject_images(_minimal_metadata("subject", ["101"]), input_cohorts=[1])


def test_main_async_limits_concurrency_isolates_failures_and_limits_patients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subjects_df = pl.DataFrame(
        {
            "ptid": ["001", "001", "002", "002", "003", "004"],
        }
    )
    lock = threading.Lock()
    active = 0
    max_active = 0
    processed_ids: list[str] = []

    def fake_process_subject(subject_id: str, subject_df: pl.DataFrame) -> None:
        nonlocal active, max_active
        assert subject_df["ptid"].unique().to_list() == [subject_id]
        with lock:
            active += 1
            max_active = max(max_active, active)
            processed_ids.append(subject_id)
        try:
            time.sleep(0.03)
            if subject_id == "002":
                raise RuntimeError("bad subject")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(to_tensors, "load_subjects_dataframe", lambda: subjects_df)
    monkeypatch.setattr(to_tensors, "process_subject", fake_process_subject)
    monkeypatch.setattr(to_tensors, "TENSOR_OUTPUT", tmp_path / "tensors")

    results = asyncio.run(to_tensors.main_async(limit=3, max_workers=2))

    assert sorted(processed_ids) == ["001", "002", "003"]
    assert 1 < max_active <= 2
    assert (tmp_path / "tensors").is_dir()
    assert [(result.subject_id, result.success) for result in results] == [
        ("001", True),
        ("002", False),
        ("003", True),
    ]
    assert results[1].error == "bad subject"


def test_main_async_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        asyncio.run(to_tensors.main_async(limit=-1))
