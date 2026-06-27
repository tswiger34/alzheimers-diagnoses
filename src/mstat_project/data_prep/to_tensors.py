"""Takes the preprocessed NIfTI images and turns them into tensors to be used in the LTSA model."""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import torch
import torchio as tio
from dotenv import load_dotenv

from mstat_project.utils import ADNISubjectTensorDictMin, get_db_engine

from .utils import resolve_max_workers

load_dotenv()
COHORTS = list(range(1, 11))
IMAGE_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))
INPUT_PATH = IMAGE_PATH / "preprocessed"
TENSOR_OUTPUT = IMAGE_PATH / "tensors"

NIFTI_TRANSFORM = tio.Compose(
    [
        tio.ToCanonical(),
        tio.Resample(1.0),
        tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        ),
        tio.CropOrPad(target_shape=(182, 218, 182)),
    ]
)


@dataclass(frozen=True)
class SubjectTensorResult:
    """Result from processing one subject's longitudinal image sequence."""

    subject_id: str
    success: bool
    error: str | None = None


def nifti_to_tensor(input_path: str | Path) -> torch.Tensor:
    """Convert a T1 NIfTI image into a standardized MRI tensor.

    Args:
        input_path: Path to the preprocessed NIfTI file.

    Returns:
        MRI tensor with shape ``(1, 182, 218, 182)`` and dtype
        ``torch.float32``.

    Raises:
        FileNotFoundError: If the input NIfTI file does not exist.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {input_path}")

    subject = tio.Subject(t1=tio.ScalarImage(str(input_path)))
    subject = NIFTI_TRANSFORM(subject)

    return subject.t1.data.to(torch.float32)


def get_subject_metadata(subject_id: str, subject_df: pl.DataFrame) -> ADNISubjectTensorDictMin:
    """Get the subject metadata from the database.

    Args:
        subject_id (str): The ADNI ID of the subject
        subject_df (pl.DataFrame): DataFrame containing the subject's metadata

    Returns:
        ADNISubjectTensorDictMin: The subject metadata
    """
    if subject_df.is_empty():
        raise ValueError(f"Subject {subject_id} not found in the database.")

    subject_metadata: ADNISubjectTensorDictMin = {
        "ptid": subject_id,
        "img_ids": [str(img_id) for img_id in subject_df["image_id"].to_list()],
        "images": None,
        "months_since_prior_mri": torch.tensor(
            subject_df["months_since_prior_image"].to_list(), dtype=torch.float32
        ),
        "months_since_baseline_mri": torch.tensor(
            subject_df["months_since_baseline_image"].to_list(), dtype=torch.float32
        ),
        "time_to_event_from_baseline": torch.tensor(
            subject_df["months_to_ad_from_baseline"].to_list()[0], dtype=torch.float32
        ),
        "time_to_event_from_mri": torch.tensor(
            subject_df["months_to_ad_from_image"].to_list(), dtype=torch.float32
        ),
        "dx_code_at_visit": torch.tensor(subject_df["diagnosis_code_at_visit"].to_list(), dtype=torch.long),
        "age_at_baseline": torch.tensor(subject_df["age_at_baseline"].to_list()[0], dtype=torch.float32),
        "age_at_image": torch.tensor(subject_df["age_at_image"].to_list(), dtype=torch.float32),
        "is_censored": torch.tensor(subject_df["is_censored"].to_list()[0], dtype=torch.bool),
    }
    return subject_metadata


def get_subject_images(
    subject_metadata: ADNISubjectTensorDictMin,
    input_cohorts: list[int],
) -> torch.Tensor:
    """Load and stack all MRI tensors for one subject.

    Args:
        subject_metadata: Subject metadata containing ``ptid`` and ``img_ids``.
        input_cohorts: Cohort for each image ID, in the same chronological order.

    Returns:
        Tensor with shape ``(n_visits, 1, 182, 218, 182)``.

    Raises:
        ValueError: If image/cohort counts differ, no images are supplied, or an
            archive contains multiple NIfTI files for an image ID.
        FileNotFoundError: If an image ID has no NIfTI file in its cohort archive.
    """
    image_ids = subject_metadata["img_ids"]
    if not image_ids:
        raise ValueError(f"Subject {subject_metadata['ptid']} has no image IDs")
    if len(image_ids) != len(input_cohorts):
        raise ValueError(
            f"Subject {subject_metadata['ptid']} has {len(image_ids)} image IDs "
            f"but {len(input_cohorts)} cohort values"
        )

    image_locations = list(zip(image_ids, input_cohorts, strict=True))
    extracted_paths: dict[tuple[str, int], Path] = {}

    with tempfile.TemporaryDirectory() as export_folder:
        export_path = Path(export_folder)

        for cohort in dict.fromkeys(input_cohorts):
            cohort_image_ids = {image_id for image_id, image_cohort in image_locations if image_cohort == cohort}
            archive_path = INPUT_PATH / f"cohort_{cohort:02d}.zip"

            with ZipFile(archive_path) as zip_file:
                members_by_image_id: dict[str, list[str]] = {image_id: [] for image_id in cohort_image_ids}
                for member in zip_file.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".nii.gz"):
                        continue

                    member_parts = Path(member.filename).parts
                    if member_parts and member_parts[0] in members_by_image_id:
                        members_by_image_id[member_parts[0]].append(member.filename)

                for image_id, members in members_by_image_id.items():
                    if not members:
                        raise FileNotFoundError(f"No NIfTI file found for image {image_id} in {archive_path}")
                    if len(members) > 1:
                        raise ValueError(
                            f"Multiple NIfTI files found for image {image_id} in {archive_path}: {members}"
                        )

                    member_name = members[0]
                    extracted_paths[(image_id, cohort)] = Path(zip_file.extract(member_name, path=export_path))

        img_tensors = [
            nifti_to_tensor(extracted_paths[(image_id, cohort)]) for image_id, cohort in image_locations
        ]
        print(f"Loaded {len(img_tensors)} images for subject {subject_metadata['ptid']}")

    return torch.stack(img_tensors, dim=0)


def get_output_path(subject_id: str) -> Path:
    """Creates the file path to save the tensor to.

    File path is constructed as `TENSOR_OUTPUT/{subject_id}.pt`

    Args:
        subject_id (str): The ADNI ID of the subject

    Returns:
        Path: Path to the `.pt` file to save the tensor to
    """
    return TENSOR_OUTPUT / f"{subject_id}.pt"


def validate_subject_tensor_dict(subject: ADNISubjectTensorDictMin) -> None:
    """Validate minimal subject tensor package.

    Args:
        subject: Subject-level tensor package.

    Raises:
        ValueError: If tensor dimensions or metadata lengths are inconsistent.
    """
    images = subject["images"]

    if images is None:
        raise ValueError(f"Subject {subject['ptid']} has no images loaded")

    if images.ndim != 5:
        raise ValueError(f"Expected images to have 5 dims, got {images.shape}")

    n_visits = images.shape[0]

    visit_level_keys = [
        "months_since_prior_mri",
        "months_since_baseline_mri",
        "age_at_image",
        "dx_code_at_visit",
        "time_to_event_from_mri",
    ]

    for key in visit_level_keys:
        if subject[key].shape[0] != n_visits:  # ty: ignore
            raise ValueError(f"{key} has length {subject[key].shape[0]}, expected {n_visits}")  # ty: ignore

    if len(subject["img_ids"]) != n_visits:
        raise ValueError(f"img_ids has length {len(subject['img_ids'])}, expected {n_visits}")


def load_subjects_dataframe() -> pl.DataFrame:
    """Load chronologically ordered subject image metadata from the database."""
    eng = get_db_engine()
    with eng.connect() as conn:
        return pl.read_database(
            query="""
            SELECT
                image_id,
                ptid,
                image_date,
                processing_set_cohort,
                visit_code,
                diagnosis_code_at_visit,
                months_since_prior_image,
                months_since_baseline_image,
                age_at_baseline,
                age_at_image,
                months_to_ad_from_image,
                months_to_ad_from_baseline,
                is_censored
            FROM _core.core_image_set
            ORDER BY ptid, image_date, ptid_img_number
            """,
            connection=conn,
        )


def process_subject(subject_id: str, subject_df: pl.DataFrame) -> None:
    """Build, validate, and save the tensor package for one subject."""
    metadata = get_subject_metadata(subject_id, subject_df)
    input_cohorts = [int(cohort) for cohort in subject_df["processing_set_cohort"].to_list()]
    images = get_subject_images(metadata, input_cohorts=input_cohorts)

    subject_tensor_dict: ADNISubjectTensorDictMin = {
        **metadata,
        "images": images,
    }

    validate_subject_tensor_dict(subject_tensor_dict)
    torch.save(subject_tensor_dict, get_output_path(subject_id))


async def process_subject_async(subject_id: str, subject_df: pl.DataFrame) -> SubjectTensorResult:
    """Process one subject in a worker thread without blocking the event loop."""
    try:
        await asyncio.to_thread(process_subject, subject_id, subject_df)
    except Exception as exc:
        error = str(exc)
        print(f"Skipping {subject_id}: {error}")
        return SubjectTensorResult(subject_id=subject_id, success=False, error=error)

    return SubjectTensorResult(subject_id=subject_id, success=True)


async def main_async(
    limit: int | None = None,
    max_workers: int | None = None,
) -> list[SubjectTensorResult]:
    """Process subjects concurrently with a bounded async worker queue."""
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    subjects_df = await asyncio.to_thread(load_subjects_dataframe)
    subject_ids = subjects_df["ptid"].unique(maintain_order=True).to_list()
    if limit is not None:
        subject_ids = subject_ids[:limit]

    print(f"Found {len(subject_ids)} subjects")
    print(subject_ids[:10])

    TENSOR_OUTPUT.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_max_workers(max_workers, env_var="TENSOR_WORKERS", default=4)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    results: list[SubjectTensorResult] = []

    for subject_id in subject_ids:
        queue.put_nowait(subject_id)

    async def worker() -> None:
        while True:
            subject_id = await queue.get()
            try:
                if subject_id is None:
                    return

                subject_df = subjects_df.filter(pl.col("ptid") == subject_id)
                result = await process_subject_async(subject_id, subject_df)
                results.append(result)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    for _ in workers:
        queue.put_nowait(None)

    await queue.join()
    await asyncio.gather(*workers)

    n_saved = sum(result.success for result in results)
    n_skipped = len(results) - n_saved
    print(f"Saved {n_saved} subjects")
    print(f"Skipped {n_skipped} subjects")

    return sorted(results, key=lambda result: result.subject_id)


def main(limit: int | None = None, max_workers: int | None = None) -> None:
    """Synchronous command-line entry point for tensor generation."""
    asyncio.run(main_async(limit=limit, max_workers=max_workers))


if __name__ == "__main__":
    main(100)
