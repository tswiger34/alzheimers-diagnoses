"""Takes the preprocessed NIfTI images and turns them into tensors to be used in the LTSA model."""

import nibabel as nib  # noqa
import os
from pathlib import Path
from dotenv import load_dotenv
from src.mstat_project.utils import get_db_engine, ADNISubjectTensorDictMin
import torch
import torchio as tio
import polars as pl

load_dotenv()
COHORTS = list(range(1, 11))
IMAGE_PATH = Path(os.getenv("IMAGE_PATH", "data/images"))
INPUT_PATH = IMAGE_PATH / "preprocessed"
TENSOR_OUTPUT = IMAGE_PATH / "tensors"


def nifti_to_tensor(input_path: str | Path) -> torch.Tensor:
    """Convert a T1 NIfTI image into a standardized 182x218x182 FP32 tensor."""

    subject = tio.Subject(t1=tio.ScalarImage(str(input_path)))

    transform = tio.Compose(
        [
            tio.ToCanonical(),
            tio.Resample(1.0),
            tio.RescaleIntensity(
                out_min_max=(0, 1),
                percentiles=(1, 99),
            ),
            # Force fixed spatial size
            tio.CropOrPad(target_shape=(182, 218, 182)),
        ]
    )

    subject = transform(subject)

    image = subject.t1.data

    tensor = image.to(torch.float32)

    return tensor


def get_subject_metadata(subject_id: str, subjects_df: pl.DataFrame) -> ADNISubjectTensorDictMin:
    """Get the subject metadata from the database.

    Args:
        subject_id (str): The ADNI ID of the subject
        subjects_df (pl.DataFrame): DataFrame containing all subjects' metadata

    Returns:
        ADNISubjectTensorDictMin: The subject metadata
    """
    subject_df = subjects_df.filter(pl.col("ptid") == subject_id)

    if subject_df.is_empty():
        raise ValueError(f"Subject {subject_id} not found in the database.")

    subject_metadata: ADNISubjectTensorDictMin = {
        "ptid": subject_id,
        "img_ids": subject_df["image_id"].to_list(),
        "images": None,
        "months_since_prior_mri": torch.tensor(
            subject_df["months_since_prior_image"].to_list(), dtype=torch.float32
        ),
        "months_since_baseline_mri": torch.tensor(
            subject_df["months_since_baseline_image"].to_list(), dtype=torch.float32
        ),
        "time_to_event": torch.tensor(subject_df["time_to_ad_from_image"].to_list()[0], dtype=torch.float32),
        "dx_code_at_visit": torch.tensor(subject_df["diagnosis_code_at_visit"].to_list(), dtype=torch.int16),
        "age_at_baseline": torch.tensor(subject_df["age_at_baseline"].to_list()[0], dtype=torch.float32),
    }

    return subject_metadata


def get_subject_images(subject_metadata: ADNISubjectTensorDictMin):
    input_paths = [
        INPUT_PATH / f"{subject_metadata['ptid']}/{img_id}.nii.gz" for img_id in subject_metadata["img_ids"]
    ]
    img_tensors = [nifti_to_tensor(input_path) for input_path in input_paths]
    return img_tensors


def get_output_path(subject_id: str, visit_num: int, img_id: str) -> Path:
    """Creates the file path to save the tensor to.

    File path is constructed as `TENSOR_OUTPUT/{subject_id}/{img_id}_visit_{visit_num}.pt`

    Args:
        subject_id (str): The ADNI ID of the subject
        visit_num (int): The visit number the image was taken on
        img_id (str): The ID of the image

    Returns:
        Path: Path to the `.pt` file to save the tensor to
    """
    rel_path = Path(f"{subject_id}/{img_id}_visit_{visit_num}.pt")
    return TENSOR_OUTPUT / rel_path


def main():
    eng = get_db_engine()
    with eng.connect() as conn:
        subjects_df = pl.read_database(
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
                time_to_ad_from_image,
                is_censored,
                ptid_img_number
            FROM _core.core_image_set
            ORDER BY ptid, image_date, ptid_img_number
            """,
            connection=conn,
        )
