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


def nifti_to_tensor(input_path: str | Path) -> None:
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


def get_subject_metadata(subject_id: str) -> ADNISubjectTensorDictMin:
    """Get the subject metadata from the database.

    Args:
        subject_id (str): The ADNI ID of the subject

    Returns:
        ADNISubjectTensorDictMin: The subject metadata
    """
    ...  # Implementation goes here


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


def process_img(): ...


def process_cohort(cohort: int): ...


def main():
    eng = get_db_engine()
    with eng.connect() as conn:
        subjects_df = pl.read_database(
            query="""
            SELECT
                ptid_visit_date,
                image_id,
                ptid,
                processing_set_cohort,
                image_date,
                visit_code,
                months_since_prior_image,
                months_since_baseline_image,
                age_at_baseline,
                age_at_image,
                first_ad_diagnosis,
                observed_time_months,
                time_to_ad_from_baseline,
                time_to_ad_from_visit,
                dx.is_censored,
                last_diagnosis,
                is_ad_at_visit
            FROM _core.core_image_set
            """,
            connection=conn,
        )
