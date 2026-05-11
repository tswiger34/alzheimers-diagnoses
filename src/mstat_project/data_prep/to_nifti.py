import os
import subprocess
import zipfile
from pathlib import Path
from typing import Mapping

import sqlalchemy

from mstat_project.utils import get_db_engine

COHORTS = list(range(1, 12))


def dicom_to_nifti(dicom_dir: str | Path, output_dir: str | Path) -> None:
    """Converts a folder of DICOM files to a NIfTI file.

    Args:
        dicom_dir (_type_): The directory containing the DICOM files
        output_dir (_type_): The output directory for the NIfTI file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "y",
        "-o",
        str(output_dir),
        str(dicom_dir),
    ]

    subprocess.run(args=cmd, check=True)


def get_dicom_folders_in_cohort(cohort: int) -> Mapping[str, str]:
    """Get a mapping of DICOM image folders to their image IDs

    Args:
        cohort (int): The cohort number, used to build the file path

    Returns:
        Mapping[str, Path]: A mapping of DICOM image folder paths, keyed by their image ID
    """
    data_path = os.getenv(key="IMAGES_PATH", default="data/raw_images")
    extract_dir = f"{data_path}/cohort_{str(cohort).zfill(2)}"
    zfile_path = f"{extract_dir}.zip"
    with zipfile.ZipFile(file=zfile_path, mode="r") as f:
        files = f.filelist
        dicoms: dict[str, str] = {}
        img_folders = set(
            [Path(zfile.filename).parent.as_posix() for zfile in files if zfile.filename.endswith(".dcm")]
        )
        dicoms = {fldr.split("/")[-1]: f"{extract_dir}/{fldr}" for fldr in img_folders}
        print(f"Extracting {zfile_path} to {extract_dir}")
        f.extractall(path=extract_dir)
    print(f"Finished extracting {zfile_path}")
    return dicoms


def process_cohort(cohort: int, selected_ids: set[str]) -> set[str]:
    """Converts selected image IDs to NIfTI files

    Args:
        cohort (int): Cohort number to process
        selected_ids (set[str]): The image IDs that still need to be processed
    """
    print(f"Processing Cohort: {cohort}")
    dicoms = get_dicom_folders_in_cohort(cohort)
    imgs_to_process = set([img_id for img_id in dicoms.keys() if img_id in selected_ids])
    dicoms_to_process = {img_id: img_folder for img_id, img_folder in dicoms.items() if img_id in imgs_to_process}
    for _img_id, dicom_dir in dicoms_to_process.items():
        dicom_to_nifti(dicom_dir, output_dir="data/images/nifti")
    data_path = os.getenv(key="IMAGES_PATH", default="data/raw_images")
    extract_dir = f"{data_path}/cohort_{str(cohort).zfill(2)}"
    os.rmdir(path=extract_dir)
    return imgs_to_process


def process_all():

    # Get selected image IDs
    db_eng = get_db_engine()
    with db_eng.connect() as conn:
        result = conn.execute(statement=sqlalchemy.text(text="SELECT image_id FROM _core.core_image_set")).all()
    ids = set(str(row[0]) for row in result)
    for cohort in COHORTS:
        imgs_processed = process_cohort(cohort, selected_ids=ids)
        ids = ids - imgs_processed

    return ids


if __name__ == "__main__":
    ids = process_all()
    process_cohort(cohort=1, selected_ids=ids)
