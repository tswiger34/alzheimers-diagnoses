"""Takes the selected images from `_core.core_image_set` and performs gradunwarp where required and N4 bias correction.

For each `output_cohort`:
1. Unzip NIfTI files to temp dir `staging_temp_dir`
2. Perform N4 Bias Correction
    a. Create `output_temp_dir`
    b. Set output to `output_temp_dir`/{image_id}/{image_name}.nii.gz
3. Zip `output_temp_dir` --> {IMAGE_PATH}/preprocessed/{output_cohort}
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import SimpleITK as sitk
import sqlalchemy

from mstat_project.utils import get_db_engine

from .utils import IMAGES_PATH, get_cohort_name

NIFTI_INPUT_PATH = IMAGES_PATH / "nifti"
PREPROCESSED_IMAGE_PATH = IMAGES_PATH / "preprocessed"
INPUT_COHORTS: list[str] = list(get_cohort_name(i) for i in range(1, 13))
OUTPUT_COHORTS: list[int] = list(range(1, 11))


@dataclass
class ImageMetadata:
    image_id: str
    mri_manufacturer: str | None
    input_cohort: str
    processing_set_cohort: int
    input_image_name: str | None = None

    def get_staged_input_image_path(self, staging_temp_dir: Path) -> Path:
        """Gets the staged image path as `staging_temp_dir / self.image_id / self.input_image_name`

        Args:
            staging_temp_dir (Path): The temp directory used for staging the pre-processed images

        Raises:
            ValueError: If it cannot find a NIfTI file matching the image ID

        Returns:
            Path: The file path to the input image in the staging temp directory
        """
        if self.input_image_name is None:
            raise ValueError(f"No staged NIfTI filename recorded for image {self.image_id}")
        return staging_temp_dir / self.image_id / self.input_image_name

    def get_gradunwarp_image_path(self, staging_temp_dir: Path) -> Path:
        """Gets the path to the gradunwarp output image in the staging temp directory.

        Args:
            staging_temp_dir (Path): The temp directory used for staging the pre-processed images

        Returns:
            Path: The file path to the gradunwarp output image in the staging temp directory
        """
        if self.input_image_name is None:
            raise ValueError(f"No NIfTI filename recorded for image {self.image_id}")
        return (
            staging_temp_dir / self.image_id / f"{get_nifti_stem(path=self.input_image_name)}__gradunwarp.nii.gz"
        )

    def get_output_image_path(self, output_temp_dir: Path) -> Path:
        """Gets the path to the output image in the output temp directory.

        Args:
            output_temp_dir (Path): The temp directory used for staging the pre-processed images

        Returns:
            Path: The file path to the output image in the output temp directory
        """
        if not self.input_image_name:
            raise ValueError(f"No NIfTI filename recorded for image {self.image_id}")
        return output_temp_dir / self.image_id / f"{get_nifti_stem(path=self.input_image_name)}.nii.gz"


def is_nifti_file(path: str | Path) -> bool:
    path_str = str(path).lower()
    return path_str.endswith(".nii") or path_str.endswith(".nii.gz")


def get_nifti_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]

    raise ValueError(f"Expected a NIfTI file path, got {path}")


def get_cohort_archive_stem(cohort: str | int) -> str:
    cohort_str = str(cohort)
    if cohort_str.startswith("cohort_"):
        return cohort_str
    if cohort_str.isdigit():
        return get_cohort_name(int(cohort_str))
    return cohort_str


def get_nifti_archive_path(input_cohort: str | int) -> Path:
    return NIFTI_INPUT_PATH / f"{get_cohort_archive_stem(input_cohort)}.zip"


def get_preprocessed_archive_base_path(output_cohort: int) -> Path:
    return PREPROCESSED_IMAGE_PATH / get_cohort_name(output_cohort)


def get_image_list(output_cohort: int) -> list[ImageMetadata]:
    with get_db_engine().connect() as conn:
        qry = (
            "SELECT "
            "image_id::TEXT AS image_id, "
            "mri_manufacturer, "
            "cohort AS input_cohort, "
            "processing_set_cohort "
            "FROM _core.core_image_set "
            "WHERE processing_set_cohort = :output_cohort "
            "ORDER BY image_id"
        )
        result = conn.execute(
            statement=sqlalchemy.text(text=qry),
            parameters={"output_cohort": output_cohort},
        ).fetchall()
        image_list = [
            ImageMetadata(
                image_id=str(row.image_id),
                mri_manufacturer=row.mri_manufacturer,
                input_cohort=str(row.input_cohort),
                processing_set_cohort=int(row.processing_set_cohort),
            )
            for row in result
        ]

    return image_list


def get_zip_nifti_members_by_image_id(zip_file: ZipFile, image_ids: set[str]) -> dict[str, str]:
    """Return the single NIfTI archive member for each requested image ID."""

    members_by_image_id: dict[str, list[str]] = {image_id: [] for image_id in image_ids}
    for member in zip_file.infolist():
        if member.is_dir() or not is_nifti_file(member.filename):
            continue

        member_path = Path(member.filename)
        if not member_path.parts:
            continue

        image_id = member_path.parts[0]
        if image_id in members_by_image_id:
            members_by_image_id[image_id].append(member.filename)

    selected_members: dict[str, str] = {}
    for image_id, members in members_by_image_id.items():
        if not members:
            raise FileNotFoundError(f"No NIfTI file found in archive for image {image_id}")
        if len(members) > 1:
            raise ValueError(f"Multiple NIfTI files found in archive for image {image_id}: {members}")
        selected_members[image_id] = members[0]

    return selected_members


def extract_nifti_member(zip_file: ZipFile, member_name: str, output_dir: Path) -> Path:
    """Extract one NIfTI archive member into the image staging folder."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(member_name).name
    with zip_file.open(member_name) as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return output_path


def extract_cohort_nifti_files(
    image_list: list[ImageMetadata],
    *,
    input_cohort: str,
    staging_temp_dir: Path,
) -> None:
    cohort_images = [img for img in image_list if img.input_cohort == input_cohort]
    if not cohort_images:
        return

    cohort_zip_path = get_nifti_archive_path(input_cohort)
    requested_image_ids = {img.image_id for img in cohort_images}

    with ZipFile(cohort_zip_path, "r") as cohort_zip:
        members_by_image_id = get_zip_nifti_members_by_image_id(cohort_zip, requested_image_ids)
        for image_metadata in cohort_images:
            member_name = members_by_image_id[image_metadata.image_id]
            extracted_path = extract_nifti_member(
                cohort_zip,
                member_name,
                staging_temp_dir / image_metadata.image_id,
            )
            image_metadata.input_image_name = extracted_path.name

    print(f"Extracted {len(cohort_images)} NIfTI files for input cohort {input_cohort} to staging directory")


def perform_n4_bias_correction(input_image_path: str | Path, output_image_path: str | Path) -> None:
    """Performs N4 bias correction on the input image.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the corrected image.
    """
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    img = sitk.ReadImage(str(input_image_path), sitk.sitkFloat32)
    mask: sitk.Image = sitk.OtsuThreshold(img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected: sitk.Image = corrector.Execute(img, mask)

    sitk.WriteImage(corrected, str(output_image_path))


def preprocess_images_for_output_cohort(output_cohort: int) -> None:

    image_list = get_image_list(output_cohort)
    if not image_list:
        print(f"No images found for output cohort {output_cohort}")
        return

    PREPROCESSED_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    archive_base_path = get_preprocessed_archive_base_path(output_cohort)

    with tempfile.TemporaryDirectory() as staging_tmp, tempfile.TemporaryDirectory() as output_tmp:
        staging_temp_dir = Path(staging_tmp)
        output_temp_dir = Path(output_tmp)

        # Unzip NIfTI files to temp dir `staging_temp_dir`
        for input_cohort in sorted({img.input_cohort for img in image_list}):
            extract_cohort_nifti_files(
                image_list,
                input_cohort=input_cohort,
                staging_temp_dir=staging_temp_dir,
            )

        total_images = len(image_list)
        print(f"Starting N4 bias correction for {total_images} images in output cohort {output_cohort}")
        for completed_count, image_metadata in enumerate(image_list, start=1):
            n4_input_path = image_metadata.get_staged_input_image_path(staging_temp_dir)
            output_path = image_metadata.get_output_image_path(output_temp_dir)
            perform_n4_bias_correction(n4_input_path, output_path)
            if completed_count % 50 == 0:
                print(
                    f"Completed N4 bias correction for {completed_count}/{total_images} "
                    f"images in output cohort {output_cohort}"
                )

        print(f"Completed N4 bias correction for {total_images}/{total_images} images in output cohort {output_cohort}")

        shutil.make_archive(
            base_name=str(archive_base_path),
            format="zip",
            root_dir=output_temp_dir,
        )
        print(f"Wrote preprocessed archive to {archive_base_path}.zip")


def main(output_cohorts: list[int]) -> None:
    for output_cohort in output_cohorts:
        preprocess_images_for_output_cohort(output_cohort)


if __name__ == "__main__":
    main([1])
