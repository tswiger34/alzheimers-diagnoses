import asyncio
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy

from mstat_project.utils import get_db_engine

COHORTS = list(range(1, 13))
DEFAULT_NIFTI_WORKERS = 16


@dataclass(frozen=True)
class DicomFolderResult:
    """Result from processing one image folder."""

    image_id: str
    success: bool
    skipped: bool
    dicom_count: int
    error: str | None = None


def _resolve_max_workers(max_workers: int | None = DEFAULT_NIFTI_WORKERS) -> int:
    """Resolve the configured worker count."""

    if max_workers is None:
        max_workers = min(DEFAULT_NIFTI_WORKERS, os.cpu_count() or 1)

    if max_workers < 1:
        msg = f"max_workers must be at least 1, got {max_workers}"
        raise ValueError(msg)

    return max_workers


def _has_existing_output(output_dir: Path) -> bool:
    """Return True when an image output directory already contains generated files."""

    return output_dir.exists() and any(path.is_file() for path in output_dir.iterdir())


def _find_selected_dicom_members(zip_path: Path, selected_ids: set[str]) -> dict[str, list[str]]:
    """Map selected ADNI image IDs to DICOM member names in a cohort zip."""

    members_by_image_id: dict[str, list[str]] = {}

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            if member_path.suffix.lower() != ".dcm":
                continue

            # Example:
            # ADNI/.../2025-04-09_09_12_52.0/I11193011/file.dcm
            image_id = member_path.parent.name.removeprefix("I")
            if image_id not in selected_ids:
                continue

            members_by_image_id.setdefault(image_id, []).append(member.filename)

    return members_by_image_id


def _extract_dicom_members(zip_path: Path, member_names: list[str], output_dir: Path) -> None:
    """Extract a selected image folder's DICOM files from the cohort zip."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        for member_name in member_names:
            out_path = output_dir / Path(member_name).name
            with zf.open(member_name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


async def dicom_to_nifti_async(dicom_dir: str | Path, output_dir: str | Path) -> None:
    """Convert a folder of DICOM files to NIfTI without blocking the event loop."""

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

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode:
        output = "\n".join(chunk.decode(errors="replace").strip() for chunk in (stdout, stderr) if chunk.strip())
        msg = f"dcm2niix failed with exit code {process.returncode}"
        if output:
            msg = f"{msg}: {output}"
        raise RuntimeError(msg)


async def process_dicom_folder_async(
    *,
    image_id: str,
    member_names: list[str],
    zip_path: Path,
    temp_base_dir: Path,
    output_base_dir: Path,
    overwrite: bool = False,
) -> DicomFolderResult:
    """Extract and convert one selected DICOM image folder."""

    image_output_dir = output_base_dir / image_id
    if not overwrite and _has_existing_output(image_output_dir):
        print(f"Skipping {image_id}: output already exists")
        return DicomFolderResult(
            image_id=image_id,
            success=True,
            skipped=True,
            dicom_count=len(member_names),
        )

    image_extract_dir = temp_base_dir / image_id

    try:
        await asyncio.to_thread(_extract_dicom_members, zip_path, member_names, image_extract_dir)
        print(f"Converting {image_id}: {len(member_names)} DICOM files")

        await dicom_to_nifti_async(
            dicom_dir=image_extract_dir,
            output_dir=image_output_dir,
        )
        return DicomFolderResult(
            image_id=image_id,
            success=True,
            skipped=False,
            dicom_count=len(member_names),
        )
    except Exception as exc:
        error = str(exc)
        print(f"Failed to process {image_id}: {error}")
        return DicomFolderResult(
            image_id=image_id,
            success=False,
            skipped=False,
            dicom_count=len(member_names),
            error=error,
        )
    finally:
        await asyncio.to_thread(shutil.rmtree, image_extract_dir, ignore_errors=True)


async def process_cohort_async(
    cohort: int,
    selected_ids: set[str],
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
    archive: bool = True,
) -> set[str]:
    """Convert selected ADNI image IDs from a cohort zip to NIfTI asynchronously.

    Only extracts DICOM files belonging to selected image folders.
    """

    print(f"Processing Cohort: {cohort}")

    data_path = Path(os.getenv("IMAGES_PATH", "data/raw_images"))
    zfile_path = data_path / f"cohort_{str(cohort).zfill(2)}.zip"
    output_base_dir = Path(f"data/images/nifti/cohort_{str(cohort).zfill(2)}")
    worker_count = _resolve_max_workers(max_workers)

    members_by_image_id = _find_selected_dicom_members(zfile_path, selected_ids)
    print(f"Found {len(members_by_image_id)} selected image folders in zip")

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        queue: asyncio.Queue[tuple[str, list[str]] | None] = asyncio.Queue()
        results: list[DicomFolderResult] = []

        for image_id, member_names in members_by_image_id.items():
            queue.put_nowait((image_id, member_names))

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return

                    image_id, member_names = item
                    result = await process_dicom_folder_async(
                        image_id=image_id,
                        member_names=member_names,
                        zip_path=zfile_path,
                        temp_base_dir=temp_dir,
                        output_base_dir=output_base_dir,
                        overwrite=overwrite,
                    )
                    results.append(result)
                finally:
                    queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        for _ in tasks:
            queue.put_nowait(None)

        await queue.join()
        await asyncio.gather(*tasks)

    processed_ids = {result.image_id for result in results if result.success}
    skipped_count = sum(result.skipped for result in results)
    failed_results = [result for result in results if not result.success]
    converted_count = len(processed_ids) - skipped_count

    print(
        f"Cohort {cohort} complete: {converted_count} converted, "
        f"{skipped_count} skipped, {len(failed_results)} failed"
    )
    if failed_results:
        failed_ids = ", ".join(result.image_id for result in failed_results)
        print(f"Failed image IDs: {failed_ids}")

    if archive and output_base_dir.exists():
        await asyncio.to_thread(
            shutil.make_archive,
            base_name=str(output_base_dir),
            format="zip",
            root_dir=output_base_dir,
        )
    if os.path.exists(output_base_dir) and not os.path.isfile(output_base_dir):
        await asyncio.to_thread(shutil.rmtree, path=output_base_dir)
    return processed_ids


def process_cohort(
    cohort: int,
    selected_ids: set[str],
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
    archive: bool = True,
) -> set[str]:
    """Convert selected ADNI image IDs from a cohort zip to NIfTI."""

    return asyncio.run(
        process_cohort_async(
            cohort=cohort,
            selected_ids=selected_ids,
            max_workers=max_workers,
            overwrite=overwrite,
            archive=archive,
        )
    )


def process_all(
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
    archive: bool = True,
) -> set[str]:

    # Get selected image IDs
    db_eng = get_db_engine()
    with db_eng.connect() as conn:
        result = conn.execute(statement=sqlalchemy.text(text="SELECT image_id FROM _core.core_image_set")).all()
    ids = set(str(row[0]) for row in result)
    for cohort in COHORTS:
        imgs_processed = process_cohort(
            cohort,
            selected_ids=ids,
            max_workers=max_workers,
            overwrite=overwrite,
            archive=archive,
        )
        ids = ids - imgs_processed

    return ids


if __name__ == "__main__":
    ids = process_all()
    print(len(ids))
    if len(ids) > 0:
        print(ids)
