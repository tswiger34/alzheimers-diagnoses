import asyncio
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import JSONB

from mstat_project.utils import get_db_engine

from .models import NiFTiMetadata
from .utils import IMAGES_PATH, get_cohort_name, resolve_max_workers

load_dotenv()
NIFTI_IMAGE_PATH = IMAGES_PATH / "nifti"
RAW_NIFTI_METADATA_SCHEMA = "_raw"
RAW_NIFTI_METADATA_TABLE = "raw_nifti_metadata"


@dataclass(frozen=True)
class DicomFolderResult:
    """Result from processing one image folder."""

    image_id: str
    success: bool
    skipped: bool
    dicom_count: int
    error: str | None = None


def check_has_existing_output(output_dir: Path) -> bool:
    """Return True when an image output directory already contains generated files."""

    return output_dir.exists() and any(path.is_file() for path in output_dir.iterdir())


def find_selected_dicom_members(zip_path: Path) -> dict[str, list[str]]:
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

            members_by_image_id.setdefault(image_id, []).append(member.filename)

    return members_by_image_id


def extract_dicom_members(zip_path: Path, member_names: list[str], output_dir: Path) -> None:
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
    if not overwrite and check_has_existing_output(image_output_dir):
        print(f"Skipping {image_id}: output already exists")
        return DicomFolderResult(
            image_id=image_id,
            success=True,
            skipped=True,
            dicom_count=len(member_names),
        )

    image_extract_dir = temp_base_dir / image_id

    try:
        await asyncio.to_thread(extract_dicom_members, zip_path, member_names, image_extract_dir)
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


async def process_cohort_async(
    cohort: int,
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
    archive: bool = True,
) -> None:
    """Convert ADNI image IDs from a cohort zip to NIfTI asynchronously."""

    print(f"Processing Cohort: {cohort}")

    zfile_path = IMAGES_PATH / "raw" / f"{get_cohort_name(cohort)}.zip"

    output_base_dir = NIFTI_IMAGE_PATH / get_cohort_name(cohort)
    worker_count = resolve_max_workers(max_workers)

    members_by_image_id = find_selected_dicom_members(zfile_path)
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
    return None


def process_all(
    cohorts: list[int],
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
    archive: bool = True,
) -> None:
    for cohort in cohorts:
        asyncio.run(
            process_cohort_async(
                cohort=cohort,
                max_workers=max_workers,
                overwrite=overwrite,
                archive=archive,
            )
        )

    return None


def _unwrap_optional_annotation(annotation: Any) -> Any:
    """Return the concrete type from Optional[T] style annotations."""

    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return annotation

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) != 1:
        msg = f"Unsupported NIfTI metadata union field type: {annotation!r}"
        raise TypeError(msg)

    return args[0]


def _metadata_column_type(annotation: Any) -> sqlalchemy.types.TypeEngine[Any]:
    """Map NiFTiMetadata annotations to Postgres-compatible SQLAlchemy types."""

    annotation = _unwrap_optional_annotation(annotation)

    if get_origin(annotation) is list:
        return JSONB()
    if annotation is bool:
        return sqlalchemy.Boolean()
    if annotation is int:
        return sqlalchemy.Integer()
    if annotation is float:
        return sqlalchemy.Double()
    if annotation is str:
        return sqlalchemy.Text()

    msg = f"Unsupported NIfTI metadata field type: {annotation!r}"
    raise TypeError(msg)


def create_nifti_metadata_table() -> sqlalchemy.Table:
    """Build the raw NIfTI metadata table definition."""

    metadata = sqlalchemy.MetaData()
    columns = [
        sqlalchemy.Column("image_id", sqlalchemy.Text(), nullable=False),
        sqlalchemy.Column("cohort", sqlalchemy.Text(), nullable=False),
        sqlalchemy.Column("source_json_path", sqlalchemy.Text(), nullable=False),
    ]

    columns.extend(
        sqlalchemy.Column(name, _metadata_column_type(field.annotation), nullable=True)
        for name, field in NiFTiMetadata.model_fields.items()
    )

    return sqlalchemy.Table(
        RAW_NIFTI_METADATA_TABLE,
        metadata,
        *columns,
        schema=RAW_NIFTI_METADATA_SCHEMA,
    )


def _replace_cohort_metadata(records: list[dict[str, Any]], cohort_str: str) -> None:
    """Replace raw NIfTI metadata for one cohort in a single transaction."""

    db_eng = get_db_engine()
    with db_eng.begin() as conn:
        table = create_nifti_metadata_table()
        conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {RAW_NIFTI_METADATA_SCHEMA}"))
        table.create(bind=conn, checkfirst=True)

        conn.execute(table.delete().where(table.c.cohort == cohort_str))

        if not records:
            return
        conn.execute(table.insert(), records)
        return


def get_image_id_from_json_member(member_name: str) -> str:
    """Derive the ADNI image ID from a converted metadata JSON archive member."""

    member_path = Path(member_name)
    image_id = member_path.parent.name
    if not image_id:
        msg = f"Cannot derive image_id from metadata path {member_name!r}"
        raise ValueError(msg)

    return image_id


def get_metadata_record_from_json_text(
    *,
    contents: str,
    source_json_path: str,
    cohort_str: str,
) -> dict[str, Any]:
    """Validate one metadata JSON archive member into a database record."""

    metadata = NiFTiMetadata.model_validate_json(contents, strict=False, extra="allow")
    record = metadata.model_dump(mode="json")
    return {
        "image_id": get_image_id_from_json_member(source_json_path),
        "cohort": cohort_str,
        "source_json_path": source_json_path,
        **record,
    }


async def load_cohort_metadata_records(cohort: int) -> tuple[str, list[dict[str, Any]]]:
    """Extract and validate all metadata records from one converted cohort archive."""

    cohort_str = str(cohort).zfill(2)
    zip_path = NIFTI_IMAGE_PATH / f"{get_cohort_name(cohort)}.zip"

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        json_files = [
            file.filename
            for file in zf.filelist
            if not file.is_dir() and Path(file.filename).suffix.lower() == ".json"
        ]

        records = [
            get_metadata_record_from_json_text(
                contents=zf.read(json_file).decode("utf-8"),
                source_json_path=json_file,
                cohort_str=cohort_str,
            )
            for json_file in json_files
        ]

    return cohort_str, list(records)


async def load_cohort_metadata(cohort: int) -> None:
    cohort_str, records = await load_cohort_metadata_records(cohort)
    print(records)
    await asyncio.to_thread(_replace_cohort_metadata, records, cohort_str)


def load_all_metadata(cohorts: list[int]) -> None:
    """Load raw NIfTI JSON metadata for all requested cohorts."""

    async def load_all() -> None:
        for cohort in cohorts:
            await load_cohort_metadata(cohort)

    asyncio.run(load_all())


def main(cohorts: list[int]) -> None:
    # process_all(cohorts)
    load_all_metadata(cohorts)


if __name__ == "__main__":
    cohorts = list(range(1, 13))
    main(cohorts)
