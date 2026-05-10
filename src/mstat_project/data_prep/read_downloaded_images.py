import os
import pathlib
import uuid
import zipfile
from dataclasses import dataclass
from typing import Self

import polars as pl
import sqlalchemy

from mstat_project.utils import get_db_engine

COHORTS = {i for i in range(1, 13)}


@dataclass(slots=True)
class ImageVolumePathMetadata:
    run_id: str
    full_img_path: str
    root_folder: str
    n_sub_dirs: int
    file_name: str
    is_dcm: bool
    cohort: str
    file_size: int | None = None

    @classmethod
    def from_zip_info(cls, zip_info: zipfile.ZipInfo, cohort: str, run_id: str) -> Self:
        path = pathlib.Path(zip_info.filename)
        root_folder = path.parts[0]
        n_sub_dirs = len(path.parts) - 2
        file_name = path.name
        is_dcm = file_name.endswith(".dcm")
        file_size = zip_info.file_size
        return cls(
            run_id=run_id,
            full_img_path=zip_info.filename,
            root_folder=root_folder,
            n_sub_dirs=n_sub_dirs,
            file_name=file_name,
            is_dcm=is_dcm,
            cohort=cohort,
            file_size=file_size,
        )

    @classmethod
    def as_polars_schema(cls) -> pl.Schema:
        fields = cls.__annotations__
        return pl.Schema(fields)


def read_image_paths_from_zip(cohort: str):
    run_id = str(uuid.uuid4())
    data_path = os.getenv(key="IMAGES_PATH", default="data/raw_images")
    file_name = f"cohort_{cohort}.zip"
    file = pathlib.Path(data_path) / file_name
    with zipfile.ZipFile(file=file, mode="r") as f:
        files = f.filelist
        dcm_files = [
            ImageVolumePathMetadata.from_zip_info(zip_info=file, cohort=cohort, run_id=run_id)
            for file in files
            if file.filename.endswith(".dcm")
        ]
        non_dcm_files = [
            ImageVolumePathMetadata.from_zip_info(zip_info=file, cohort=cohort, run_id=run_id)
            for file in files
            if not file.filename.endswith(".dcm")
        ]
        all_files = [*dcm_files, *non_dcm_files]
        df = pl.DataFrame(data=all_files, schema=ImageVolumePathMetadata.as_polars_schema())

        db_eng = get_db_engine()
        with db_eng.connect() as conn:
            schema = "_raw"
            table = "raw_mri_download_metadata"
            table_name = f"{schema}.{table}"
            conn.begin()
            check_stmt = sqlalchemy.text(
                text=f"SELECT 1 FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = '{table}'"
            )
            table_exists = conn.execute(statement=check_stmt)
            if table_exists.first():
                delete_stmt = sqlalchemy.text(text=f"DELETE FROM {table_name} WHERE cohort = '{cohort}'")
                conn.execute(statement=delete_stmt)
            df.write_database(table_name, connection=conn, if_table_exists="append")
            conn.commit()

    return None


if __name__ == "__main__":
    for cohort in COHORTS:
        cohort_str = str(object=cohort).zfill(2)
        read_image_paths_from_zip(cohort=cohort_str)
