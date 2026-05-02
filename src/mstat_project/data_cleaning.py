import csv  # noqa
import os  # noqa
import pathlib  # noqa
import uuid
import zipfile  # noqa
from dataclasses import dataclass  # noqa
from datetime import datetime  # noqa
from typing import Self


@dataclass(slots=True)
class RunMetadata:
    run_id: str
    run_start_time: datetime
    run_end_time: datetime
    run_total_time_sec: float
    n_files: int
    n_dcm_files: int
    n_non_dcm_files: int | None = None
    other_extensions: list[str] | None = None


@dataclass(slots=True)
class ImageVolumePathMetadata:
    full_img_path: str
    root_folder: str
    n_sub_dirs: int
    file_name: str
    is_dcm: bool
    file_size: int | None = None

    @classmethod
    def from_zip_info(cls, zip_info: zipfile.ZipInfo) -> Self:
        path = pathlib.Path(zip_info.filename)
        root_folder = path.parts[0]
        n_sub_dirs = len(path.parts) - 2
        file_name = path.name
        is_dcm = file_name.endswith(".dcm")
        file_size = zip_info.file_size
        return cls(
            full_img_path=zip_info.filename,
            root_folder=root_folder,
            n_sub_dirs=n_sub_dirs,
            file_name=file_name,
            is_dcm=is_dcm,
            file_size=file_size,
        )

    @staticmethod
    def parse_path_parts(full_img_path: str, n_sub_dirs: int):
        path = pathlib.Path(full_img_path)
        if n_sub_dirs != 4:
            raise ValueError(f"Expected 4 sub directories, but got {n_sub_dirs}")
        root_folder = path.parts[0]
        ptid: str = path.parts[1]
        series_type: str = path.parts[2]
        visit_timestamp: str = path.parts[3]
        img_id: str = path.parts[4]
        img_name: str = path.parts[5]
        # Example file name: ADNI_022_S_0004_MR_MPRAGE_br_raw_20050922154614543_98_S9233_I7273.dcm
        # first part should be ADNI/root folder name
        # second part should be the ptid
        # third part should be the series type
        # fourth part is unknown, e.g. "br_raw"
        # fifth part should be the visit timestamp, e.g. "20050922154614543"
        # sixth part should be the volume number, e.g. "98"
        # seventh part is unknown, e.g. "S9233"
        # eighth part should be the image id, e.g. "I7273"

        ## TODO:
        # get metadata from db using the image id
        # verify visit timestamp matches the image date in the db
        # verify series type matches the series type in the db
        # verify ptid matches the ptid in the db
        return root_folder, ptid, series_type, visit_timestamp, img_id, img_name


@dataclass(slots=True)
class ImageFolderPathMetadata:
    img_folder_path: str
    n_volumes: int
    img_id: str
    img_timestamp: str
    img_series_type: str
    ptid: str
    volume_metadata: list[ImageVolumePathMetadata]


def validate_path_metadata(
    image_folder_path_metadata: ImageFolderPathMetadata, image_volume_metadata: ImageVolumePathMetadata
): ...


def read_image_paths_from_zip(file_name: str):
    run_id = str(uuid.uuid4())
    start_time = datetime.now()
    data_path = os.getenv(key="DATA_PATH", default="data")
    file = pathlib.Path(data_path) / "raw_images" / file_name
    with zipfile.ZipFile(file=file, mode="r") as f:
        files = f.filelist
        dcm_files = [
            ImageVolumePathMetadata.from_zip_info(zip_info=file)
            for file in files
            if file.filename.endswith(".dcm")
        ]
        non_dcm_files = [
            ImageVolumePathMetadata.from_zip_info(zip_info=file)
            for file in files
            if not file.filename.endswith(".dcm")
        ]
        n_sub_dirs = set([file.n_sub_dirs for file in dcm_files])
        print(f"Number of sub directories in DCM files: {n_sub_dirs}")

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    run_metadata = RunMetadata(
        run_id=run_id,
        run_start_time=start_time,
        run_end_time=end_time,
        run_total_time_sec=total_time,
        n_files=len(files),
        n_dcm_files=len(dcm_files),
        n_non_dcm_files=len(non_dcm_files),
        other_extensions=list(set([file.file_name.split(".")[-1] for file in non_dcm_files])),
    )
    return run_metadata


if __name__ == "__main__":
    run_metadata = read_image_paths_from_zip(file_name="cohort_12.zip")
    print(run_metadata)
