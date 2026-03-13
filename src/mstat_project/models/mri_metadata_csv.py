import re
from datetime import date
from typing import Literal, Optional, Self

import patito as pt
import polars as pl

from mstat_project.utils import clean_str_col_exprs  # noqa


class MRIMetadataRaw(pt.Model):
    image_id: str = pt.Field(dtype=pl.String)
    subject_id: str = pt.Field(dtype=pl.String)
    image_visit: str = pt.Field(dtype=pl.String)
    image_date: str = pt.Field(dtype=pl.String)
    series_type: str = pt.Field(dtype=pl.String)
    mri_protocol_phase: str = pt.Field(dtype=pl.String)
    series_description: str = pt.Field(dtype=pl.String)
    acceleration: str = pt.Field(dtype=pl.String)
    acquisition_type: str = pt.Field(dtype=pl.String)
    acquisition_plane: str = pt.Field(dtype=pl.String)
    number_volumes: str = pt.Field(dtype=pl.String)
    slices_per_volume: str = pt.Field(dtype=pl.String)
    slice_thickness: str = pt.Field(dtype=pl.String)
    scanner_manufacturer: str = pt.Field(dtype=pl.String)
    scanner_model: str = pt.Field(dtype=pl.String)
    software_version: str = pt.Field(dtype=pl.String)
    magnetic_field_strength: str = pt.Field(dtype=pl.String)
    receive_coil_name: Optional[str] = pt.Field(dtype=pl.String)
    study_instance_uid: str = pt.Field(dtype=pl.String)
    series_instance_uid: str = pt.Field(dtype=pl.String)
    loni_study: str = pt.Field(dtype=pl.String)
    loni_series: str = pt.Field(dtype=pl.String)
    loni_image: str = pt.Field(dtype=pl.String)

    @classmethod
    def read_file(cls) -> pt.DataFrame[Self]:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        data_dir: str = os.getenv(key="DATA_DIR", default="data")
        file_name: str = os.getenv(key="FILE_NAME_MRI_METADATA", default="")

        df: pl.DataFrame = pl.read_csv(source=f"{data_dir}/files/{file_name}")
        return pt.DataFrame(data=df).set_model(model=cls)


class MRIMetadataCleaned(pt.Model):
    # Primary key
    obs_id: str = pt.Field(
        unique=True,
        derived_from=pl.concat_str(
            exprs=[pl.col(name="subject_id"), pl.col(name="image_date")], separator="|"
        ).hash(),
    )

    # Fields from raw file
    image_id: str = pt.Field(dtype=pl.String, unique=True)
    subject_id: str = pt.Field(dtype=pl.String)
    image_visit: str = pt.Field(dtype=pl.String)
    image_date: date = pt.Field(dtype=pl.Date)
    series_type: Literal["T1W"] = pt.Field(dtype=pl.String)
    mri_protocol_phase: Literal["ADNI1", "ADNI2", "ADNI3", "ADNI4", "ADNIGO", "ADNIGO/ADNI2"] = pt.Field(
        dtype=pl.String
    )
    series_description: str = pt.Field(dtype=pl.String)
    acceleration: Literal["ACCELERATED", "UNACCELERATED", "ULTRAFAST"] = pt.Field(dtype=pl.String)
    acquisition_type: Literal["3D"] = pt.Field(dtype=pl.String)
    acquisition_plane: Literal["SAGITTAL"] = pt.Field(dtype=pl.String)
    number_volumes: int = pt.Field(dtype=pl.Int16)
    slices_per_volume: int = pt.Field(dtype=pl.Int32)
    slice_thickness: float = pt.Field(dtype=pl.Float16)

    # Flag fields
    is_possible_dupe: bool = pt.Field(dtype=pl.Boolean, derived_from=pl.struct(["obs_id"]).is_duplicated())
    is_repeat_scan: bool = pt.Field(
        dtype=pl.Boolean,
        derived_from=pl.col(name="series_description").str.contains(pattern=".*REPEAT.*|.*REPET.*"),
    )
    is_accelerated: bool = pt.Field(
        dtype=pl.Boolean,
        derived_from=pl.col(name="acceleration").str.contains(pattern=r"^(ACCELERATED|ULTRAFAST)$"),
    )
    is_first_in_seq: bool = pt.Field(
        dtype=pl.Boolean,
        derived_from=pl.col(name="image_date") == pl.col(name="image_date").min().over(partition_by="subject_id"),
    )
    is_last_in_seq: bool = pt.Field(
        dtype=pl.Boolean,
        derived_from=pl.col(name="image_date") == pl.col(name="image_date").max().over(partition_by="subject_id"),
    )
    is_keep: bool = pt.Field(
        dtype=pl.Boolean, derived_from=pl.col(name="obs_id").cum_count().over(partition_by="obs_id") == 1
    )

    # Lagged fields
    time_elapsed_prev_visit_days: int = pt.Field(
        dtype=pl.Int64,
        derived_from=pl.col("image_date")
        .diff()
        .over(partition_by="subject_id")
        .dt.total_days()
        .fill_null(value=0),
    )

    time_elapsed_first_visit_days: int = pt.Field(
        dtype=pl.Int64,
        derived_from=(
            pl.col("image_date") - pl.col(name="image_date").min().over(partition_by="subject_id")
        ).dt.total_days(),
    )

    @staticmethod
    def _prep_raw_df(raw_df: pt.DataFrame[MRIMetadataRaw]) -> pt.DataFrame:
        """Preps the raw data frame for cleaning by normalizing string columns and drop unused columns

        Args:
            raw_df (pt.DataFrame[MRIMetadataRaw]): The raw data frame

        Returns:
            pt.DataFrame: The prepped raw data frame
        """
        RAW_BASELINE_VIS_CODE_NAMES: set[str] = {"BL", "4_INIT", "4_SC", "INIT", "SC", "SCMRI"}

        clean_df: pl.DataFrame = raw_df.with_columns(*clean_str_col_exprs(raw_df=raw_df))

        # Normalize viscode baseline values to use 'BL'
        clean_df: pl.DataFrame = clean_df.with_columns(
            pl.col(name="image_visit").str.replace(
                pattern="^(?:" + "|".join(map(re.escape, RAW_BASELINE_VIS_CODE_NAMES)) + ")$",
                value="BL",
            ),
        )

        cols_to_drop: list[str] = [
            "scanner_manufacturer",
            "scanner_model",
            "software_version",
            "magnetic_field_strength",
            "receive_coil_name",
            "study_instance_uid",
            "series_instance_uid",
            "loni_study",
            "loni_series",
            "loni_image",
        ]

        return clean_df.set_model(model=MRIMetadataRaw).cast().derive().validate().drop(columns=cols_to_drop)

    @classmethod
    def from_raw_df(cls, raw_df: pt.DataFrame[MRIMetadataRaw]) -> pt.DataFrame[Self]:
        """Takes in the un-prepped raw data frame, calls :method:``_prep_raw_df``, then uses
        :class:``MRIMetadataCleaned`` to derive, filter, clean, and validate the data
        """
        prepped_df: pt.DataFrame[Self] = (
            cls._prep_raw_df(raw_df=raw_df)
            .set_model(model=cls)
            .cast()
            .derive()
            .filter(predicate=pl.col(name="is_keep"))
        )

        return prepped_df.validate()

    @classmethod
    def from_file(cls) -> pt.DataFrame[Self]:
        """Reads the MRI metadata csv file, then preps, cleans, transforms, and validates the data.

        Returns:
            pt.DataFrame[Self]: The validated and cleaned MRI metadata as a data frame
        """
        raw_df: pt.DataFrame[MRIMetadataRaw] = MRIMetadataRaw.read_file()

        return cls.from_raw_df(raw_df=raw_df)


if __name__ == "__main__":
    raw_df = MRIMetadataRaw.read_file()
    clean_df1 = MRIMetadataCleaned.from_file()
    clean_df2 = MRIMetadataCleaned.from_raw_df(raw_df=raw_df)

    print(raw_df.head())
    print(clean_df1.head())
    print(clean_df2.head())
