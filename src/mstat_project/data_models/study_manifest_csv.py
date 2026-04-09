from datetime import date

import patito as pt
import polars as pl

from mstat_project.data_models._base import BaseFileDataModel


class StudyManifestRaw(BaseFileDataModel):
    image_id: str = pt.Field(
        dtype=pl.String,
        unique=True,
        description="Unique identifier for the image, stripped of leading/trailing whitespace and converted to uppercase",
    )
    subject_id: str = pt.Field(
        dtype=pl.String,
        description="Unique identifier for the subject, stripped of leading/trailing whitespace and converted to uppercase",
    )
    study_id: str = pt.Field(
        dtype=pl.String, description="stripped of leading/trailing whitespace and converted to uppercase"
    )
    series_id: str = pt.Field(
        dtype=pl.String,
        description="Unique identifier for the cohort/series the image is a part of, stripped of leading/trailing whitespace and converted to uppercase",
    )
    image_visit: str = pt.Field(
        dtype=pl.String,
        description="Visit code when the image as taken, each study uses different cadences and image codenaming schemes, stripped of leading/trailing whitespace and converted to uppercase",
    )
    image_date: date = pt.Field(dtype=pl.Date, description="Date that the image was taken as a string")
    image_description: str = pt.Field(dtype=pl.String, description="Describes the MRI sequencing used")
