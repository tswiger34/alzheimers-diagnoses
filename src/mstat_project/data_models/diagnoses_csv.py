import re
from datetime import date
from typing import Self

import patito as pt
import polars as pl
from polars.dataframe.frame import DataFrame

from mstat_project.data_models._base import BaseFileDataModel
from mstat_project.utils import clean_str_col_exprs  # noqa


class PatientViscodeDiagnoses(BaseFileDataModel):
    """Raw DXSUM schema.

    Many branch-dependent fields use ``-4`` as a sentinel value for missing/null values.
    Multi-select checkbox responses are stored as pipe-delimited strings such as ``1|03``.
    Date and timestamp columns remain strings here because the raw extract includes quoted
    blanks that do not cast cleanly under the shared base loader.
    """

    # IDs
    ptid_viscode: str = pt.Field(
        unique=True,
        derived_from=pl.concat_str(exprs=[pl.col(name="ptid"), pl.col(name="viscode")], separator="|").hash(),
    )
    obs_id: str = pt.Field(
        unique=True,
        derived_from=pl.concat_str(exprs=[pl.col(name="ptid"), pl.col(name="examdate")], separator="|").hash(),
    )
    ptid: str = pt.Field(dtype=pl.String, description="Participant ID.")
    id: int = pt.Field(dtype=pl.Int32, description="Record ID.")
    rid: int = pt.Field(dtype=pl.Int16, description="Participant roster ID.")

    # Time Metadata
    phase: str = pt.Field(dtype=pl.String, description="ADNI study phase.")
    viscode: str = pt.Field(dtype=pl.String, description="Visit code.")
    viscode2: str | None = pt.Field(dtype=pl.String, description="Translated visit code.")
    examdate: str | None = pt.Field(
        dtype=pl.String, description="Date form completed / examination date. Raw value format: YYYY-MM-DD."
    )

    # Diagnostics
    diagnosis: int | None = pt.Field(
        dtype=pl.Int8, description="Current diagnosis. Codes: 1=CN; 2=MCI; 3=Dementia."
    )
    dxnorm: int | None = pt.Field(dtype=pl.Int8, description="Normal. Codes: 1=Yes.")
    dxmci: int | None = pt.Field(dtype=pl.Int8, description="Mild cognitive impairment. Codes: 1=Yes.")
    dxmdue: int | None = pt.Field(
        dtype=pl.Int8,
        description=(
            "Suspected cause of MCI. Codes: 1=MCI due to Alzheimer's Disease; 2=MCI due to other etiology."
        ),
    )
    dxdsev: int | None = pt.Field(
        dtype=pl.Int8, description="Dementia severity, clinician impression. Codes: 1=Mild; 2=Moderate; 3=Severe."
    )
    dxddue: int | None = pt.Field(
        dtype=pl.Int8,
        description=(
            "Suspected cause of dementia. Codes: 1=Dementia due to Alzheimer's Disease;"
            "2=Dementia due to other etiology."
        ),
    )
    dxad: int | None = pt.Field(
        dtype=pl.Int8, description="Alzheimer's Disease. Used as the label for images Codes: 1=Yes."
    )
    dxapp: int | None = pt.Field(
        dtype=pl.Int8,
        description="If dementia is due to Alzheimer's Disease, indicate likelihood. Codes: 1=Probable; 2=Possible.",
    )
    dxconfid: int | None = pt.Field(
        dtype=pl.Int8,
        description=(
            "Physician confidence in diagnosis. Codes: 1=Uncertain; "
            "2=Mildly Confident; 3=Moderately Confident; 4=Highly Confident."
        ),
    )
    has_qc_error: int | None = pt.Field(
        dtype=pl.Int8,
        description="Has quality check error. Codes: 0=Does not have QC error or QC error has been approved; 1=Has QC error.",
    )

    # Update Timestamp Fields
    userdate: str = pt.Field(dtype=pl.String, description="Date record created. Raw value format: YYYY-MM-DD.")
    userdate2: str | None = pt.Field(
        dtype=pl.String, description="Date record last updated. Raw value format: YYYY-MM-DD."
    )
    update_stamp: str = pt.Field(
        dtype=pl.String, description="Source extract update timestamp. Raw value format: YYYY-MM-DD HH:MM:SS."
    )

    # Derived Fields
    is_keep: bool = pt.Field(
        dtype=pl.Boolean, derived_from=pl.col(name="obs_id").cum_count().over(partition_by="obs_id") == 1
    )

    @classmethod
    def from_file(cls, file_path: str) -> pt.DataFrame[Self]:  # ty: ignore[invalid-method-override]
        schema_overrides: dict[str, pl.DataType] = {  # ty: ignore[invalid-assignment]
            "PHASE": pl.String,
            "PTID": pl.String,
            "VISCODE": pl.String,
            "VISCODE2": pl.String,
            "EXAMDATE": pl.String,
            "DXMDES": pl.String,
            "DXMOTHET": pl.String,
            "DXAPROB": pl.String,
            "DXAPOSS": pl.String,
            "USERDATE": pl.String,
            "USERDATE2": pl.String,
            "DD_CRF_VERSION_LABEL": pl.String,
            "LANGUAGE_CODE": pl.String,
            "update_stamp": pl.String,
        }

        # Load data and clean/norm string cols
        df1: pl.DataFrame = pl.read_csv(
            source=file_path, schema_overrides=schema_overrides, null_values=["", "-4"]
        )
        df2: DataFrame = df1.with_columns(*clean_str_col_exprs(raw_df=df1))

        rename_dict: dict[str, str] = {col: col.lower() for col in df2.columns}
        df3: pl.DataFrame = df2.rename(mapping=rename_dict)

        RAW_BASELINE_VIS_CODE_NAMES: set[str] = {"BL", "4_INIT", "4_SC", "INIT", "SC", "SCMRI"}
        df4: pl.DataFrame = df3.with_columns(
            pl.col(name="viscode").str.replace(
                pattern="^(?:" + "|".join(map(re.escape, RAW_BASELINE_VIS_CODE_NAMES)) + ")$",
                value="BL",
            ),
        )

        df5: pt.DataFrame[Self] = pt.DataFrame(data=pt.DataFrame(data=df4).set_model(model=cls).cast().derive())

        # Drop extra cols, some records are duplicated due to updates, get the most updated record
        df6: pl.DataFrame = (
            df5.drop_nulls(subset="examdate")
            .drop([col for col in df5.columns if col not in cls.columns])
            .sort(by=["ptid_viscode", "update_stamp", "userdate", "userdate2"])
        )

        # Derive columns and
        df7: pt.DataFrame[Self] = (
            pt.DataFrame(data=df6.group_by("ptid_viscode").first())
            .set_model(model=cls)
            .cast()
            .derive()
            .filter(predicate=pl.col(name="is_keep"))
        )
        df7.validate()
        return df7


class PatientDiagnosesSummary(pt.Model):
    ptid: str = pt.Field(dtype=pl.String, description="Participant ID.", unique=True)
    time_to_event: int | None = pt.Field(
        dtype=pl.Int64, description="Time to AD diagnosis in months, None if patient is censored"
    )
    first_observed_dt: date = pt.Field(dtype=pl.Date, description="Date of the patient's first exam")
    last_observed_dt: date = pt.Field(dtype=pl.Date, description="Date of the patient's last exam")
    n_visits: int = pt.Field(dtype=pl.Int32, description="Number of visits patient has attended")
    diagnosis_at_start: int | None = pt.Field(
        dtype=pl.Int8, description="Patient's code for the patient at the first visit"
    )
    diagnosis_at_end: int | None = pt.Field(
        dtype=pl.Int8, description="Patient's Diagnosis code at the last visit"
    )
    is_censored: bool = pt.Field(
        dtype=pl.Boolean, description="Whether or not the patient's time-to-event is censored"
    )
    img_label: str = pt.Field(
        dtype=pl.Int32,
        description="Time to event value casted as a string to be used as the image label",
        derived_from=pl.col("time_to_event").cast(pl.String),
    )

    @classmethod
    def from_pt_viscode(cls, df: pt.DataFrame[PatientViscodeDiagnoses]) -> pt.DataFrame[Self]:
        month_delta_expr: pl.Expr = (
            (pl.col("event_dt").dt.year() - pl.col("first_observed_dt").dt.year()) * 12
            + (pl.col("event_dt").dt.month() - pl.col("first_observed_dt").dt.month())
            - pl.when(pl.col("event_dt").dt.day() < pl.col("first_observed_dt").dt.day()).then(1).otherwise(0)
        )
        cols_to_drop = [col for col in df.columns if col not in cls.columns]

        summary_df: pl.DataFrame = (
            df.with_columns(examdate_dt=pl.col("examdate").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
            .drop_nulls(subset="examdate_dt")
            .sort(by=["ptid", "examdate_dt"])
            .group_by("ptid", maintain_order=True)
            .agg(
                first_observed_dt=pl.col("examdate_dt").first(),
                last_observed_dt=pl.col("examdate_dt").last(),
                n_visits=pl.n_unique("viscode").cast(pl.Int32),
                diagnosis_at_start=pl.col("diagnosis").first(),
                diagnosis_at_end=pl.col("diagnosis").last(),
                event_dt=pl.col("examdate_dt").filter(pl.col("dxad") == 1).min(),
            )
            .with_columns(
                is_censored=pl.col("event_dt").is_null(),
                time_to_event=pl.when(pl.col("event_dt").is_null())
                .then(None)
                .otherwise(month_delta_expr.cast(pl.Int64)),
            )
            .drop(cols_to_drop)
        )

        return pt.DataFrame(data=summary_df).set_model(model=cls).cast().derive().validate()
