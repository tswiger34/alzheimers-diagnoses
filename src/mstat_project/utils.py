import os
from typing import Annotated, Literal, TypeAlias, TypedDict

import sqlalchemy
import torch
from dotenv import load_dotenv


def get_db_engine() -> sqlalchemy.Engine:
    load_dotenv()
    user = os.getenv(key="POSTGRES_USER", default="postgres")
    password = os.getenv(key="POSTGRES_PASSWORD")
    db_name = os.getenv(key="POSTGRES_DB_NAME", default="mstat-db")
    host = os.getenv(key="POSTGRES_HOST", default="127.0.0.1")
    port = os.getenv(key="POSTGRES_PORT", default="5434")

    if password is None:
        raise ValueError("No password set for connecting to Postgres DB")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{int(port)}/{db_name}"
    return sqlalchemy.create_engine(url)


Diagnosis = Literal["CN", "MCI", "AD"]

Sex = Literal["M", "F"]

MagneticFieldStrength = Literal["1.5T", "3.0T"]


MRISequenceTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: (n_visits, 1, D, H, W)",
    "dtype: torch.float32",
    "description: Chronologically ordered MRI tensor sequence for one participant.",
]

VisitFloatTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: (n_visits,)",
    "dtype: torch.float32",
]

VisitBoolTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: (n_visits,)",
    "dtype: torch.bool",
]

VisitLongTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: (n_visits,)",
    "dtype: torch.int64",
]

ScalarFloatTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: ()",
    "dtype: torch.float32",
]

ScalarBoolTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: ()",
    "dtype: torch.bool",
]

ScalarLongTensor: TypeAlias = Annotated[
    torch.Tensor,
    "shape: ()",
    "dtype: torch.int64",
]


class ADNISubjectTensorDictMin(TypedDict):
    """Minimal subject level info

    This structure stores the minimum information needed to identify a subject
    and their MRI scans. It is used for indexing and retrieving subject data.

    Attributes:
        ptid (str): ADNI participant identifier.

        img_ids (list[str]): List of scan identifiers for the subject's MRI scans.

        images (MRISequenceTensor | None): Chronologically ordered MRI sequence with shape
            ``(n_visits, 1, D, H, W)``. For a T1-only model, the channel
            dimension is 1. If using multiple modalities, this structure should
            be updated to allow more channels.

        months_since_prior_mri (VisitFloatTensor): Time gap in months from the previous MRI.
            The first visit should usually be 0. Shape: ``(n_visits,)``.

        months_since_baseline_mri (VisitFloatTensor): Time gap in months from the baseline MRI.
            Shape: ``(n_visits,)``.

        time_to_event_from_baseline (ScalarFloatTensor): Time to event or censoring, measured in
            months from baseline. Shape: ``()``.

        time_to_event_from_mri (VisitFloatTensor): Time to event or censoring, measured in
            months from each MRI visit. Shape: ``(n_visits,)``.

        dx_code_at_visit (VisitLongTensor): Diagnosis code at each visit, either 0 ("CN"), 1 ("MCI"), or 2 ("AD").
            Shape: ``(n_visits,)``.

        age_at_baseline (ScalarFloatTensor): Scalar participant age at baseline visit.

        age_at_image (VisitFloatTensor): Participant age at each MRI visit. Shape: ``(n_visits,)``.

        is_censored (ScalarBoolTensor): Scalar boolean indicating whether the participant is censored.

    """

    ptid: str
    img_ids: list[str]
    images: MRISequenceTensor | None
    months_since_prior_mri: VisitFloatTensor
    months_since_baseline_mri: VisitFloatTensor
    time_to_event_from_baseline: ScalarFloatTensor
    time_to_event_from_mri: VisitFloatTensor
    dx_code_at_visit: VisitLongTensor
    age_at_baseline: ScalarFloatTensor
    age_at_image: VisitFloatTensor
    is_censored: ScalarBoolTensor


class ADNISubjectTensorDictFull(TypedDict):
    """Subject-level tensor package for longitudinal ADNI survival modeling.

    This structure stores all MRI visits and aligned metadata for a single
    ADNI participant. The first dimension of every visit-level tensor must
    correspond to the same chronological MRI visit order as the ``image``
    tensor.

    The intended modeling setup is dynamic survival prediction using MRI
    history up to each valid landmark visit. The image sequence and
    time-varying covariates may be passed to the model, while survival label
    fields should be used only for loss computation, masking, and evaluation.

    Attributes:
        image: Chronologically ordered MRI sequence with shape
            ``(n_visits, 1, D, H, W)``. For a T1-only model, the channel
            dimension is 1. If using multiple modalities, this structure should
            be updated to allow more channels.

        months_since_baseline_mri: Visit time in months from the participant's
            baseline visit. Shape: ``(n_visits,)``.

        months_since_prior_mri: Time gap in months from the previous MRI.
            The first visit should usually be 0. Shape: ``(n_visits,)``.

        age_at_visit: Participant age at each MRI visit. Shape:
            ``(n_visits,)``.

        visit_mask: Boolean mask identifying real, non-padded visits. Shape:
            ``(n_visits,)``.

        event_observed: Scalar boolean indicating whether the participant
            experienced the target event, such as conversion to AD.

        observed_time_months: Scalar time to event or censoring, measured in
            months from baseline.

        duration_from_landmark: Time from each visit landmark to the observed
            event or censoring time. Shape: ``(n_visits,)``.

        event_from_landmark: Boolean event indicator from each landmark visit.
            For censored participants, all valid entries should usually be
            False. Shape: ``(n_visits,)``.

        valid_landmark: Boolean mask identifying visits that are eligible as
            survival prediction landmarks. Visits at or after AD diagnosis
            should usually be False. Shape: ``(n_visits,)``.

        participant_id: ADNI participant identifier.

        image_id: Ordered scan identifiers aligned to the first dimension of
            ``image``.

        dx_code_at_visit: Diagnosis code at each visit, either 0 ("CN"), 1 ("MCI"), or 2 ("AD").
            Shape: ``(n_visits,)``.

        baseline_dx: Scalar baseline diagnosis code.

        sex: Scalar sex code. Use a consistent project-level encoding, such as
            ``0 = female`` and ``1 = male``, or store an explicit mapping
            elsewhere.

        education_years: ScalarFloatTensor indicating years of education for the participant.

    """

    img_ids: list[str]
    ptid: str
    images: MRISequenceTensor

    months_since_baseline_mri: VisitFloatTensor
    months_since_prior_mri: VisitFloatTensor
    age_at_visit: VisitFloatTensor
    visit_mask: VisitBoolTensor

    event_observed: ScalarBoolTensor
    observed_time_months: ScalarFloatTensor
    duration_from_landmark: VisitFloatTensor
    event_from_landmark: VisitBoolTensor
    valid_landmark: VisitBoolTensor

    dx_code_at_visit: VisitLongTensor
    baseline_dx: ScalarLongTensor
    sex: ScalarLongTensor
    education_years: ScalarFloatTensor
