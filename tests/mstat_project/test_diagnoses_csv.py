from datetime import date

import patito as pt
import polars as pl

from mstat_project.data_models.diagnoses_csv import PatientDiagnosesSummary
from mstat_project.data_models.diagnoses_csv import PatientViscodeDiagnoses


def _make_pt_viscode_df(rows: list[dict[str, object]]) -> pt.DataFrame[PatientViscodeDiagnoses]:
    prepared_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        prepared_rows.append(
            {
                "ptid": row["ptid"],
                "id": index,
                "rid": index,
                "phase": "ADNI1",
                "viscode": row.get("viscode", f"m{index:02d}"),
                "viscode2": row.get("viscode2"),
                "examdate": row["examdate"],
                "diagnosis": row.get("diagnosis"),
                "dxnorm": row.get("dxnorm"),
                "dxmci": row.get("dxmci"),
                "dxmdue": row.get("dxmdue"),
                "dxdsev": row.get("dxdsev"),
                "dxddue": row.get("dxddue"),
                "dxad": row.get("dxad"),
                "dxapp": row.get("dxapp"),
                "dxconfid": row.get("dxconfid"),
                "has_qc_error": row.get("has_qc_error", 0),
                "userdate": row.get("userdate", row["examdate"]),
                "userdate2": row.get("userdate2"),
                "update_stamp": row.get("update_stamp", f"{row['examdate']} 00:00:00"),
            }
        )

    return pt.DataFrame(data=pl.DataFrame(prepared_rows)).set_model(model=PatientViscodeDiagnoses).cast().derive()


def test_from_pt_viscode_returns_null_time_to_event_for_censored_patient() -> None:
    diagnoses_df = _make_pt_viscode_df(
        [
            {"ptid": "PT001", "viscode": "BL", "examdate": "2020-01-15", "diagnosis": 1, "dxnorm": 1},
            {"ptid": "PT001", "viscode": "m06", "examdate": "2020-07-15", "diagnosis": 2, "dxmci": 1},
        ]
    )

    summary_df = PatientDiagnosesSummary.from_pt_viscode(diagnoses_df)

    assert summary_df.to_dicts() == [
        {
            "ptid": "PT001",
            "time_to_event": None,
            "first_observed_dt": date(2020, 1, 15),
            "last_observed_dt": date(2020, 7, 15),
            "n_visits": 2,
            "diagnosis_at_start": 1,
            "diagnosis_at_end": 2,
            "is_censored": True,
        }
    ]


def test_from_pt_viscode_uses_first_ad_event_and_calendar_months() -> None:
    diagnoses_df = _make_pt_viscode_df(
        [
            {"ptid": "PT002", "viscode": "BL", "examdate": "2020-01-15", "diagnosis": 2, "dxmci": 1},
            {"ptid": "PT002", "viscode": "m02", "examdate": "2020-03-14", "diagnosis": 3, "dxad": 1},
            {"ptid": "PT002", "viscode": "m05", "examdate": "2020-06-20", "diagnosis": 3, "dxad": 1},
        ]
    )

    summary_df = PatientDiagnosesSummary.from_pt_viscode(diagnoses_df)

    assert summary_df.to_dicts() == [
        {
            "ptid": "PT002",
            "time_to_event": 1,
            "first_observed_dt": date(2020, 1, 15),
            "last_observed_dt": date(2020, 6, 20),
            "n_visits": 3,
            "diagnosis_at_start": 2,
            "diagnosis_at_end": 3,
            "is_censored": False,
        }
    ]


def test_from_pt_viscode_returns_one_summary_row_per_patient() -> None:
    diagnoses_df = _make_pt_viscode_df(
        [
            {"ptid": "PT003", "viscode": "BL", "examdate": "2021-02-01", "diagnosis": 3, "dxad": 1},
            {"ptid": "PT003", "viscode": "m06", "examdate": "2021-08-01", "diagnosis": 3, "dxad": 1},
            {"ptid": "PT004", "viscode": "BL", "examdate": "2021-03-10", "diagnosis": 1, "dxnorm": 1},
            {"ptid": "PT004", "viscode": "m12", "examdate": "2022-03-10", "diagnosis": 2, "dxmci": 1},
        ]
    )

    summary_rows = sorted(PatientDiagnosesSummary.from_pt_viscode(diagnoses_df).to_dicts(), key=lambda row: row["ptid"])

    assert summary_rows == [
        {
            "ptid": "PT003",
            "time_to_event": 0,
            "first_observed_dt": date(2021, 2, 1),
            "last_observed_dt": date(2021, 8, 1),
            "n_visits": 2,
            "diagnosis_at_start": 3,
            "diagnosis_at_end": 3,
            "is_censored": False,
        },
        {
            "ptid": "PT004",
            "time_to_event": None,
            "first_observed_dt": date(2021, 3, 10),
            "last_observed_dt": date(2022, 3, 10),
            "n_visits": 2,
            "diagnosis_at_start": 1,
            "diagnosis_at_end": 2,
            "is_censored": True,
        },
    ]
