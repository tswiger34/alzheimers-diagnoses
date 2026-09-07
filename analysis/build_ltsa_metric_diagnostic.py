"""Build the LTSA-versus-baseline metric diagnostic artifacts.

The database work is read-only. Run from the repository root with::

    uv run python analysis/build_ltsa_metric_diagnostic.py
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from mstat_project.utils import get_db_engine


BASELINE_RUN_ID = "075f26f2-e43e-452a-9854-08d75fccdce1"
LTSA_RUN_ID = "68f1b6de-f2b0-4faa-a674-e6e19ccc9735"
LTSA_SOURCE_RUN_ID = "07335dac-a57c-46f9-994f-4790af799221"

OUTPUT_DIR = Path(__file__).resolve().parent / "ltsa_metric_diagnostic"
ARTIFACT_PATH = OUTPUT_DIR / "artifact.json"
NOTEBOOK_PATH = OUTPUT_DIR / "ltsa_metric_diagnostic.ipynb"

PARAMS = {
    "baseline_run_id": BASELINE_RUN_ID,
    "ltsa_run_id": LTSA_RUN_ID,
    "ltsa_source_run_id": LTSA_SOURCE_RUN_ID,
}


RUN_SQL = """
WITH baseline AS (
    SELECT
        'Legacy Cox baseline'::text AS model,
        r.run_id::text AS run_id,
        'All follow-up; baseline-to-outcome time'::text AS task,
        'Event-averaged Cox partial NLL'::text AS loss_definition,
        quoted.epoch AS quoted_epoch,
        quoted.validation_loss::double precision AS quoted_validation_loss,
        quoted.validation_c_index::double precision AS quoted_validation_c_index,
        r.best_epoch AS selected_epoch,
        selected.validation_loss::double precision AS selected_validation_loss,
        selected.validation_c_index::double precision AS selected_validation_c_index,
        r.test_c_index::double precision AS test_c_index,
        r.train_patients,
        r.validation_patients,
        r.test_patients,
        r.train_events,
        r.validation_events,
        r.test_events
    FROM _ml.baseline_runs AS r
    JOIN _ml.baseline_epoch_metrics AS quoted
      ON quoted.run_id = r.run_id AND quoted.epoch = 15
    JOIN _ml.baseline_epoch_metrics AS selected
      ON selected.run_id = r.run_id AND selected.epoch = r.best_epoch
    WHERE r.run_id = :baseline_run_id
), ltsa AS (
    SELECT
        'LTSA at 60 months'::text AS model,
        r.run_id::text AS run_id,
        'At risk after month 60; residual time'::text AS task,
        'Mean discrete-hazard NLL + next-visit MSE'::text AS loss_definition,
        quoted.epoch AS quoted_epoch,
        quoted.validation_total_loss::double precision AS quoted_validation_loss,
        quoted.validation_c_index::double precision AS quoted_validation_c_index,
        r.best_epoch AS selected_epoch,
        selected.validation_total_loss::double precision AS selected_validation_loss,
        selected.validation_c_index::double precision AS selected_validation_c_index,
        r.test_c_index::double precision AS test_c_index,
        r.train_patients,
        r.validation_patients,
        r.test_patients,
        r.train_events,
        r.validation_events,
        r.test_events
    FROM _ml.survival_runs AS r
    JOIN _ml.survival_epoch_metrics AS quoted
      ON quoted.run_id = r.run_id AND quoted.epoch = 45
    JOIN _ml.survival_epoch_metrics AS selected
      ON selected.run_id = r.run_id AND selected.epoch = r.best_epoch
    WHERE r.run_id = :ltsa_run_id
)
SELECT * FROM baseline
UNION ALL
SELECT * FROM ltsa
ORDER BY quoted_epoch
""".strip()


EPOCH_SQL = """
SELECT
    'Legacy Cox baseline'::text AS model,
    epoch,
    validation_loss::double precision AS validation_loss,
    validation_c_index::double precision AS validation_c_index
FROM _ml.baseline_epoch_metrics
WHERE run_id = :baseline_run_id
UNION ALL
SELECT
    'LTSA at 60 months'::text AS model,
    epoch,
    validation_total_loss::double precision AS validation_loss,
    validation_c_index::double precision AS validation_c_index
FROM _ml.survival_epoch_metrics
WHERE run_id IN (:ltsa_source_run_id, :ltsa_run_id)
ORDER BY model, epoch
""".strip()


COHORT_SQL = """
WITH cohorts AS (
    SELECT
        'Legacy Cox baseline'::text AS model,
        ptid,
        split,
        observed_time_months,
        event_observed
    FROM _ml.baseline_run_patients
    WHERE run_id = :baseline_run_id
    UNION ALL
    SELECT
        'LTSA at 60 months'::text AS model,
        ptid,
        split,
        observed_time_months,
        event_observed
    FROM _ml.survival_run_patients
    WHERE run_id = :ltsa_run_id
), counts AS (
    SELECT
        model,
        split,
        count(*) AS patients,
        count(*) FILTER (WHERE event_observed) AS events
    FROM cohorts
    WHERE split IN ('validation', 'test')
    GROUP BY model, split
), pairs AS (
    SELECT
        earlier.model,
        earlier.split,
        count(*) AS comparable_pairs
    FROM cohorts AS earlier
    JOIN cohorts AS later
      ON later.model = earlier.model
     AND later.split = earlier.split
     AND earlier.event_observed
     AND earlier.observed_time_months < later.observed_time_months
    WHERE earlier.split IN ('validation', 'test')
    GROUP BY earlier.model, earlier.split
)
SELECT counts.model, counts.split, counts.patients, counts.events,
       pairs.comparable_pairs
FROM counts
JOIN pairs USING (model, split)
ORDER BY counts.split DESC, counts.model
""".strip()


AUDIT_SQL = """
WITH shared AS (
    SELECT
        legacy.ptid,
        legacy.split AS baseline_split,
        ltsa.split AS ltsa_split,
        legacy.image_id AS baseline_image_id,
        ltsa.selected_image_id AS ltsa_image_id,
        legacy.observed_time_months AS baseline_observed_time,
        legacy.event_observed AS baseline_event,
        images.months_since_baseline_image::double precision AS baseline_scan_month
    FROM _ml.survival_run_patients AS ltsa
    JOIN _ml.baseline_run_patients AS legacy
      ON legacy.run_id = :baseline_run_id
     AND legacy.ptid = ltsa.ptid
    JOIN _core.core_image_set AS images
      ON images.ptid = legacy.ptid
     AND images.image_id::text = legacy.image_id
    WHERE ltsa.run_id = :ltsa_run_id
), all_baseline_scans AS (
    SELECT
        patients.observed_time_months,
        images.months_since_baseline_image::double precision AS scan_month
    FROM _ml.baseline_run_patients AS patients
    JOIN _core.core_image_set AS images
      ON images.ptid = patients.ptid
     AND images.image_id::text = patients.image_id
    WHERE patients.run_id = :baseline_run_id
), baseline_scan_correlation AS (
    SELECT corr(scan_month, observed_time_months) AS value
    FROM all_baseline_scans
), baseline_early_events AS (
    SELECT count(*) AS value
    FROM _ml.baseline_run_patients
    WHERE run_id = :baseline_run_id
      AND split = 'validation'
      AND event_observed
      AND observed_time_months <= 60
)
SELECT
    count(*) AS shared_patients,
    count(*) FILTER (WHERE baseline_image_id = ltsa_image_id) AS same_selected_image,
    count(*) FILTER (WHERE baseline_scan_month > 60) AS baseline_scans_after_month_60,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY baseline_scan_month)
        AS median_baseline_scan_month,
    count(*) FILTER (WHERE baseline_split <> ltsa_split) AS changed_split,
    count(*) FILTER (
        WHERE ltsa_split = 'validation' AND baseline_split = 'validation'
    ) AS ltsa_validation_also_baseline_validation,
    count(*) FILTER (WHERE ltsa_split = 'validation') AS ltsa_validation_patients,
    max(baseline_early_events.value) AS baseline_validation_events_by_month_60,
    max(baseline_scan_correlation.value) AS full_baseline_scan_outcome_correlation
FROM shared
CROSS JOIN baseline_scan_correlation
CROSS JOIN baseline_early_events
GROUP BY ()
""".strip()


def _clean(value: Any) -> Any:
    """Convert database and pandas scalars to portable JSON values."""

    if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _read_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = get_db_engine()
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("SET TRANSACTION READ ONLY"))
                runs = pd.read_sql_query(text(RUN_SQL), connection, params=PARAMS)
                epochs = pd.read_sql_query(text(EPOCH_SQL), connection, params=PARAMS)
                cohorts = pd.read_sql_query(text(COHORT_SQL), connection, params=PARAMS)
                audit = pd.read_sql_query(text(AUDIT_SQL), connection, params=PARAMS)
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
    return runs, epochs, cohorts, audit


def _source(
    source_id: str,
    label: str,
    *,
    path: str | None = None,
    href: str | None = None,
    sql: str | None = None,
    description: str | None = None,
    executed_at: str | None = None,
    tables_used: list[str] | None = None,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {"id": source_id, "label": label}
    if path:
        source["path"] = path
    if href:
        source["href"] = href
    if sql:
        source["query"] = {
            "engine": "PostgreSQL",
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": tables_used or [],
            "metric_definitions": metric_definitions or [],
        }
    return source


def _build_artifact(
    runs: pd.DataFrame,
    epochs: pd.DataFrame,
    cohorts: pd.DataFrame,
    audit: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    baseline = runs.loc[runs["model"] == "Legacy Cox baseline"].iloc[0]
    ltsa = runs.loc[runs["model"] == "LTSA at 60 months"].iloc[0]
    audit_row = audit.iloc[0]

    pair_lookup = {
        (row["model"], row["split"]): int(row["comparable_pairs"])
        for _, row in cohorts.iterrows()
    }
    run_rows = _records(runs)
    for row in run_rows:
        row["validation_comparable_pairs"] = pair_lookup[(row["model"], "validation")]

    headline = [{
        "baseline_quoted_c_index": float(baseline["quoted_validation_c_index"]),
        "ltsa_quoted_c_index": float(ltsa["quoted_validation_c_index"]),
        "ltsa_selected_c_index": float(ltsa["selected_validation_c_index"]),
        "ltsa_validation_events": int(ltsa["validation_events"]),
    }]

    audit_rows = [
        {
            "check": "Loss optimized",
            "legacy_baseline": "Event-averaged Cox partial NLL",
            "ltsa_60m": "Patient-mean discrete NLL + auxiliary MSE",
            "why_it_matters": "The numeric loss scales have no common zero or denominator.",
        },
        {
            "check": "Population and clock",
            "legacy_baseline": "2,636 patients; time from baseline",
            "ltsa_60m": "529 month-60 survivors; residual time after month 60",
            "why_it_matters": "LTSA excludes early converters and answers a later, smaller task.",
        },
        {
            "check": "MRI information boundary",
            "legacy_baseline": "Latest valid MRI across follow-up",
            "ltsa_60m": "MRI history capped at month 60",
            "why_it_matters": "495/529 shared patients use a post-landmark legacy scan.",
        },
        {
            "check": "Validation identities",
            "legacy_baseline": "Historical split snapshot",
            "ltsa_60m": "Current split snapshot",
            "why_it_matters": "Only 18/74 LTSA validation patients were baseline validation patients.",
        },
        {
            "check": "Risk summarized for C-index",
            "legacy_baseline": "Cox log-risk",
            "ltsa_60m": "Negative restricted mean survival",
            "why_it_matters": "Both signs are correct, but the scores target different summaries.",
        },
    ]

    sources = [
        _source(
            "run_metrics_sql",
            "Exact matching training runs and checkpoints",
            path="analysis/build_ltsa_metric_diagnostic.py",
            sql=RUN_SQL,
            description="Matches the two quoted epoch lines and loads selected-checkpoint metrics.",
            executed_at=generated_at,
            tables_used=[
                "_ml.baseline_runs",
                "_ml.baseline_epoch_metrics",
                "_ml.survival_runs",
                "_ml.survival_epoch_metrics",
            ],
            metric_definitions=[
                "Quoted C-index is the validation Harrell C-index stored for the logged epoch.",
                "Selected checkpoint is the persisted best_epoch for each run.",
            ],
        ),
        _source(
            "epoch_sql",
            "Validation metric trajectories",
            path="analysis/build_ltsa_metric_diagnostic.py",
            sql=EPOCH_SQL,
            description="Loads the baseline curve and the complete resumed LTSA lineage.",
            executed_at=generated_at,
            tables_used=["_ml.baseline_epoch_metrics", "_ml.survival_epoch_metrics"],
        ),
        _source(
            "cohort_sql",
            "Validation/test support and comparable-pair counts",
            path="analysis/build_ltsa_metric_diagnostic.py",
            sql=COHORT_SQL,
            description="Counts patients, observed events, and ordered Harrell-comparable pairs.",
            executed_at=generated_at,
            tables_used=["_ml.baseline_run_patients", "_ml.survival_run_patients"],
            metric_definitions=[
                "An ordered pair is comparable when the shorter observed time is an event.",
            ],
        ),
        _source(
            "audit_sql",
            "Cohort, scan-timing, and split-overlap audit",
            path="analysis/build_ltsa_metric_diagnostic.py",
            sql=AUDIT_SQL,
            description="Audits shared patients, selected MRI timing, and persisted split assignments.",
            executed_at=generated_at,
            tables_used=[
                "_ml.baseline_run_patients",
                "_ml.survival_run_patients",
                "_core.core_image_set",
            ],
        ),
        _source(
            "loss_code",
            "Local Cox and discrete survival losses",
            path="packages/ltsa/src/ltsa/losses.py",
        ),
        _source(
            "training_code",
            "Local LTSA loss composition and checkpoint selection",
            path="src/mstat_project/ml/ltsa.py",
        ),
        _source(
            "landmark_code",
            "Local fixed-landmark cohort and risk score",
            path="src/mstat_project/ml/landmarks.py",
        ),
        _source(
            "baseline_code",
            "Legacy latest-valid-MRI baseline",
            path="src/mstat_project/ml/baseline.py",
        ),
        _source(
            "comparison_code",
            "Paired fixed-landmark comparison workflow",
            path="src/mstat_project/ml/compare.py",
        ),
        _source(
            "split_code",
            "Train/validation/test split generation",
            path="src/mstat_project/data_prep/data_splitting.py",
        ),
        _source(
            "nature_paper",
            "Zhang et al., Longitudinal transformer for survival analysis",
            href="https://www.nature.com/articles/s41746-024-01207-4",
        ),
    ]

    manifest_sources = [
        {key: value for key, value in source.items() if key in {"id", "label", "path", "href"}}
        for source in sources
    ]

    title = "Why the LTSA-60 loss is lower while its C-index is worse"
    epoch20 = epochs.loc[
        (epochs["model"] == "LTSA at 60 months") & (epochs["epoch"] == 20)
    ].iloc[0]

    technical_summary = f"""## Technical summary

The loss values are **not comparable across these runs**. The legacy baseline optimizes an event-averaged Cox partial negative log-likelihood; LTSA optimizes a patient-mean discrete-time survival likelihood plus a small next-visit reconstruction term. Cox loss also grows with risk-set size, so a value around 4–6 is unsurprising for hundreds or thousands of patients. A smaller LTSA loss therefore does not mean a better model.

The quoted C-indices also come from **different prediction tasks and validation sets**. The baseline evaluates all follow-up from study baseline, while LTSA retains only patients observed beyond month 60 and predicts residual time. The baseline validation set has {int(baseline['validation_patients']):,} patients, {int(baseline['validation_events'])} events, and {pair_lookup[('Legacy Cox baseline', 'validation')]:,} comparable pairs; LTSA has {int(ltsa['validation_patients'])}, {int(ltsa['validation_events'])}, and {pair_lookup[('LTSA at 60 months', 'validation')]:,}. The LTSA estimate is consequently much noisier.

Finally, neither log line is the selected checkpoint. Baseline selected epoch {int(baseline['selected_epoch'])} at validation C-index {float(baseline['selected_validation_c_index']):.4f}. LTSA selected epoch {int(ltsa['selected_epoch'])} at {float(ltsa['selected_validation_c_index']):.4f}; epoch 45 is the early-stopping tail, ten non-improving epochs later."""

    key_findings = f"""## Key findings

- **Likelihood and ranking can move independently.** C-index asks whether patients are ordered correctly; likelihood also judges the probability assigned to event and censoring times. In this LTSA lineage, epoch 20 has lower validation loss ({float(epoch20['validation_loss']):.4f}) but much worse C-index ({float(epoch20['validation_c_index']):.4f}) than selected epoch 35 (loss {float(ltsa['selected_validation_loss']):.4f}, C-index {float(ltsa['selected_validation_c_index']):.4f}).
- **Month 60 creates a harder, smaller task.** Most early converters disappear. Of the legacy baseline's 41 validation events, 34 occur by month 60 and therefore cannot appear as later events in the fixed-landmark task.
- **The legacy MRI input is later.** For {int(audit_row['baseline_scans_after_month_60'])}/{int(audit_row['shared_patients'])} shared patients, the legacy baseline selected a valid MRI after month 60 (median selected month {float(audit_row['median_baseline_scan_month']):.2f}); LTSA caps its history at month 60. These are valid, generally pre-diagnosis images for converters—not literal post-diagnosis label leakage—but the selection uses patient-specific future follow-up relative to the landmark. That likely makes the legacy task easier; the size of the effect requires a paired rerun.
- **The split snapshot drifted.** {int(audit_row['changed_split'])}/{int(audit_row['shared_patients'])} shared patients changed split. Only {int(audit_row['ltsa_validation_also_baseline_validation'])}/{int(audit_row['ltsa_validation_patients'])} LTSA validation patients were also in baseline validation, so the two validation C-indices do not even rank mostly the same people."""

    chart_read = """The trajectory chart is useful only **within each run**. It shows that the quoted LTSA epoch is a degraded endpoint, while its selected checkpoint is materially better. It should not be read as a fair cross-model curve because the cohorts and risk targets differ."""

    scope = """## Scope, data, and metric definitions

- **Baseline scope:** legacy run over all 2,636 patients; latest valid MRI across each patient's observed follow-up; outcome clock starts at baseline.
- **LTSA-60 scope:** 529 patients still observed and event-free through the landmark; longitudinal MRI prefix ends at month 60; outcome is residual time after month 60.
- **C-index:** both local workflows use the same Harrell pair-ordering implementation and the risk directions are correct. Cox emits log-risk; LTSA uses negative restricted mean survival, so earlier predicted failure means higher risk in both cases.
- **Loss:** baseline Cox loss is averaged over observed events and depends on the risk sets. LTSA loss is averaged over patients and combines discrete-hazard NLL with auxiliary next-visit MSE. Absolute values cannot be compared across those objectives."""

    paper_alignment = """## Methodology and paper alignment

The local month-60 workflow is an adaptation, not a direct reproduction of the [published LTSA evaluation](https://www.nature.com/articles/s41746-024-01207-4). In the paper, landmark time is an evaluation time for one absolute-time model; risk is horizon-specific conditional incidence, and time-dependent C-index is reported across landmark/horizon combinations. Checkpoints are selected by mean validation C-index across those combinations. The paper's single-image baseline also uses the same discrete survival head/loss as LTSA.

Locally, month 60 defines a separately filtered survivor cohort and residual-time task; LTSA is ranked by a full-follow-up Harrell C-index using negative restricted mean survival, while the quoted legacy baseline uses Cox loss on all follow-up. That difference is statistically meaningful and explains why the paper's loss or C-index expectations cannot be transferred directly to these two log lines."""

    methodology = """## Methodology

I matched each four-decimal log line to its persisted epoch row, joined it to run metadata and the selected checkpoint, reconstructed the resumed LTSA epoch lineage, counted exact Harrell-comparable pairs from stored patient snapshots, and audited patient/split/image overlap by `ptid`. All database statements ran in a read-only transaction. I then traced the local Cox loss, LTSA loss composition, fixed-landmark cohort construction, risk score, and checkpoint rules, and compared them with the paper's published methods."""

    limitations = """## Limitations and robustness

- This diagnosis compares one legacy baseline run and one resumed LTSA run; it does not estimate run-to-run variation.
- No uncertainty interval was persisted for either C-index. With only nine LTSA validation events and 382 comparable pairs, a paired bootstrap or repeated seeds is important.
- The database proves split assignments changed, but not exactly when or why. The split-generation code replaces the table and is order-sensitive, which is a plausible mechanism rather than a proven historical cause.
- The post-landmark scan audit establishes a temporal input mismatch. Its contribution to the baseline's C-index is an inference until the fixed-landmark baseline is trained on the identical cohort and MRI cutoff."""

    next_steps = """## Recommended next steps

1. Run the repository's paired workflow: `uv run python -m mstat_project.ml.compare --landmarks 60`. It freezes one cohort, uses the same split snapshot and month-60 input boundary, aligns test predictions, and computes a paired comparison.
2. Treat discrimination and likelihood/calibration as separate model-selection views. Report Harrell or time-dependent C-index alongside a horizon-specific Brier score or calibrated NLL; never rank Cox and LTSA by raw loss magnitude.
3. Persist an immutable split version (or deterministically sort patient IDs before seeded shuffling) so reruns cannot silently reassign patients.
4. If paper fidelity matters, add the paper's conditional horizon risk and time-dependent C-index at explicit horizons after month 60, then select checkpoints using the same landmark/horizon rule.
5. Repeat with several seeds and report events, censoring, comparable pairs, and paired bootstrap intervals for every comparison."""

    further_questions = """## Further questions

- Does a paired month-60 Cox baseline retain the apparent advantage once it is restricted to the same 529 patients and MRI history?
- Which horizon matters clinically—12, 24, 60, or 96 months after the landmark? Negative restricted mean survival can hide horizon-specific ranking differences.
- Are the large LTSA validation swings reproducible across seeds, or primarily a consequence of the nine-event validation sample?
- Would freezing split manifests and adding a minimum-event rule per landmark materially stabilize model selection?"""

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "A source-backed diagnostic of the legacy baseline and LTSA month-60 training results.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "baseline_quoted_c",
                    "description": "Legacy baseline validation C-index at the quoted epoch 15.",
                    "dataset": "headline_metrics",
                    "sourceId": "run_metrics_sql",
                    "metrics": [{"label": "Baseline quoted C-index", "field": "baseline_quoted_c_index", "format": "number"}],
                },
                {
                    "id": "ltsa_quoted_c",
                    "description": "LTSA validation C-index at the quoted early-stopping epoch 45.",
                    "dataset": "headline_metrics",
                    "sourceId": "run_metrics_sql",
                    "metrics": [{"label": "LTSA quoted C-index", "field": "ltsa_quoted_c_index", "format": "number"}],
                },
                {
                    "id": "ltsa_selected_c",
                    "description": "LTSA validation C-index at selected epoch 35.",
                    "dataset": "headline_metrics",
                    "sourceId": "run_metrics_sql",
                    "metrics": [{"label": "LTSA selected C-index", "field": "ltsa_selected_c_index", "format": "number"}],
                },
                {
                    "id": "ltsa_validation_events",
                    "description": "Observed events supporting the LTSA validation C-index.",
                    "dataset": "headline_metrics",
                    "sourceId": "run_metrics_sql",
                    "metrics": [{"label": "LTSA validation events", "field": "ltsa_validation_events", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "validation_c_curve",
                    "title": "Validation C-index by epoch",
                    "subtitle": "Within-run trajectory only; the models use different cohorts and targets.",
                    "type": "line",
                    "dataset": "epoch_curve",
                    "sourceId": "epoch_sql",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                        "y": {"field": "validation_c_index", "type": "quantitative", "label": "Validation C-index"},
                        "color": {"field": "model", "type": "nominal", "label": "Run"},
                        "tooltip": [
                            {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                            {"field": "validation_c_index", "type": "quantitative", "label": "C-index", "format": "number"},
                        ],
                    },
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "run_comparison",
                    "title": "Exact matched runs and selected checkpoints",
                    "subtitle": "Raw loss values are displayed for identification, not cross-model ranking.",
                    "dataset": "run_comparison",
                    "sourceId": "run_metrics_sql",
                    "defaultSort": {"field": "quoted_epoch", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "Run", "type": "text"},
                        {"field": "quoted_epoch", "label": "Quoted epoch", "type": "number"},
                        {"field": "quoted_validation_loss", "label": "Quoted val loss", "type": "number"},
                        {"field": "quoted_validation_c_index", "label": "Quoted val C", "type": "number"},
                        {"field": "selected_epoch", "label": "Selected epoch", "type": "number"},
                        {"field": "selected_validation_c_index", "label": "Selected val C", "type": "number"},
                        {"field": "test_c_index", "label": "Test C", "type": "number"},
                        {"field": "validation_patients", "label": "Val patients", "type": "number"},
                        {"field": "validation_events", "label": "Val events", "type": "number"},
                        {"field": "validation_comparable_pairs", "label": "Comparable pairs", "type": "number"},
                    ],
                },
                {
                    "id": "cohort_support",
                    "title": "Statistical support behind each C-index",
                    "subtitle": "Exact patient, event, and ordered comparable-pair counts.",
                    "dataset": "cohort_support",
                    "sourceId": "cohort_sql",
                    "defaultSort": {"field": "comparable_pairs", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "Run", "type": "text"},
                        {"field": "split", "label": "Split", "type": "text"},
                        {"field": "patients", "label": "Patients", "type": "number"},
                        {"field": "events", "label": "Events", "type": "number"},
                        {"field": "comparable_pairs", "label": "Comparable pairs", "type": "number"},
                    ],
                },
                {
                    "id": "comparability_audit",
                    "title": "Why the logged values are not a paired model comparison",
                    "subtitle": "Differences verified from persisted cohorts and local training code.",
                    "dataset": "comparability_audit",
                    "sourceId": "audit_sql",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "legacy_baseline", "label": "Legacy baseline", "type": "text"},
                        {"field": "ltsa_60m", "label": "LTSA-60", "type": "text"},
                        {"field": "why_it_matters", "label": "Implication", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}\n\nGenerated {generated_at}. Database queries were read-only."},
                {"id": "headline", "type": "metric-strip", "cardIds": ["baseline_quoted_c", "ltsa_quoted_c", "ltsa_selected_c", "ltsa_validation_events"]},
                {"id": "technical_summary", "type": "markdown", "body": technical_summary},
                {"id": "run_table", "type": "table", "tableId": "run_comparison"},
                {"id": "key_findings", "type": "markdown", "body": key_findings},
                {"id": "trajectory", "type": "chart", "chartId": "validation_c_curve"},
                {"id": "trajectory_read", "type": "markdown", "body": chart_read},
                {"id": "scope", "type": "markdown", "body": scope},
                {"id": "cohort_table", "type": "table", "tableId": "cohort_support"},
                {"id": "audit_table", "type": "table", "tableId": "comparability_audit"},
                {"id": "paper_alignment", "type": "markdown", "body": paper_alignment},
                {"id": "methodology", "type": "markdown", "body": methodology},
                {"id": "limitations", "type": "markdown", "body": limitations},
                {"id": "next_steps", "type": "markdown", "body": next_steps},
                {"id": "further_questions", "type": "markdown", "body": further_questions},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": headline,
                "run_comparison": run_rows,
                "epoch_curve": _records(epochs),
                "cohort_support": _records(cohorts),
                "comparability_audit": audit_rows,
            },
        },
        "sources": sources,
    }
    return artifact


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _build_notebook() -> dict[str, Any]:
    cells = [
        _markdown_cell(
            "# LTSA-60 versus legacy baseline: reproducible metric diagnostic\n\n"
            "This notebook re-runs the read-only database evidence used in the portable report."
        ),
        _code_cell(
            "import pandas as pd\n"
            "from sqlalchemy import text\n"
            "from mstat_project.utils import get_db_engine\n\n"
            f"PARAMS = {PARAMS!r}\n"
            "engine = get_db_engine()\n"
            "connection = engine.connect()\n"
            "transaction = connection.begin()\n"
            "connection.execute(text('SET TRANSACTION READ ONLY'))\n"
            "print('Connected in a read-only transaction.')\n"
        ),
        _markdown_cell("## Exact run and checkpoint match"),
        _code_cell(
            f"RUN_SQL = {RUN_SQL!r}\n"
            "runs = pd.read_sql_query(text(RUN_SQL), connection, params=PARAMS)\n"
            "print(runs.to_string(index=False))\n"
        ),
        _markdown_cell("## Validation trajectories and metric decoupling"),
        _code_cell(
            f"EPOCH_SQL = {EPOCH_SQL!r}\n"
            "epochs = pd.read_sql_query(text(EPOCH_SQL), connection, params=PARAMS)\n"
            "milestones = []\n"
            "for model, group in epochs.groupby('model'):\n"
            "    min_loss = group.loc[group['validation_loss'].idxmin()]\n"
            "    max_c = group.loc[group['validation_c_index'].idxmax()]\n"
            "    milestones.append({'model': model, 'minimum-loss epoch': int(min_loss['epoch']), "
            "'minimum loss': min_loss['validation_loss'], 'C at minimum loss': min_loss['validation_c_index'], "
            "'maximum-C epoch': int(max_c['epoch']), 'maximum C': max_c['validation_c_index']})\n"
            "print(pd.DataFrame(milestones).to_string(index=False))\n"
        ),
        _markdown_cell("## Cohort support and comparable pairs"),
        _code_cell(
            f"COHORT_SQL = {COHORT_SQL!r}\n"
            "cohorts = pd.read_sql_query(text(COHORT_SQL), connection, params=PARAMS)\n"
            "print(cohorts.to_string(index=False))\n"
        ),
        _markdown_cell("## Scan timing and split-overlap audit"),
        _code_cell(
            f"AUDIT_SQL = {AUDIT_SQL!r}\n"
            "audit = pd.read_sql_query(text(AUDIT_SQL), connection, params=PARAMS)\n"
            "print(audit.to_string(index=False))\n"
        ),
        _markdown_cell(
            "## Interpretation\n\n"
            "The raw losses are incommensurate, the C-indices are supported by very different "
            "numbers of events/pairs, and the cohorts differ in target clock, MRI information boundary, "
            "and validation membership. The correct next experiment is the paired fixed-landmark workflow."
        ),
        _code_cell(
            "transaction.rollback()\n"
            "connection.close()\n"
            "engine.dispose()\n"
            "print('Closed the read-only transaction.')\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _execute_notebook(notebook: dict[str, Any]) -> None:
    """Execute code cells in order without requiring nbformat/nbclient."""

    namespace: dict[str, Any] = {"__name__": "__main__"}
    execution_count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        source = "".join(cell["source"])
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, f"notebook-cell-{execution_count}", "exec"), namespace)
        except Exception as exc:
            captured = stdout.getvalue()
            outputs: list[dict[str, Any]] = []
            if captured:
                outputs.append({"output_type": "stream", "name": "stdout", "text": captured})
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
            raise
        captured = stdout.getvalue()
        cell["outputs"] = (
            [{"output_type": "stream", "name": "stdout", "text": captured}]
            if captured
            else []
        )
        cell["execution_count"] = execution_count


def main() -> None:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    runs, epochs, cohorts, audit = _read_frames()
    if len(runs) != 2 or len(audit) != 1:
        raise RuntimeError("Expected two exact run matches and one audit row.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _build_artifact(runs, epochs, cohorts, audit, generated_at)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    notebook = _build_notebook()
    _execute_notebook(notebook)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {ARTIFACT_PATH}")
    print(f"Wrote and executed {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
