"""Read-only inspection of survival-training result tables.

Run from the repository root with::

    uv run python analysis/query_training_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from mstat_project.utils import get_db_engine


def main() -> None:
    engine = get_db_engine()
    with engine.connect() as connection:
        tables = list(
            connection.execute(
                text(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema IN ('_ml', 'public')
                    ORDER BY table_schema, table_name
                    """
                )
            ).mappings()
        )
        print("TABLES")
        print(json.dumps([dict(row) for row in tables], default=str, indent=2))

        legacy_runs = list(
            connection.execute(
                text(
                    """
                    SELECT
                        r.run_id,
                        r.started_at,
                        r.completed_at,
                        r.status,
                        r.config,
                        r.train_patients,
                        r.validation_patients,
                        r.test_patients,
                        r.train_events,
                        r.validation_events,
                        r.test_events,
                        r.best_epoch,
                        r.best_validation_loss,
                        r.test_c_index
                    FROM _ml.baseline_runs AS r
                    ORDER BY r.started_at DESC
                    LIMIT 25
                    """
                )
            ).mappings()
        )
        print("LEGACY_BASELINE_RUNS")
        print(json.dumps([dict(row) for row in legacy_runs], default=str, indent=2))

        matching_legacy_epochs = list(
            connection.execute(
                text(
                    """
                    SELECT r.started_at, r.status, r.config, r.train_patients,
                           r.validation_patients, r.test_patients, r.train_events,
                           r.validation_events, r.test_events, m.*
                    FROM _ml.baseline_epoch_metrics AS m
                    JOIN _ml.baseline_runs AS r USING (run_id)
                    WHERE abs(m.train_loss - 5.9579) < 0.001
                       OR abs(m.validation_c_index - 0.8196) < 0.001
                    ORDER BY r.started_at DESC, m.epoch
                    """
                )
            ).mappings()
        )
        print("MATCHING_LEGACY_EPOCHS")
        print(json.dumps([dict(row) for row in matching_legacy_epochs], default=str, indent=2))

        unified_runs = list(
            connection.execute(
                text(
                    """
                    SELECT
                        r.run_id,
                        r.comparison_id,
                        r.model_type,
                        r.landmark_months,
                        r.started_at,
                        r.completed_at,
                        r.status,
                        r.config,
                        r.train_patients,
                        r.validation_patients,
                        r.test_patients,
                        r.train_events,
                        r.validation_events,
                        r.test_events,
                        r.best_epoch,
                        r.best_validation_loss,
                        r.best_validation_c_index,
                        r.test_c_index
                    FROM _ml.survival_runs AS r
                    ORDER BY r.started_at DESC
                    LIMIT 30
                    """
                )
            ).mappings()
        )
        print("UNIFIED_RUNS")
        print(json.dumps([dict(row) for row in unified_runs], default=str, indent=2))

        matching_ltsa_epochs = list(
            connection.execute(
                text(
                    """
                    SELECT r.started_at, r.status, r.config, r.train_patients,
                           r.validation_patients, r.test_patients, r.train_events,
                           r.validation_events, r.test_events, m.*
                    FROM _ml.survival_epoch_metrics AS m
                    JOIN _ml.survival_runs AS r USING (run_id)
                    WHERE r.model_type = 'ltsa'
                      AND r.landmark_months = 60
                      AND (
                          abs(m.train_total_loss - 0.5160) < 0.001
                          OR abs(m.validation_c_index - 0.5524) < 0.001
                      )
                    ORDER BY r.started_at DESC, m.epoch
                    """
                )
            ).mappings()
        )
        print("MATCHING_LTSA_EPOCHS")
        print(json.dumps([dict(row) for row in matching_ltsa_epochs], default=str, indent=2))

        if matching_legacy_epochs and matching_ltsa_epochs:
            legacy_run_id = str(matching_legacy_epochs[0]["run_id"])
            ltsa_run_id = str(matching_ltsa_epochs[0]["run_id"])
            resume_from = matching_ltsa_epochs[0]["config"].get("resume_from")
            source_ltsa_run_id = Path(resume_from).parent.name if resume_from else ltsa_run_id
            params = {
                "legacy_run_id": legacy_run_id,
                "ltsa_run_id": ltsa_run_id,
                "source_ltsa_run_id": source_ltsa_run_id,
            }

            epoch_curves = list(
                connection.execute(
                    text(
                        """
                        SELECT 'legacy_baseline' AS model, epoch,
                               train_loss AS train_total_loss,
                               train_loss AS train_survival_loss,
                               NULL::double precision AS train_auxiliary_loss,
                               train_c_index, validation_loss AS validation_total_loss,
                               validation_loss AS validation_survival_loss,
                               NULL::double precision AS validation_auxiliary_loss,
                               validation_c_index
                        FROM _ml.baseline_epoch_metrics
                        WHERE run_id = :legacy_run_id
                        UNION ALL
                        SELECT 'ltsa_60m' AS model, epoch, train_total_loss,
                               train_survival_loss, train_auxiliary_loss,
                               train_c_index, validation_total_loss,
                               validation_survival_loss, validation_auxiliary_loss,
                               validation_c_index
                        FROM _ml.survival_epoch_metrics
                        WHERE run_id IN (:source_ltsa_run_id, :ltsa_run_id)
                        ORDER BY model, epoch
                        """
                    ),
                    params,
                ).mappings()
            )
            print("EPOCH_CURVES")
            print(json.dumps([dict(row) for row in epoch_curves], default=str, indent=2))

            epoch_extrema = list(
                connection.execute(
                    text(
                        """
                        SELECT 'legacy_baseline' AS model,
                               min(validation_loss) AS minimum_validation_loss,
                               max(validation_c_index) AS maximum_validation_c_index,
                               corr(validation_loss, validation_c_index) AS loss_c_index_correlation
                        FROM _ml.baseline_epoch_metrics
                        WHERE run_id = :legacy_run_id
                        UNION ALL
                        SELECT 'ltsa_60m' AS model,
                               min(validation_total_loss),
                               max(validation_c_index),
                               corr(validation_total_loss, validation_c_index)
                        FROM _ml.survival_epoch_metrics
                        WHERE run_id IN (:source_ltsa_run_id, :ltsa_run_id)
                        """
                    ),
                    params,
                ).mappings()
            )
            print("EPOCH_EXTREMA")
            print(json.dumps([dict(row) for row in epoch_extrema], default=str, indent=2))

            epoch_milestones = list(
                connection.execute(
                    text(
                        """
                        WITH metrics AS (
                            SELECT 'legacy_baseline' AS model, epoch,
                                   validation_loss AS validation_total_loss,
                                   validation_c_index
                            FROM _ml.baseline_epoch_metrics
                            WHERE run_id = :legacy_run_id
                            UNION ALL
                            SELECT 'ltsa_60m' AS model, epoch,
                                   validation_total_loss, validation_c_index
                            FROM _ml.survival_epoch_metrics
                            WHERE run_id IN (:source_ltsa_run_id, :ltsa_run_id)
                        ), ranked AS (
                            SELECT *,
                                   row_number() OVER (
                                       PARTITION BY model ORDER BY validation_total_loss, epoch
                                   ) AS loss_rank,
                                   row_number() OVER (
                                       PARTITION BY model ORDER BY validation_c_index DESC, epoch
                                   ) AS c_index_rank
                            FROM metrics
                        )
                        SELECT model, epoch, validation_total_loss,
                               validation_c_index,
                               CASE
                                   WHEN loss_rank = 1 AND c_index_rank = 1 THEN 'minimum loss and maximum C-index'
                                   WHEN loss_rank = 1 THEN 'minimum loss'
                                   WHEN c_index_rank = 1 THEN 'maximum C-index'
                               END AS milestone
                        FROM ranked
                        WHERE loss_rank = 1 OR c_index_rank = 1
                        ORDER BY model, epoch
                        """
                    ),
                    params,
                ).mappings()
            )
            print("EPOCH_MILESTONES")
            print(json.dumps([dict(row) for row in epoch_milestones], default=str, indent=2))

            cohort_summary = list(
                connection.execute(
                    text(
                        """
                        WITH cohorts AS (
                            SELECT 'legacy_baseline' AS model, split,
                                   observed_time_months, event_observed,
                                   1 AS visit_count
                            FROM _ml.baseline_run_patients
                            WHERE run_id = :legacy_run_id
                            UNION ALL
                            SELECT 'ltsa_60m' AS model, split,
                                   observed_time_months, event_observed,
                                   jsonb_array_length(image_ids) AS visit_count
                            FROM _ml.survival_run_patients
                            WHERE run_id = :ltsa_run_id
                        )
                        SELECT model, split, count(*) AS patients,
                               count(*) FILTER (WHERE event_observed) AS events,
                               count(*) FILTER (WHERE NOT event_observed) AS censored,
                               avg(event_observed::int) AS event_fraction,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY observed_time_months)
                                   AS median_observed_time_months,
                               min(observed_time_months) AS minimum_observed_time_months,
                               max(observed_time_months) AS maximum_observed_time_months,
                               avg(visit_count) AS mean_visits
                        FROM cohorts
                        GROUP BY model, split
                        ORDER BY model, split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("COHORT_SUMMARY")
            print(json.dumps([dict(row) for row in cohort_summary], default=str, indent=2))

            comparable_pairs = list(
                connection.execute(
                    text(
                        """
                        WITH cohorts AS (
                            SELECT 'legacy_baseline' AS model, ptid, split,
                                   observed_time_months, event_observed
                            FROM _ml.baseline_run_patients
                            WHERE run_id = :legacy_run_id
                            UNION ALL
                            SELECT 'ltsa_60m' AS model, ptid, split,
                                   observed_time_months, event_observed
                            FROM _ml.survival_run_patients
                            WHERE run_id = :ltsa_run_id
                        )
                        SELECT left_cohort.model, left_cohort.split,
                               count(*) AS ordered_comparable_pairs
                        FROM cohorts AS left_cohort
                        JOIN cohorts AS right_cohort
                          ON right_cohort.model = left_cohort.model
                         AND right_cohort.split = left_cohort.split
                         AND left_cohort.event_observed
                         AND left_cohort.observed_time_months < right_cohort.observed_time_months
                        GROUP BY left_cohort.model, left_cohort.split
                        ORDER BY left_cohort.model, left_cohort.split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("COMPARABLE_PAIRS")
            print(json.dumps([dict(row) for row in comparable_pairs], default=str, indent=2))

            cohort_overlap = list(
                connection.execute(
                    text(
                        """
                        SELECT ltsa.split,
                               count(*) AS ltsa_patients,
                               count(legacy.ptid) AS also_in_legacy,
                               count(*) FILTER (
                                   WHERE legacy.observed_time_months > 60
                               ) AS legacy_at_risk_after_60,
                               count(*) FILTER (
                                   WHERE legacy.event_observed = ltsa.event_observed
                               ) AS matching_event_flags,
                               max(abs((legacy.observed_time_months - 60) - ltsa.observed_time_months))
                                   AS max_residual_time_difference
                        FROM _ml.survival_run_patients AS ltsa
                        LEFT JOIN _ml.baseline_run_patients AS legacy
                          ON legacy.run_id = :legacy_run_id
                         AND legacy.ptid = ltsa.ptid
                        WHERE ltsa.run_id = :ltsa_run_id
                        GROUP BY ltsa.split
                        ORDER BY ltsa.split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("COHORT_OVERLAP")
            print(json.dumps([dict(row) for row in cohort_overlap], default=str, indent=2))

            split_crosswalk = list(
                connection.execute(
                    text(
                        """
                        SELECT legacy.split AS legacy_split,
                               ltsa.split AS ltsa_split,
                               count(*) AS patients,
                               count(*) FILTER (WHERE ltsa.event_observed) AS events
                        FROM _ml.survival_run_patients AS ltsa
                        JOIN _ml.baseline_run_patients AS legacy
                          ON legacy.run_id = :legacy_run_id
                         AND legacy.ptid = ltsa.ptid
                        WHERE ltsa.run_id = :ltsa_run_id
                        GROUP BY legacy.split, ltsa.split
                        ORDER BY legacy.split, ltsa.split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("SPLIT_CROSSWALK")
            print(json.dumps([dict(row) for row in split_crosswalk], default=str, indent=2))

            legacy_at_risk_counts = list(
                connection.execute(
                    text(
                        """
                        SELECT split,
                               count(*) FILTER (WHERE observed_time_months > 60) AS patients_at_risk_after_60,
                               count(*) FILTER (
                                   WHERE observed_time_months > 60 AND event_observed
                               ) AS later_events
                        FROM _ml.baseline_run_patients
                        WHERE run_id = :legacy_run_id
                        GROUP BY split
                        ORDER BY split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("LEGACY_AT_RISK_COUNTS")
            print(json.dumps([dict(row) for row in legacy_at_risk_counts], default=str, indent=2))

            scan_timing = list(
                connection.execute(
                    text(
                        """
                        WITH scans AS (
                            SELECT 'legacy_baseline' AS model, patients.split,
                                   patients.event_observed,
                                   images.months_since_baseline_image::double precision AS scan_month,
                                   images.months_to_ad_from_image::double precision AS months_from_scan_to_outcome
                            FROM _ml.baseline_run_patients AS patients
                            JOIN _core.core_image_set AS images
                              ON images.ptid = patients.ptid
                             AND images.image_id::text = patients.image_id
                            WHERE patients.run_id = :legacy_run_id
                            UNION ALL
                            SELECT 'ltsa_60m_latest' AS model, patients.split,
                                   patients.event_observed,
                                   images.months_since_baseline_image::double precision AS scan_month,
                                   images.months_to_ad_from_image::double precision AS months_from_scan_to_outcome
                            FROM _ml.survival_run_patients AS patients
                            JOIN _core.core_image_set AS images
                              ON images.ptid = patients.ptid
                             AND images.image_id::text = patients.selected_image_id
                            WHERE patients.run_id = :ltsa_run_id
                        )
                        SELECT model, split, count(*) AS patients,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY scan_month) AS median_scan_month,
                               avg(scan_month) AS mean_scan_month,
                               count(*) FILTER (WHERE scan_month > 60) AS scans_after_month_60,
                               percentile_cont(0.5) WITHIN GROUP (
                                   ORDER BY months_from_scan_to_outcome
                               ) FILTER (WHERE event_observed) AS median_event_gap_after_scan
                        FROM scans
                        GROUP BY model, split
                        ORDER BY model, split
                        """
                    ),
                    params,
                ).mappings()
            )
            print("SCAN_TIMING")
            print(json.dumps([dict(row) for row in scan_timing], default=str, indent=2))

            future_scan_leakage = connection.execute(
                text(
                    """
                    SELECT count(*) AS ltsa_cohort_patients,
                           count(*) FILTER (
                               WHERE legacy.image_id <> ltsa.selected_image_id
                           ) AS different_legacy_image,
                           count(*) FILTER (
                               WHERE legacy.image_id <> ltsa.selected_image_id
                                 AND images.months_since_baseline_image::double precision > 60
                           ) AS different_and_after_landmark,
                           percentile_cont(0.5) WITHIN GROUP (
                               ORDER BY images.months_since_baseline_image::double precision
                           ) AS median_legacy_scan_month,
                           max(images.months_since_baseline_image::double precision)
                               AS maximum_legacy_scan_month,
                           corr(
                               images.months_since_baseline_image::double precision,
                               legacy.observed_time_months
                           ) AS scan_outcome_time_correlation,
                           count(*) FILTER (WHERE legacy.split = ltsa.split)
                               AS unchanged_split,
                           count(*) FILTER (WHERE legacy.split <> ltsa.split)
                               AS changed_split
                    FROM _ml.survival_run_patients AS ltsa
                    JOIN _ml.baseline_run_patients AS legacy
                      ON legacy.run_id = :legacy_run_id
                     AND legacy.ptid = ltsa.ptid
                    JOIN _core.core_image_set AS images
                      ON images.ptid = legacy.ptid
                     AND images.image_id::text = legacy.image_id
                    WHERE ltsa.run_id = :ltsa_run_id
                    """
                ),
                params,
            ).mappings().one()
            print("FUTURE_SCAN_AND_SPLIT_CHECK")
            print(json.dumps(dict(future_scan_leakage), default=str, indent=2))

            current_split_check = connection.execute(
                text(
                    """
                    SELECT count(*) AS ltsa_cohort_patients,
                           count(*) FILTER (WHERE current_split.split::text = ltsa.split)
                               AS current_matches_ltsa,
                           count(*) FILTER (WHERE current_split.split::text = legacy.split)
                               AS current_matches_legacy
                    FROM _ml.survival_run_patients AS ltsa
                    JOIN _ml.baseline_run_patients AS legacy
                      ON legacy.run_id = :legacy_run_id
                     AND legacy.ptid = ltsa.ptid
                    JOIN _raw.train_test_split AS current_split
                      ON current_split.ptid = ltsa.ptid
                    WHERE ltsa.run_id = :ltsa_run_id
                    """
                ),
                params,
            ).mappings().one()
            print("CURRENT_SPLIT_CHECK")
            print(json.dumps(dict(current_split_check), default=str, indent=2))

        counts = connection.execute(
            text(
                """
                SELECT
                    (SELECT count(*) FROM _ml.baseline_runs) AS legacy_runs,
                    (SELECT count(*) FROM _ml.survival_runs) AS unified_runs
                """
            )
        ).mappings().one()
        print("RUN_COUNTS")
        print(json.dumps(dict(counts), default=str, indent=2))


if __name__ == "__main__":
    main()
