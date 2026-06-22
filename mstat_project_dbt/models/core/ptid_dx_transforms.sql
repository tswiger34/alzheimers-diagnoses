
WITH src AS (
    SELECT
        ptid,
        visit_code_normed,
        exam_date,
        current_diagnosis AS diagnosis_code_at_visit,
        CASE
            WHEN current_diagnosis = 1 THEN 'CN'
            WHEN current_diagnosis = 2 THEN 'MCI'
            WHEN current_diagnosis = 3 THEN 'AD'
            ELSE NULL::TEXT
        END AS diagnosis_at_visit
    FROM {{ ref('stg_dxsum') }}
),

t1 AS (
    SELECT
        ptid,
        visit_code_normed,
        exam_date,
        diagnosis_code_at_visit,
        diagnosis_at_visit,
        exam_date AS valid_from,
        LEAD(exam_date) OVER (PARTITION BY ptid ORDER BY exam_date) AS valid_to,
        CASE
            WHEN diagnosis_at_visit = 'AD' THEN TRUE
            ELSE FALSE
        END AS is_ad_at_visit,
        ROW_NUMBER() OVER (PARTITION BY ptid ORDER BY exam_date) AS visit_number,
        FIRST_VALUE(diagnosis_at_visit) OVER (PARTITION BY ptid ORDER BY exam_date) AS baseline_diagnosis,
        FIRST_VALUE(exam_date) OVER (PARTITION BY ptid ORDER BY exam_date) AS baseline_exam_date,
        MIN(exam_date) FILTER (WHERE diagnosis_at_visit = 'AD') OVER (PARTITION BY ptid ORDER BY exam_date) AS first_ad_diagnosis,
        LAST_VALUE(diagnosis_at_visit) OVER (
            PARTITION BY ptid 
            ORDER BY exam_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_diagnosis,
        LAST_VALUE(exam_date) OVER (
            PARTITION BY ptid 
            ORDER BY exam_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_exam_date,
        COUNT(*) OVER (PARTITION BY ptid) AS n_visits,
        LAG(diagnosis_at_visit) OVER (PARTITION BY ptid ORDER BY exam_date) AS prior_diagnosis,
        LAG(diagnosis_code_at_visit) OVER (PARTITION BY ptid ORDER BY exam_date) AS prior_diagnosis_code,
        LAG(exam_date) OVER (PARTITION BY ptid ORDER BY exam_date) AS prior_exam_date
    FROM src
    WHERE diagnosis_at_visit IS NOT NULL    
),

t2 AS (
    SELECT
        *,
        first_ad_diagnosis - exam_date AS time_to_ad_from_visit,
        first_ad_diagnosis - baseline_exam_date AS time_to_ad_from_baseline,
        COALESCE(baseline_diagnosis = 'AD', FALSE) AS is_ad_at_baseline,
        CASE
            WHEN diagnosis_at_visit = prior_diagnosis THEN 'No Change' || ' - ' || diagnosis_at_visit
            ELSE prior_diagnosis || ' - ' || diagnosis_at_visit
        END AS change_from_prior_visit,
        CASE
            WHEN diagnosis_code_at_visit > prior_diagnosis_code THEN 'Worsened'
            WHEN diagnosis_code_at_visit < prior_diagnosis_code THEN 'Improved'
            ELSE 'No Change'
        END AS change_from_prior_visit_code,
        CASE 
            WHEN last_diagnosis = baseline_diagnosis THEN 'Stable' || ' - ' || baseline_diagnosis
            ELSE 'Conversion' || ' - ' || baseline_diagnosis || ' to ' || last_diagnosis
        END AS change_from_baseline,
        COALESCE(first_ad_diagnosis IS NULL, FALSE) AS is_censored
    FROM t1
    ORDER BY ptid, exam_date
)

SELECT 
    *,
    COALESCE(change_from_baseline ILIKE 'Conversion%', FALSE) AS is_conversion_from_baseline
FROM t2
ORDER BY ptid, exam_date