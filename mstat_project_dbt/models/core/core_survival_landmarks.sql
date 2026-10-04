-- MRI bins select inputs only. Outcomes come from the complete clinical history,
-- including examinations with no MRI and examinations beyond the image schedule.
WITH clinical AS (
    SELECT
        ptid,
        MIN(exam_date) FILTER (WHERE diagnosis_code_at_visit = 3) AS ad_exam_date,
        MAX(exam_date) FILTER (WHERE diagnosis_code_at_visit IN (1, 2, 3)) AS last_exam_date
    FROM {{ ref('ptid_dx_transforms') }}
    GROUP BY ptid
), conflicts AS (
    SELECT DISTINCT ptid
    FROM {{ ref('ptid_dx_transforms') }}
    WHERE diagnosis_code_at_visit IN (1, 2, 3)
    GROUP BY ptid, exam_date
    HAVING COUNT(DISTINCT diagnosis_code_at_visit) > 1
), dated AS (
    SELECT
        i.ptid, i.image_id, i.image_date, i.baseline_image_date,
        i.visit_num, i.visit_code, i.months_since_baseline_image AS binning_months,
        i.diagnosis_at_visit,
        c.ad_exam_date, c.last_exam_date,
        COALESCE(c.ad_exam_date, c.last_exam_date) AS outcome_date,
        c.ad_exam_date IS NOT NULL AS event_observed,
        x.ptid IS NOT NULL AS conflicting_diagnoses
    FROM {{ ref('core_image_set') }} AS i
    LEFT JOIN clinical AS c USING (ptid)
    LEFT JOIN conflicts AS x USING (ptid)
)
SELECT
    *,
    (image_date - baseline_image_date) * 12.0 / 365.0 AS acquisition_months,
    (outcome_date - baseline_image_date) * 12.0 / 365.0 AS outcome_months,
    CASE
        WHEN conflicting_diagnoses THEN 'conflicting_clinical_diagnoses'
        WHEN outcome_date IS NULL THEN 'missing_clinical_outcome'
        WHEN outcome_date <= baseline_image_date THEN 'not_at_risk_at_baseline'
        WHEN image_date >= outcome_date THEN 'at_or_after_outcome'
        WHEN diagnosis_at_visit = 'AD' THEN 'disease_determining_image'
        ELSE NULL
    END AS exclusion_reason
FROM dated
