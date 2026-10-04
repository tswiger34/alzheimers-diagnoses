SELECT *
FROM {{ ref('core_survival_landmarks') }}
WHERE exclusion_reason IS NULL
    AND (acquisition_months < 0 OR acquisition_months >= outcome_months
        OR image_date >= outcome_date OR outcome_months IS NULL
        OR diagnosis_at_visit = 'AD')
