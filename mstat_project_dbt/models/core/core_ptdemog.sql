
WITH src AS (
    SELECT
        ptid,
        visit_date,
        FIRST_VALUE(birth_month_year) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS birth_month_year,
        FIRST_VALUE(birth_year) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS birth_year,
        FIRST_VALUE(marital_status_code) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS marital_status,
        FIRST_VALUE(education_years) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS education_years,
        FIRST_VALUE(gender_code) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS gender_code,
        FIRST_VALUE(visit_date) OVER (PARTITION BY ptid ORDER BY visit_date NULLS LAST) AS baseline_visit_date
    FROM {{ ref('stg_ptdemog') }}
),

deduped AS (
    SELECT
        ptid,
        visit_date,
        birth_month_year,
        birth_year,
        marital_status,
        education_years,
        gender_code,
        baseline_visit_date,
        TO_DATE(birth_month_year, 'MM/YYYY') AS birth_date
    FROM src
    WHERE visit_date = baseline_visit_date
)

SELECT
    ptid,
    visit_date,
    TO_DATE(birth_month_year, 'MM/YYYY') AS birth_date,
    marital_status,
    education_years,
    gender_code,
    baseline_visit_date
FROM deduped
ORDER BY ptid, visit_date