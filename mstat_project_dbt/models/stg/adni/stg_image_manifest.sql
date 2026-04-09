WITH casted AS (
    SELECT
        {{ norm_text_codes('"image_id"') }}::INTEGER AS image_id,
        {{ norm_text_codes('"subject_id"', to_upper=true) }}::TEXT AS ptid,
        {{ norm_text_codes('"study_id"') }}::INTEGER AS study_id,
        {{ norm_text_codes('"series_id"') }}::INTEGER AS series_id,
        {{ norm_text_codes('"image_visit"', to_upper=true) }}::TEXT AS visit_code,
        NULLIF(TRIM("image_date"::TEXT), '')::DATE AS image_date,
        {{ norm_text_codes('"image_description"', to_upper=true) }}::TEXT AS image_description
    FROM {{ ref('image_manifest') }}
),

normed AS (
    SELECT
        *,
        {{ norm_viscodes('visit_code') }} AS visit_code_normed
    FROM casted
)

SELECT
    image_id,
    ptid,
    study_id,
    series_id,
    visit_code_normed,
    image_date,
    image_description
FROM normed
