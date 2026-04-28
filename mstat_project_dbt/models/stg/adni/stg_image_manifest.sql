
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
        ptid || '-' || visit_code AS ptid_visit_code,
        {{ norm_viscodes('visit_code') }} AS visit_code_normed
    FROM casted
),

img_info AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                ptid
            ORDER BY image_date
        ) AS ptid_img_number,
        ROW_NUMBER() OVER (
            PARTITION BY
                ptid_visit_code
            ORDER BY image_date
        ) AS ptid_visit_img_number
    FROM normed
)

SELECT
    image_id,
    ptid,
    study_id,
    series_id,
    visit_code AS visit_code_raw,
    visit_code_normed,
    image_date,
    image_description,
    ptid_img_number,
    ptid_visit_img_number
FROM img_info