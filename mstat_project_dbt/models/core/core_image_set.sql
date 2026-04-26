
WITH src AS (
    SELECT
        manifest.image_id,
        manifest.ptid,
        manifest.study_id,
        manifest.series_id,
        manifest.visit_code_normed,
        manifest.image_date::DATE AS image_date,
        manifest.image_description,
        key_mri.ptid AS key_mri_ptid,
        key_mri.visit_code_normed AS key_mri_visit_code_normed,
        key_mri.image_date::DATE AS key_mri_image_date,
        key_mri.series_type AS key_mri_series_type,
        key_mri.study_phase AS key_mri_study_phase,
        key_mri.series_description AS key_mri_series_description,
        key_mri.acceleration AS key_mri_acceleration,
        key_mri.acquisition_type AS key_mri_acquisition_type,
        key_mri.acquisition_plane AS key_mri_acquisition_plane,
        key_mri.number_volumes AS key_mri_number_volumes,
        key_mri.slices_per_volume AS key_mri_slices_per_volume,
        key_mri.slice_thickness_mm AS key_mri_slice_thickness_mm,
        key_mri.scanner_manufacturer AS key_mri_scanner_manufacturer,
        key_mri.scanner_model AS key_mri_scanner_model,
        key_mri.magnetic_field_strength_tesla AS key_mri_magnetic_field_strength_tesla,
        key_mri.loni_image_id AS key_mri_loni_image_id,
        key_mri.is_mprage AS key_mri_is_mprage,
        key_mri.is_mprage_repeat AS key_mri_is_mprage_repeat,
        key_mri.is_repeat AS key_mri_is_repeat,
        key_mri.is_accelerated AS key_mri_is_accelerated
    FROM {{ ref('stg_image_manifest') }} AS manifest
    LEFT JOIN {{ ref('stg_key_mri') }} AS key_mri
        ON manifest.image_id = key_mri.image_id
),

ptid_visits_id AS (
    SELECT
        image_id,
        ptid,
        visit_code_normed,
        ROW_NUMBER() OVER (PARTITION BY ptid, visit_code_normed ORDER BY key_mri_is_mprage DESC, key_mri_is_repeat DESC, key_mri_is_accelerated DESC) AS prt_id
    FROM src
),

unq_ptid_visits AS (
    SELECT
        image_id,
        ptid,
        visit_code_normed
    FROM ptid_visits_id
    WHERE prt_id = 1
),

fltered AS (
    SELECT *
    FROM src
    WHERE
        image_id IN (
            SELECT image_id
            FROM unq_ptid_visits
        )
)

SELECT
    *,
    image_date - LAG(image_date) OVER (
        PARTITION BY
            ptid
        ORDER BY image_date
    ) AS days_since_previous_visit,
    image_date - MIN(
        CASE
            WHEN visit_code_normed = 'BL' THEN image_date
            ELSE NULL
        END
    ) OVER (
        PARTITION BY
            ptid
    ) AS days_since_baseline_visit,
    ROW_NUMBER() OVER (
        PARTITION BY
            ptid
        ORDER BY image_date
    ) AS ptid_img_number
FROM fltered