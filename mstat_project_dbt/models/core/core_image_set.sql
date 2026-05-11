
WITH src AS (
    SELECT
        manifest.ptid || '-' || manifest.image_date::TEXT AS ptid_visit_date,
        manifest.image_id AS image_id,
        manifest.ptid AS ptid,
        manifest.study_id AS study_id,
        manifest.series_id AS series_id,
        manifest.visit_code_normed AS visit_code,
        manifest.image_date::DATE AS image_date,
        manifest.image_description AS image_description,
        key_mri.ptid AS key_mri_ptid,
        key_mri.visit_code_normed AS key_mri_visit_code_normed,
        key_mri.image_date::DATE AS key_mri_image_date,
        key_mri.series_type AS key_mri_series_type,
        key_mri.study_phase AS key_mri_study_phase,
        key_mri.series_description AS key_mri_image_description,
        key_mri.acceleration AS acceleration,
        key_mri.acquisition_type AS acquisition_type,
        key_mri.acquisition_plane AS acquisition_plane,
        key_mri.number_volumes AS number_volumes,
        key_mri.slices_per_volume AS slices_per_volume,
        key_mri.slice_thickness_mm AS slice_thickness_mm,
        key_mri.scanner_manufacturer AS scanner_manufacturer,
        key_mri.scanner_model AS scanner_model,
        key_mri.magnetic_field_strength_tesla AS magnetic_field_strength_tesla,
        key_mri.loni_image_id AS loni_image_id,
        key_mri.is_mprage AS is_mprage,
        key_mri.is_mprage_repeat AS is_mprage_repeat,
        key_mri.is_repeat AS is_repeat,
        key_mri.is_accelerated AS is_accelerated
    FROM {{ ref('stg_image_manifest') }} AS manifest
    INNER JOIN {{ ref('stg_key_mri') }} AS key_mri
        ON manifest.image_id = key_mri.image_id
),

get_ids AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ptid, image_date ORDER BY is_mprage DESC, is_repeat DESC, is_accelerated DESC, magnetic_field_strength_tesla) AS ptid_visit_img_number
    FROM src
),

deduped AS (
    SELECT *
    FROM get_ids
    WHERE ptid_visit_img_number = 1
),

get_img_numbers AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ptid ORDER BY image_date) AS ptid_img_number
    FROM deduped
),

derive_visit_info AS (
    SELECT
        *,
        image_date - LAG(image_date, 1) OVER (PARTITION BY ptid ORDER BY image_date) AS days_since_prior_image,
        image_date - FIRST_VALUE(image_date) OVER (PARTITION BY ptid ORDER BY image_date) AS days_since_baseline
    FROM get_img_numbers
)

SELECT * FROM derive_visit_info ORDER BY ptid, ptid_img_number