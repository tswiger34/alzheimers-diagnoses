
WITH casted AS (
    SELECT
        {{ norm_text_codes('"image_id"') }}::INTEGER AS image_id,
        {{ norm_text_codes('"subject_id"', to_upper=true) }}::TEXT AS ptid,
        {{ norm_text_codes('"image_visit"', to_upper=true) }}::TEXT AS visit_code,
        NULLIF(TRIM("image_date"::TEXT), '')::DATE AS image_date,
        {{ norm_text_codes('"series_type"', to_upper=true) }}::TEXT AS series_type,
        {{ norm_text_codes('"mri_protocol_phase"', to_upper=true) }}::TEXT AS study_phase,
        {{ norm_text_codes('"series_description"', to_upper=true) }}::TEXT AS series_description,
        {{ norm_text_codes('"acceleration"', to_upper=true) }}::TEXT AS acceleration,
        {{ norm_text_codes('"acquisition_type"', to_upper=true) }}::TEXT AS acquisition_type,
        {{ norm_text_codes('"acquisition_plane"', to_upper=true) }}::TEXT AS acquisition_plane,
        {{ norm_text_codes('"number_volumes"') }}::INTEGER AS number_volumes,
        {{ norm_text_codes('"slices_per_volume"') }}::INTEGER AS slices_per_volume,
        {{ norm_text_codes('"slice_thickness"') }}::NUMERIC AS slice_thickness_mm,
        {{ norm_text_codes('"scanner_manufacturer"', to_upper=true) }}::TEXT AS scanner_manufacturer,
        {{ norm_text_codes('"scanner_model"', to_upper=true) }}::TEXT AS scanner_model,
        {{ norm_text_codes('"software_version"', to_upper=true) }}::TEXT AS software_version,
        {{ norm_text_codes('"magnetic_field_strength"') }}::TEXT AS magnetic_field_strength_tesla,
        {{ norm_text_codes('"receive_coil_name"', to_upper=true) }}::TEXT AS receive_coil_name,
        {{ norm_text_codes('"study_instance_uid"', to_upper=true) }}::TEXT AS study_instance_uid,
        {{ norm_text_codes('"series_instance_uid"', to_upper=true) }}::TEXT AS series_instance_uid,
        {{ norm_text_codes('"loni_study"') }}::INTEGER AS loni_study_id,
        {{ norm_text_codes('"loni_series"') }}::INTEGER AS loni_series_id,
        {{ norm_text_codes('"loni_image"') }}::INTEGER AS loni_image_id
    FROM {{ ref('key_mri') }}
),

normed AS (
    SELECT
        *,
        {{ norm_viscodes('visit_code') }} AS visit_code_normed
    FROM casted
),

get_flags AS (
    SELECT
        *,
        CASE 
            WHEN series_description LIKE '%MP%RAGE%' THEN 1
        ELSE 0
        END AS is_mprage,
        CASE 
            WHEN series_description LIKE '%MP%RAGE%REPEAT%' THEN 1
        ELSE 0
        END AS is_mprage_repeat,
        CASE
            WHEN series_description LIKE '%REPEAT%' THEN 1
        ELSE 0
        END AS is_repeat,
        CASE
            WHEN acceleration LIKE '%UNACCELERATED%' THEN 0
            WHEN acceleration IS NULL THEN NULL
            ELSE 1
        END AS is_accelerated
    FROM normed        
)

SELECT
    image_id,
    ptid,
    visit_code_normed,
    image_date,
    series_type,
    study_phase,
    series_description,
    acceleration,
    acquisition_type,
    acquisition_plane,
    number_volumes,
    slices_per_volume,
    slice_thickness_mm,
    scanner_manufacturer,
    scanner_model,
    software_version,
    magnetic_field_strength_tesla,
    receive_coil_name,
    study_instance_uid,
    series_instance_uid,
    loni_study_id,
    loni_series_id,
    loni_image_id,
    is_mprage::BOOL,
    is_mprage_repeat::BOOL,
    is_repeat::BOOL,
    is_accelerated::BOOL
FROM get_flags