
WITH src AS (
    SELECT 
        ptid_visit_date,
        image_id,
        ptid,
        image_date,
        visit_code,
        cohort,
        protocol_name,
        study_phase,
        image_type,
        series_type,
        mri_manufacturer,
        modality,
        acquisition_time,
        acquisition_number,
        acquisition_type,
        slice_thickness,
        magnetic_field_strength,
        is_gradient_corrected,
        is_mprage,
        is_mprage_repeat,
        is_repeat,
        is_accelerated,
        is_original_image,
        is_primary_image,
        is_pixel_value_normalized,
        is_3d_distortion_corrected,
        is_2d_distortion_corrected,
        is_not_distortion_corrected,
        is_magnitude_image,
        COUNT(*) OVER (PARTITION BY ptid) AS n_ptid_imgs,
        image_date - LAG(image_date) OVER (PARTITION BY ptid ORDER BY image_date) AS days_since_prior_image,
        image_date - MIN(image_date) OVER (PARTITION BY ptid) AS days_since_baseline_image
    FROM {{ ref('core_image_metadata') }} 
    WHERE NOT flag_for_drop
),

image_set_cohorts AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY ptid, image_date, image_id) AS processing_set_cohort
    FROM src
)



SELECT * FROM image_set_cohorts
ORDER BY ptid, image_date, image_id