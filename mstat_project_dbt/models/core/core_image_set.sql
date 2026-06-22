
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
        NTILE(10) OVER (ORDER BY ptid, image_date, image_id) AS processing_set_cohort,
        ROW_NUMBER() OVER (PARTITION BY ptid ORDER BY image_date, image_id) AS ptid_img_number,
        ROUND(((days_since_prior_image / 365) * 12)::DECIMAL, 2) AS months_since_prior_image,
        ROUND(((days_since_baseline_image / 365) * 12)::DECIMAL, 2) AS months_since_baseline_image
    FROM src
),

get_dx_info AS (
    SELECT 
        image_set_cohorts.*,
        dx.diagnosis_code_at_visit,
        dx.diagnosis_at_visit,
        dx.valid_from AS dx_valid_from,
        dx.valid_to AS dx_valid_to,
        dx.is_ad_at_visit,
        dx.baseline_diagnosis,
        dx.baseline_exam_date,
        dx.first_ad_diagnosis,
        dx.last_diagnosis,
        dx.last_exam_date,
        dx.time_to_ad_from_visit,
        dx.time_to_ad_from_baseline,
        dx.is_ad_at_baseline,
        dx.change_from_prior_visit,
        dx.change_from_prior_visit_code,
        dx.change_from_baseline,
        dx.is_censored,
        dx.is_conversion_from_baseline
    FROM image_set_cohorts
    LEFT JOIN {{ ref('ptid_dx_transforms') }} AS dx
        ON image_set_cohorts.ptid = dx.ptid
        AND image_set_cohorts.image_date >= dx.valid_from
        AND (image_set_cohorts.image_date < dx.valid_to OR dx.valid_to IS NULL)
),

get_ptid_info AS (
    SELECT 
        get_dx_info.*,
        pt.birth_date AS pt_birth_date,
        pt.marital_status AS pt_marital_status,
        pt.education_years AS pt_education_years,
        EXTRACT(YEAR FROM age(get_dx_info.baseline_exam_date, pt.birth_date)) AS age_at_baseline,
        EXTRACT(YEAR FROM age(get_dx_info.image_date, pt.birth_date)) AS age_at_image
    FROM get_dx_info
    LEFT JOIN {{ ref('core_ptdemog') }} AS pt
        ON get_dx_info.ptid = pt.ptid
)

SELECT *
FROM get_ptid_info
WHERE NOT is_ad_at_baseline
ORDER BY ptid, image_date, image_id