
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
        COUNT(*) OVER (PARTITION BY ptid) AS n_ptid_imgs
    FROM {{ ref('core_image_metadata') }} 
    WHERE NOT flag_for_drop
),

image_set_cohorts AS (
    SELECT
        *,
        image_date - LAG(image_date) OVER (PARTITION BY ptid ORDER BY image_date) AS days_since_prior_image,
        image_date - MIN(image_date) OVER (PARTITION BY ptid) AS days_since_baseline_image,
        NTILE(10) OVER (ORDER BY ptid, image_date, image_id) AS processing_set_cohort,
        ROW_NUMBER() OVER (PARTITION BY ptid ORDER BY image_date, image_id) AS ptid_img_number
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
        ((image_set_cohorts.days_since_baseline_image::DECIMAL/365) * 12)::NUMERIC(5,2) AS months_since_baseline_image,
        ((image_set_cohorts.days_since_prior_image::DECIMAL/365) * 12)::NUMERIC(5,2) AS months_since_prior_image
    FROM image_set_cohorts
    LEFT JOIN {{ ref('ptid_dx_transforms') }} AS dx
        ON
            image_set_cohorts.ptid = dx.ptid
            AND (
                (
                    image_set_cohorts.image_date >= dx.valid_from
                    AND (image_set_cohorts.image_date < dx.valid_to OR dx.valid_to IS NULL)
                )
                OR (
                    image_set_cohorts.image_date < dx.valid_from
                    AND dx.visit_number = 1
                )

            )

),

derive_dx_visit_info AS (
    SELECT
        *,
        LAG(diagnosis_at_visit) OVER (PARTITION BY ptid ORDER BY image_date) AS prior_diagnosis,
        LAG(diagnosis_code_at_visit) OVER (PARTITION BY ptid ORDER BY image_date) AS prior_diagnosis_code,
        FIRST_VALUE(diagnosis_at_visit) OVER (PARTITION BY ptid ORDER BY image_date) AS baseline_diagnosis,
        FIRST_VALUE(image_date) OVER (PARTITION BY ptid ORDER BY image_date) AS baseline_image_date,
        MIN(image_date) FILTER (WHERE diagnosis_at_visit = 'AD') OVER (
            PARTITION BY ptid
            ORDER BY image_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS first_ad_diagnosis,
        LAST_VALUE(diagnosis_at_visit) OVER (
            PARTITION BY ptid
            ORDER BY image_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS final_diagnosis,
        LAST_VALUE(image_date) OVER (
            PARTITION BY ptid
            ORDER BY image_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS final_image_date
    FROM get_dx_info
),

derive_time_to_event_info AS (
    SELECT
        *,
        first_ad_diagnosis - image_date AS time_to_ad_from_image,
        first_ad_diagnosis - baseline_image_date AS time_to_ad_from_baseline,
        COALESCE(baseline_diagnosis = 'AD', FALSE) AS is_ad_at_baseline,
        CASE
            WHEN diagnosis_at_visit = prior_diagnosis THEN 'No Change' || ' - ' || diagnosis_at_visit
            ELSE prior_diagnosis || ' - ' || diagnosis_at_visit
        END AS change_from_prior_visit,
        CASE
            WHEN diagnosis_code_at_visit > prior_diagnosis_code THEN 'Worsened'
            WHEN diagnosis_code_at_visit < prior_diagnosis_code THEN 'Improved'
            ELSE 'No Change'
        END AS change_from_prior_visit_code,
        CASE
            WHEN final_diagnosis = baseline_diagnosis THEN 'Stable' || ' - ' || baseline_diagnosis
            ELSE 'Conversion' || ' - ' || baseline_diagnosis || ' to ' || final_diagnosis
        END AS change_from_baseline,
        COALESCE(first_ad_diagnosis IS NULL, FALSE) AS is_censored
    FROM derive_dx_visit_info
),

get_ptid_info AS (
    SELECT 
        derive_time_to_event_info.*,
        pt.birth_date AS pt_birth_date,
        pt.marital_status AS pt_marital_status,
        pt.education_years AS pt_education_years,
        EXTRACT(YEAR FROM age(derive_time_to_event_info.baseline_image_date, pt.birth_date)) AS age_at_baseline,
        EXTRACT(YEAR FROM age(derive_time_to_event_info.image_date, pt.birth_date)) AS age_at_image,
        ((time_to_ad_from_image::DECIMAL/365) * 12)::NUMERIC(5,2) AS months_to_ad_from_image,
        ((time_to_ad_from_baseline::DECIMAL/365) * 12)::NUMERIC(5,2) AS months_to_ad_from_baseline
    FROM derive_time_to_event_info
    LEFT JOIN {{ ref('core_ptdemog') }} AS pt
        ON derive_time_to_event_info.ptid = pt.ptid
)

SELECT *
FROM get_ptid_info
WHERE
    NOT is_ad_at_baseline
    AND diagnosis_at_visit IS NOT NULL
    AND (time_to_ad_from_image >= 0 OR is_censored)
ORDER BY ptid, image_date, image_id