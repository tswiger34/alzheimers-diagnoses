
WITH
    src AS (
        SELECT
            manifest.ptid || '-' || manifest.image_date::TEXT AS ptid_visit_date,
            manifest.image_id,
            manifest.ptid,
            manifest.study_id,
            manifest.series_id,
            nifti.cohort,
            nifti.modality,
            nifti.protocol_name,
            nifti.image_type,
            nifti.nonlinear_gradient_correction,
            nifti.acquisition_time,
            nifti.acquisition_number,
            nifti.slice_thickness,
            key_mri.study_phase,
            key_mri.acquisition_type,
            key_mri.loni_image_id,
            key_mri.is_mprage,
            key_mri.is_mprage_repeat,
            key_mri.is_repeat,
            key_mri.is_accelerated,
            nifti.is_original_image,
            nifti.is_primary_image,
            nifti.is_pixel_value_normalized,
            nifti.is_3d_distortion_corrected,
            nifti.is_2d_distortion_corrected,
            nifti.is_not_distortion_corrected,
            nifti.is_magnitude_image,
            COALESCE(manifest.visit_code_normed, key_mri.visit_code_normed) AS visit_code,
            COALESCE(manifest.image_date::DATE, key_mri.image_date::DATE) AS image_date,
            COALESCE(nifti.magnetic_field_strength, key_mri.magnetic_field_strength_tesla) AS magnetic_field_strength,
            COALESCE(key_mri.series_type, nifti.series_description) AS series_type
    FROM {{ ref('stg_image_manifest') }} AS manifest
    INNER JOIN {{ ref('stg_key_mri') }} AS key_mri
        ON manifest.image_id = key_mri.image_id
    INNER JOIN {{ ref('stg_nifti_metadata') }} AS nifti
        ON manifest.image_id::TEXT = nifti.image_id::TEXT
    WHERE nifti.modality = 'MR' AND key_mri.acquisition_type = '3D'
    ),


    get_ids AS (
        SELECT
            *,
            CASE
                WHEN study_phase ILIKE 'A%4%' OR study_phase ILIKE 'A%3%'
                    THEN TRUE
                WHEN is_pixel_value_normalized AND nonlinear_gradient_correction
                    THEN TRUE
            ELSE FALSE
            END AS is_preprocessed,
            COUNT(*) OVER (PARTITION BY ptid) AS n_ptid_imgs,
            COUNT(*) OVER (PARTITION BY ptid, image_date) AS n_imgs_ptid_visit_date,
            ROW_NUMBER() OVER (
                PARTITION BY ptid, image_date 
                ORDER BY 
                    nonlinear_gradient_correction DESC NULLS LAST, 
                    is_mprage DESC NULLS LAST,
                    is_magnitude_image DESC NULLS LAST,
                    is_repeat DESC NULLS LAST, 
                    is_accelerated DESC NULLS LAST, 
                    magnetic_field_strength DESC NULLS LAST
                ) AS ptid_visit_img_number
        FROM src
    )

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
    modality,
    acquisition_time,
    acquisition_number,
    acquisition_type,
    slice_thickness,
    magnetic_field_strength,
    nonlinear_gradient_correction,
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
    is_preprocessed,
    ptid_visit_img_number,
    n_imgs_ptid_visit_date,
    n_ptid_imgs,
    COALESCE(ptid_visit_img_number <> 1, FALSE) AS flag_for_drop
FROM get_ids
ORDER BY ptid, image_date, ptid_visit_img_number