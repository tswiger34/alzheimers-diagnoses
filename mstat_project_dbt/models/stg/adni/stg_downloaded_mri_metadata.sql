
WITH src AS (
    SELECT
        run_id::TEXT AS run_id,
        full_img_path::TEXT AS full_img_path,
        root_folder::TEXT AS root_folder,
        n_sub_dirs::INTEGER AS n_sub_dirs,
        file_name::TEXT AS file_name,
        is_dcm::BOOL AS is_dcm,
        cohort::TEXT AS cohort,
        file_size::BIGINT AS file_size
    FROM {{ source('raw', 'raw_downloaded_mri_metadata') }}
),

parse_file_path AS (
    SELECT
        *,
        SPLIT_PART(full_img_path, '/', 1)::TEXT AS parsed_root_folder,
        SPLIT_PART(full_img_path, '/', 2)::TEXT AS ptid,
        SPLIT_PART(full_img_path, '/', 3)::TEXT AS img_description,
        SPLIT_PART(full_img_path, '/', 4)::TEXT AS img_timestamp_str,
        SPLIT_PART(full_img_path, '/', 5)::TEXT AS img_id,
        SPLIT_PART(full_img_path, '/', 6)::TEXT AS file_name_parsed
    FROM src
),

parse_file_name AS (
    SELECT 
        *,
    FROM parse_file_path

)