WITH casted AS (
    SELECT
        {{ norm_text_codes('"PHASE"', to_upper=true) }}::TEXT AS study_phase,
        {{ norm_text_codes('"PTID"', to_upper=true) }}::TEXT AS ptid,
        {{ norm_text_codes('"RID"') }}::INTEGER AS roster_id,
        {{ norm_text_codes('"VISCODE"', to_upper=true) }}::TEXT AS visit_code_1,
        {{ norm_text_codes('"VISCODE2"', to_upper=true) }}::TEXT AS visit_code_2,
        NULLIF(TRIM("VISDATE"::TEXT), '')::DATE AS visit_date,
        {{ norm_text_codes('"PTSOURCE"') }}::INTEGER AS participant_source_code,
        {{ norm_text_codes('"PTGENDER"') }}::INTEGER AS gender_code,
        {{ norm_text_codes('"PTDOB"') }}::TEXT AS birth_month_year,
        EXTRACT(YEAR FROM NULLIF(TRIM("PTDOBYY"::TEXT), '')::DATE)::INTEGER AS birth_year,
        {{ norm_text_codes('"PTMARRY"') }}::INTEGER AS marital_status_code,
        {{ norm_text_codes('"PTEDUCAT"') }}::INTEGER AS education_years,
        {{ norm_text_codes('"PTWORKHS"') }}::INTEGER AS work_history_code,
        {{ norm_text_codes('"PTWORK"') }}::INTEGER AS occupation_code,
        {{ norm_text_codes('"PTNOTRT"') }}::INTEGER AS not_retired_code,
        {{ norm_text_codes('"PTRTYR"') }}::INTEGER AS retirement_year,
        {{ norm_text_codes('"PTHOME"') }}::INTEGER AS home_status_code,
        {{ norm_text_codes('"PTADBEG"') }}::INTEGER AS alzheimers_symptom_onset_year,
        {{ norm_text_codes('"PTCOGBEG"') }}::INTEGER AS cognitive_symptom_onset_year,
        {{ norm_text_codes('"PTADDX"') }}::INTEGER AS alzheimers_diagnosis_year,
        {{ norm_text_codes('"PTETHCAT"') }}::INTEGER AS ethnicity_code,
        {{ norm_text_codes('"PTRACCAT"') }}::INTEGER AS race_code,
        {{ norm_text_codes('"PTENGSPK"') }}::INTEGER AS english_speaking_code,
        {{ norm_text_codes('"ID"') }}::INTEGER AS record_id,
        {{ norm_text_codes('"SITEID"') }}::INTEGER AS site_id,
        NULLIF(TRIM("USERDATE"::TEXT), '')::TIMESTAMP AS record_created,
        NULLIF(TRIM("USERDATE2"::TEXT), '')::TIMESTAMP AS record_last_updated,
        {{ norm_text_codes('"DD_CRF_VERSION_LABEL"', to_upper=true) }}::TEXT AS crf_version_label,
        {{ norm_text_codes('"LANGUAGE_CODE"', to_upper=true) }}::TEXT AS language_code,
        {{ norm_text_codes('"HAS_QC_ERROR"') }}::INTEGER::BOOL AS has_qc_error,
        NULLIF(TRIM("update_stamp"::TEXT), '')::TIMESTAMP AS update_stamp
    FROM {{ ref('ptdemog') }}
),

new_viscodes AS (
    SELECT
        *,
        COALESCE(visit_code_2, visit_code_1) AS visit_code_final
    FROM casted
),

normed AS (
    SELECT
        *,
        {{ norm_viscodes('visit_code_final') }} AS visit_code_normed
    FROM new_viscodes
)

SELECT
    study_phase,
    ptid,
    roster_id,
    visit_code_normed,
    visit_date,
    participant_source_code,
    gender_code,
    birth_month_year,
    birth_year,
    marital_status_code,
    education_years,
    work_history_code,
    occupation_code,
    not_retired_code,
    retirement_year,
    home_status_code,
    alzheimers_symptom_onset_year,
    cognitive_symptom_onset_year,
    alzheimers_diagnosis_year,
    ethnicity_code,
    race_code,
    english_speaking_code,
    record_id,
    site_id,
    record_created,
    record_last_updated,
    crf_version_label,
    language_code,
    has_qc_error,
    update_stamp
FROM normed
