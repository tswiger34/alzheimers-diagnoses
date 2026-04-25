WITH casted AS (
    SELECT
        {{ norm_text_codes('"PHASE"', to_upper=true) }}::TEXT AS study_phase,
        {{ norm_text_codes('"PTID"', to_upper=true) }}::TEXT AS ptid,
        {{ norm_text_codes('"VISCODE"', to_upper=true) }}::TEXT AS visit_code_1,
        {{ norm_text_codes('"VISCODE2"', to_upper=true) }}::TEXT AS visit_code_2,
        "EXAMDATE"::DATE AS exam_date,
        NULLIF("BCPREDX", -4)::INTEGER AS pre_visit_diagnosis,
        NULLIF("BCADAS", -4)::BOOL AS ad_worsening,
        NULLIF("BCMMSE", -4)::BOOL AS mmse_worsening,
        NULLIF("BCMMSREC", -4)::BOOL AS mmse_recall_worsening,
        NULLIF("BCNMMMS", -4)::BOOL AS non_memory_mmse_worsening,
        NULLIF("BCNEUPSY", -4)::BOOL AS neuropsych_memory_worsening,
        NULLIF("BCNONMEM", -4)::BOOL AS neuropsych_non_memory_worsening,
        NULLIF("BCFAQ", -4)::BOOL AS adl_worsening,
        NULLIF("BCCDR", -4)::BOOL AS cdr_worsening,
        NULLIF("BCDEPRES", -4)::BOOL AS depression_worsening,
        NULLIF("BCSTROKE", -4)::BOOL AS stroke,
        NULLIF("BCDELIR", -4)::BOOL AS delirium,
        NULLIF("BCEXTCIR", -4) AS extenuating_circumstance,
        NULLIF("BCCORADL", -4)::INTEGER AS corroborated_adl,
        NULLIF("BCCORCOG", -4)::INTEGER AS corroborated_cognition,
        "USERDATE"::TIMESTAMP AS record_created,
        "USERDATE2"::TIMESTAMP AS record_last_updated,
        {{ norm_text_codes('"DD_CRF_VERSION_LABEL"', to_upper=true) }}::TEXT AS crf_version_label,
        "HAS_QC_ERROR"::BOOL AS has_qc_error,
        "update_stamp"::TIMESTAMP AS update_stamp
    FROM {{ ref('blchange') }}
),

new_viscodes AS (
    SELECT
        *,
        COALESCE(visit_code_2, visit_code_1) AS visit_code_final
    FROM casted
    WHERE exam_date IS NOT NULL AND ptid IS NOT NULL
),

normed AS (
    SELECT 
        *,
        {{ norm_viscodes('visit_code_final') }} AS visit_code_normed
    FROM new_viscodes
)

SELECT
    ptid || '|' || visit_code_normed AS blchange_exam_id,
    study_phase,
    ptid,
    visit_code_normed,
    exam_date,
    pre_visit_diagnosis,
    ad_worsening,
    mmse_worsening,
    mmse_recall_worsening,
    non_memory_mmse_worsening,
    neuropsych_memory_worsening,
    neuropsych_non_memory_worsening,
    adl_worsening,
    cdr_worsening,
    depression_worsening,
    stroke,
    delirium,
    extenuating_circumstance,
    corroborated_adl,
    corroborated_cognition,
    record_created,
    record_last_updated,
    crf_version_label,
    has_qc_error,
    update_stamp
FROM normed
