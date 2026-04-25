WITH casted AS (
    SELECT
        {{ norm_text_codes('"PHASE"', to_upper=true) }}::TEXT AS study_phase,
        {{ norm_text_codes('"PTID"', to_upper=true) }}::TEXT AS ptid,
        {{ norm_text_codes('"VISCODE"', to_upper=true) }}::TEXT AS visit_code_1,
        {{ norm_text_codes('"VISCODE2"', to_upper=true) }}::TEXT AS visit_code_2,
        NULLIF(TRIM("EXAMDATE"::TEXT), '')::DATE AS exam_date,
        {{ norm_text_codes('"DIAGNOSIS"') }}::INTEGER AS current_diagnosis,
        {{ norm_text_codes('"DXNORM"') }}::INTEGER::BOOL AS is_cognitively_normal,
        {{ norm_text_codes('"DXNODEP"') }}::INTEGER::BOOL AS has_mild_depression,
        {{ norm_text_codes('"DXMCI"') }}::INTEGER::BOOL AS has_mild_cognitive_impairment,
        {{ norm_text_codes('"DXMDES"', to_upper=true) }}::TEXT AS mci_subtype_codes,
        {{ norm_text_codes('"DXMPTR1"') }}::INTEGER::BOOL AS has_subjective_memory_complaint,
        {{ norm_text_codes('"DXMPTR2"') }}::INTEGER::BOOL AS has_informant_reported_memory_complaint,
        {{ norm_text_codes('"DXMPTR3"') }}::INTEGER::BOOL AS has_normal_general_cognitive_function,
        {{ norm_text_codes('"DXMPTR4"') }}::INTEGER::BOOL AS has_normal_activities_of_daily_living,
        {{ norm_text_codes('"DXMPTR5"') }}::INTEGER::BOOL AS has_objective_memory_impairment,
        {{ norm_text_codes('"DXMPTR6"') }}::INTEGER::BOOL AS is_not_demented_by_diagnostic_criteria,
        {{ norm_text_codes('"DXMDUE"') }}::INTEGER::BOOL AS is_mci_due_to_alzheimers_disease,
        {{ norm_text_codes('"DXMOTHET"', to_upper=true) }}::TEXT AS mci_other_etiology_codes,
        {{ norm_text_codes('"DXDSEV"') }}::INTEGER AS dementia_severity,
        {{ norm_text_codes('"DXDDUE"') }}::INTEGER AS dementia_suspected_cause,
        {{ norm_text_codes('"DXAD"') }}::INTEGER::BOOL AS has_alzheimers_disease,
        {{ norm_text_codes('"DXAPP"') }}::INTEGER AS alzheimers_disease_presentation,
        {{ norm_text_codes('"DXAPROB"', to_upper=true) }}::TEXT AS probable_alzheimers_symptom_codes,
        {{ norm_text_codes('"DXAPOSS"', to_upper=true) }}::TEXT AS possible_alzheimers_reason_codes,
        {{ norm_text_codes('"DXPARK"') }}::INTEGER::BOOL AS has_parkinsons_disease,
        {{ norm_text_codes('"DXDEP"') }}::INTEGER::BOOL AS has_depressive_symptoms,
        {{ norm_text_codes('"DXOTHDEM"') }}::INTEGER::BOOL AS has_other_dementia,
        {{ norm_text_codes('"DXODES"') }}::INTEGER AS other_dementia_diagnosis,
        {{ norm_text_codes('"DXCONFID"') }}::INTEGER AS diagnosis_confidence,
        NULLIF(TRIM("USERDATE"::TEXT), '')::TIMESTAMP AS record_created,
        NULLIF(TRIM("USERDATE2"::TEXT), '')::TIMESTAMP AS record_last_updated,
        {{ norm_text_codes('"DD_CRF_VERSION_LABEL"', to_upper=true) }}::TEXT AS crf_version_label,
        {{ norm_text_codes('"HAS_QC_ERROR"') }}::INTEGER::BOOL AS has_qc_error,
        NULLIF(TRIM("update_stamp"::TEXT), '')::TIMESTAMP AS update_stamp
    FROM {{ ref('dxsum') }}
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
),

mci_codes AS (
    SELECT
        *,
        ptid || '|' || visit_code_normed AS dxsum_exam_id,
        CASE
            WHEN mci_subtype_codes IS NULL THEN NULL
            WHEN mci_subtype_codes LIKE '%1%' THEN TRUE
            ELSE FALSE
        END AS mci_has_memory_features,
        CASE
            WHEN mci_subtype_codes IS NULL THEN NULL
            WHEN mci_subtype_codes LIKE '%2%' THEN TRUE
            ELSE FALSE
        END AS mci_has_non_memory_features
        FROM normed
)

SELECT
    dxsum_exam_id,
    study_phase,
    ptid,
    visit_code_normed,
    exam_date,
    current_diagnosis,
    is_cognitively_normal,
    has_mild_depression,
    has_mild_cognitive_impairment,
    mci_subtype_codes,
    mci_has_memory_features,
    mci_has_non_memory_features,
    has_subjective_memory_complaint,
    has_informant_reported_memory_complaint,
    has_normal_general_cognitive_function,
    has_normal_activities_of_daily_living,
    has_objective_memory_impairment,
    is_not_demented_by_diagnostic_criteria,
    is_mci_due_to_alzheimers_disease,
    mci_other_etiology_codes,
    dementia_severity,
    dementia_suspected_cause,
    has_alzheimers_disease,
    alzheimers_disease_presentation,
    probable_alzheimers_symptom_codes,
    possible_alzheimers_reason_codes,
    has_parkinsons_disease,
    has_depressive_symptoms,
    has_other_dementia,
    other_dementia_diagnosis,
    diagnosis_confidence,
    record_created,
    record_last_updated,
    crf_version_label,
    has_qc_error,
    update_stamp
FROM mci_codes
