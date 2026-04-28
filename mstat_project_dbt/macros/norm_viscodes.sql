{% macro norm_viscodes(col_expr) -%}
    CASE
        WHEN {{ col_expr }} IN ('BL', 'SC', 'INIT', '4_BL', '4_INIT', '4_SC', 'NV', 'SCMRI') THEN 'BL'
        WHEN {{ col_expr }} IN ('M12', '4_M12', 'Y1', 'V11') THEN 'M12'
        WHEN {{ col_expr }} IN ('M24', '4_M24', 'Y2') THEN 'M24'
        WHEN {{ col_expr }} IN ('M36', '4_M36', 'Y3') THEN 'M36'
        WHEN {{ col_expr }} IN ('M48', '4_M48', 'Y4') THEN 'M48'
        ELSE {{ col_expr }}
    END
{%- endmacro %}