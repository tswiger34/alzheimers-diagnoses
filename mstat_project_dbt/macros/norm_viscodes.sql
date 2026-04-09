{% macro norm_viscodes(col_expr) -%}
    CASE
        WHEN {{ col_expr }} IN ('BL', 'SC', 'INIT', '4_BL', '4_INIT', '4_SC') THEN 'BL'
        ELSE {{ col_expr }}
    END
{%- endmacro %}
