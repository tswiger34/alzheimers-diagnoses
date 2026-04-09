{% macro norm_text_codes(col_expr, to_upper=false) -%}
    NULLIF(
        NULLIF(
            TRIM(
                {% if to_upper %}
                    UPPER({{ col_expr }}::TEXT)
                {% else %}
                    {{ col_expr }}::TEXT
                {% endif %}
            ),
            ''
        ),
        '-4'
    )
{%- endmacro %}
