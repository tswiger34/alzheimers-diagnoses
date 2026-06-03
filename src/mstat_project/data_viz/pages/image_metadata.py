import dash
import polars as pl
from dash import html

from mstat_project.data_viz.components.metric_card import MetricCard
from mstat_project.data_viz.utils import get_dataset

dash.register_page(module=__name__, name="Image Metadata")


core_df = get_dataset(tbl_name="_core.core_image_set")


def get_patient_counts(core_df: pl.DataFrame):
    if core_df.is_empty() or "ptid" not in core_df.columns:
        return 0
    return core_df.select(pl.col("ptid").n_unique()).item()


def get_img_counts(core_df: pl.DataFrame):
    if core_df.is_empty():
        return 0

    if "image_id" in core_df.columns:
        return core_df.select(pl.col("image_id").n_unique()).item()
    return core_df.height


def get_avg_imgs_per_patient(core_df: pl.DataFrame):
    n_patients = get_patient_counts(core_df=core_df)
    n_images = get_img_counts(core_df=core_df)

    if n_patients == 0:
        return 0.0
    return round(n_images / n_patients, 2)


def get_unique_visit_count(core_df: pl.DataFrame):
    if core_df.is_empty():
        return 0

    visit_col = None
    for col_name in ("visit_code_normed", "viscode", "visit_code"):
        if col_name in core_df.columns:
            visit_col = col_name
            break

    if visit_col is None:
        return 0

    return core_df.select(pl.col(visit_col).n_unique()).item()


def _build_metric_cards(core_df: pl.DataFrame) -> list[MetricCard]:
    n_images = get_img_counts(core_df=core_df)
    n_patients = get_patient_counts(core_df=core_df)
    avg_imgs_per_patient = get_avg_imgs_per_patient(core_df=core_df)
    n_visits = get_unique_visit_count(core_df=core_df)

    return [
        {
            "title": "Image Records",
            "main_value": f"{n_images:,}",
            "eyebrow": "Coverage",
            "description": "Total unique scans available in the current dataset pull.",
            "sub_metrics": [f"Rows loaded: {core_df.height:,}"],
        },
        {
            "title": "Unique Patients",
            "main_value": f"{n_patients:,}",
            "eyebrow": "Cohort",
            "description": "Number of unique patients represented in these scans.",
            "sub_metrics": [
                (f"Avg scans / patient: {avg_imgs_per_patient}" if n_patients > 0 else "Avg scans / patient: n/a")
            ],
        },
        {
            "title": "Visit Codes",
            "main_value": f"{n_visits:,}",
            "eyebrow": "Longitudinal",
            "description": "Distinct visit labels available for timeline-level analysis.",
            "sub_metrics": ["Normalized visit codes"],
        },
        {
            "title": "Completeness Snapshot",
            "main_value": "Ready",
            "eyebrow": "QC",
            "description": "Initial ingestion loaded and ready for metadata quality profiling.",
            "sub_metrics": ["Scaffold metric row active"],
        },
    ]


def _render_metric_card(card: MetricCard) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "rgba(255, 255, 255, 0.78)",
            "border": "1px solid rgba(19, 51, 61, 0.12)",
            "borderRadius": "18px",
            "padding": "18px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "boxShadow": "0 10px 24px rgba(14, 43, 52, 0.06)",
        },
        children=[
            html.Span(
                card["eyebrow"],
                style={
                    "fontSize": "0.72rem",
                    "fontWeight": 700,
                    "letterSpacing": "0.14em",
                    "textTransform": "uppercase",
                    "color": "#1b6f71",
                },
            ),
            html.Div(card["main_value"], style={"fontSize": "2rem", "fontWeight": 700, "color": "#113741"}),
            html.Div(card["title"], style={"fontSize": "1rem", "fontWeight": 700, "color": "#173b46"}),
            html.P(
                card["description"],
                style={
                    "margin": "0",
                    "fontSize": "0.92rem",
                    "lineHeight": "1.6",
                    "color": "#4f6870",
                },
            ),
            html.Div(
                children=[
                    html.Span(
                        str(metric),
                        style={
                            "display": "inline-block",
                            "padding": "5px 9px",
                            "borderRadius": "999px",
                            "backgroundColor": "rgba(27, 122, 120, 0.08)",
                            "fontSize": "0.8rem",
                            "color": "#0f5d64",
                            "fontWeight": 600,
                            "marginRight": "6px",
                            "marginTop": "4px",
                        },
                    )
                    for metric in card["sub_metrics"]
                ]
            ),
        ],
    )


metric_cards = _build_metric_cards(core_df=core_df)


layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "gap": "18px"},
    children=[
        html.H1(children="Analyze Key MRI Metadata for Data Quality Checks"),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "14px",
            },
            children=[_render_metric_card(card=card) for card in metric_cards],
        ),
    ],
)
