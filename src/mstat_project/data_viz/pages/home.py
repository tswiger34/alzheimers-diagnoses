from typing import TypedDict

import dash
from dash import dcc, html

from mstat_project.data_viz.theme import BODY_FONT, PALETTE, TITLE_FONT

dash.register_page(module=__name__, path="/", name="Home", order=0)


PAGE_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "28px",
}

SECTION_STYLE = {
    "backgroundColor": PALETTE["paper"],
    "border": f"1px solid {PALETTE['line']}",
    "borderRadius": "30px",
    "boxShadow": "0 22px 48px rgba(14, 43, 52, 0.08)",
}

CARD_STYLE = {
    "background": "linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(241, 248, 247, 0.86) 100%)",
    "border": f"1px solid {PALETTE['line']}",
    "borderRadius": "24px",
    "padding": "22px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "14px",
    "minHeight": "100%",
    "boxShadow": "0 16px 32px rgba(16, 47, 57, 0.06)",
}


class DashboardCard(TypedDict):
    title: str
    href: str
    eyebrow: str
    description: str
    bullets: list[str]


DASHBOARD_CARDS: list[DashboardCard] = [
    {
        "title": "Patient Demographics",
        "href": "/patient-demographics",
        "eyebrow": "Cohort composition",
        "description": "Inspect participant age bands, sex balance, and diagnosis-level sampling characteristics before model training.",
        "bullets": ["Enrollment context", "Diagnosis stratification", "Readiness checks"],
    },
    {
        "title": "Image Metadata",
        "href": "/image-metadata",
        "eyebrow": "Acquisition audit",
        "description": "Review scanner metadata, modality coverage, and dataset-level completeness signals across imaging sessions.",
        "bullets": ["Modality inventory", "Protocol consistency", "Metadata integrity"],
    },
    {
        "title": "MRI Viewer",
        "href": "/mri-viewer",
        "eyebrow": "Visual inspection",
        "description": "Navigate example scans and visually spot artifacts, anatomical coverage issues, or preprocessing anomalies.",
        "bullets": ["Slice exploration", "Quality review", "Research communication"],
    },
    {
        "title": "Experiment Results",
        "href": "/experiment-results",
        "eyebrow": "Model evaluation",
        "description": "Track training outcomes, compare runs, and connect performance shifts back to data and architecture choices.",
        "bullets": ["Run comparisons", "Metric trends", "Failure analysis"],
    },
]


def build_dashboard_cards() -> list[dcc.Link]:
    cards = []
    for card in DASHBOARD_CARDS:
        cards.append(
            dcc.Link(
                href=card["href"],
                style={"textDecoration": "none", "color": "inherit", "display": "block", "minHeight": "100%"},
                children=html.Div(
                    style=CARD_STYLE,
                    children=[
                        html.Span(
                            card["eyebrow"],
                            style={
                                "fontSize": "0.74rem",
                                "fontWeight": 700,
                                "letterSpacing": "0.16em",
                                "textTransform": "uppercase",
                                "color": PALETTE["accent_dark"],
                            },
                        ),
                        html.H3(
                            card["title"],
                            style={
                                "margin": "0",
                                "fontSize": "1.45rem",
                                "fontFamily": TITLE_FONT,
                                "color": PALETTE["ink"],
                            },
                        ),
                        html.P(
                            card["description"],
                            style={
                                "margin": "0",
                                "fontSize": "0.98rem",
                                "lineHeight": "1.6",
                                "color": PALETTE["muted"],
                                "fontFamily": BODY_FONT,
                            },
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "8px",
                                "marginTop": "auto",
                            },
                            children=[
                                html.Span(
                                    bullet,
                                    style={
                                        "padding": "7px 11px",
                                        "borderRadius": "999px",
                                        "backgroundColor": "rgba(27, 122, 120, 0.08)",
                                        "color": PALETTE["accent_dark"],
                                        "fontSize": "0.84rem",
                                        "fontWeight": 600,
                                    },
                                )
                                for bullet in card["bullets"]
                            ],
                        ),
                        html.Div(
                            "Open dashboard",
                            style={
                                "paddingTop": "2px",
                                "fontSize": "0.92rem",
                                "fontWeight": 700,
                                "color": PALETTE["accent_dark"],
                            },
                        ),
                    ],
                ),
            )
        )
    return cards


layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.Section(
            style={
                **SECTION_STYLE,
                "overflow": "hidden",
                "background": "linear-gradient(135deg, rgba(216, 236, 233, 0.98) 0%, rgba(242, 235, 226, 0.86) 52%, rgba(255, 255, 255, 0.9) 100%)",
            },
            children=html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1.45fr) minmax(280px, 0.95fr)",
                    "gap": "24px",
                    "padding": "40px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "20px"},
                        children=[
                            html.Span(
                                "Interactive research environment",
                                style={
                                    "fontSize": "0.78rem",
                                    "fontWeight": 700,
                                    "letterSpacing": "0.18em",
                                    "textTransform": "uppercase",
                                    "color": PALETTE["accent_dark"],
                                    "fontFamily": BODY_FONT,
                                },
                            ),
                            html.H1(
                                children=(
                                    "Explore Alzheimer's imaging data, cohort structure, and model outcomes "
                                    "in one place."
                                ),
                                style={
                                    "margin": "0",
                                    "fontSize": "clamp(2.6rem, 5vw, 4.4rem)",
                                    "lineHeight": "1.00",
                                    "fontFamily": TITLE_FONT,
                                    "color": PALETTE["ink"],
                                    "maxWidth": "10ch",
                                },
                            ),
                            html.P(
                                children=(
                                    "This dashboard suite supports a deep learning research workflow including "
                                    "cohort review, image transformation visualization, and "
                                    "experiment interpretation."
                                ),
                                style={
                                    "margin": "0",
                                    "fontSize": "1.05rem",
                                    "lineHeight": "1.8",
                                    "color": PALETTE["muted"],
                                    "fontFamily": BODY_FONT,
                                    "maxWidth": "62ch",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "justifyContent": "space-between",
                            "gap": "18px",
                            "padding": "24px",
                            "borderRadius": "28px",
                            "background": "linear-gradient(180deg, rgba(19, 51, 61, 0.96) 0%, rgba(23, 80, 86, 0.94) 100%)",
                            "color": "white",
                            "minHeight": "100%",
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                                children=[
                                    html.Span(
                                        "Research focus",
                                        style={
                                            "fontSize": "0.74rem",
                                            "fontWeight": 700,
                                            "letterSpacing": "0.16em",
                                            "textTransform": "uppercase",
                                            "color": "rgba(255, 255, 255, 0.72)",
                                        },
                                    ),
                                    html.Div(
                                        children=(
                                            "Time-to-event analysis of Alzheimer's Disease diagnoses using MRIs"
                                        ),
                                        style={
                                            "fontSize": "1.30rem",
                                            "lineHeight": "1.45",
                                            "fontFamily": TITLE_FONT,
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "paddingTop": "12px",
                                    "borderTop": "1px solid rgba(255, 255, 255, 0.18)",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                },
                                children=[
                                    html.Span(
                                        children=(
                                            "Built to move between data quality, cohort understanding, "
                                            "and experiment evidence without losing context."
                                        ),
                                        style={
                                            "fontSize": "0.95rem",
                                            "lineHeight": "1.7",
                                            "color": "rgba(255, 255, 255, 0.82)",
                                        },
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ),
        html.Section(
            id="dashboard-overview",
            style={
                **SECTION_STYLE,
                "padding": "34px 36px 38px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "22px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "10px", "maxWidth": "72ch"},
                    children=[
                        html.Span(
                            "Dashboards",
                            style={
                                "fontSize": "0.76rem",
                                "fontWeight": 700,
                                "letterSpacing": "0.16em",
                                "textTransform": "uppercase",
                                "color": PALETTE["accent_dark"],
                                "fontFamily": BODY_FONT,
                            },
                        ),
                        html.H2(
                            "Choose an entry point based on the question you need to answer.",
                            style={
                                "margin": "0",
                                "fontSize": "2rem",
                                "fontFamily": TITLE_FONT,
                                "color": PALETTE["ink"],
                            },
                        ),
                        html.P(
                            "Each page highlights a different slice of the workflow, from raw cohort understanding to visual scan review and experiment interpretation.",
                            style={
                                "margin": "0",
                                "fontSize": "1rem",
                                "lineHeight": "1.75",
                                "color": PALETTE["muted"],
                                "fontFamily": BODY_FONT,
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(230px, 1fr))",
                        "gap": "18px",
                    },
                    children=build_dashboard_cards(),
                ),
            ],
        ),
    ],
)
