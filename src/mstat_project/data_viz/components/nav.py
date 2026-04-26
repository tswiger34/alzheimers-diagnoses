from dash import dcc, html

from mstat_project.data_viz.theme import GREY_PRIMARY, WHITE_PRIMARY

NAV_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "gap": "20px",
    "flexWrap": "wrap",
    "padding": "18px 22px",
    "marginBottom": "28px",
    "border": f"1px solid {GREY_PRIMARY}",
    "borderRadius": "24px",
    "backgroundColor": WHITE_PRIMARY,
    "boxShadow": "0 18px 40px rgba(17, 52, 63, 0.08)",
    "backdropFilter": "blur(10px)",
}

BLOCK_STYLING = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "2px",
}


def create_nav(nav_links: list[dcc.Link]) -> html.Nav:
    return html.Nav(
        style=NAV_STYLE,
        children=[
            html.Div(
                style=BLOCK_STYLING,
                children=[
                    html.Span(
                        id="nav-title-1",
                        children="Deep Learning Research Hub",
                        style={
                            "fontSize": "0.72rem",
                            "fontWeight": 700,
                            "letterSpacing": "0.18em",
                            "textTransform": "uppercase",
                            "color": "#2c6b73",
                        },
                    ),
                    html.Span(
                        id="nav-title-2",
                        children="Alzheimer's Imaging Dashboards",
                        style={
                            "fontSize": "1.15rem",
                            "fontWeight": 700,
                            "letterSpacing": "0.02em",
                        },
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "alignItems": "center",
                },
                children=nav_links,
            ),
        ],
    )
