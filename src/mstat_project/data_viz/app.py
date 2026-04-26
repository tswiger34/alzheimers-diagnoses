import dash
from dash import Dash, dcc, html

from mstat_project.data_viz.components.nav import create_nav
from mstat_project.data_viz.theme import BACKGROUND_COLOR, BODY_FONT, GREY_SECONDARY

app = Dash(__name__, use_pages=True)

NAV_LINK_STYLE = {
    "padding": "10px 16px",
    "borderRadius": "999px",
    "textDecoration": "none",
    "color": BACKGROUND_COLOR,
    "backgroundColor": GREY_SECONDARY,
    "border": f"1px solid {GREY_SECONDARY}",
    "fontSize": "0.92rem",
    "fontWeight": 600,
}


def build_navigation_links() -> list[dcc.Link]:
    ordered_pages = dash.page_registry.values()
    return [
        dcc.Link(
            children=page["name"],
            href=page["relative_path"],
            style=NAV_LINK_STYLE,
        )
        for page in ordered_pages
    ]


nav_links = build_navigation_links()

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(180deg, #f4f7f7 0%, #e7efee 48%, #f8faf9 100%)",
        "color": "#14323c",
        "fontFamily": BODY_FONT,
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1180px",
                "margin": "0 auto",
                "padding": "24px 24px 48px",
            },
            children=[
                create_nav(nav_links),
                html.Main(
                    style={"paddingBottom": "24px"},
                    children=dash.page_container,
                ),
            ],
        )
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
