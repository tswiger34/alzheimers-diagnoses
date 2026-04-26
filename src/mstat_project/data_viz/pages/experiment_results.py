import dash
from dash import html

dash.register_page(module=__name__, name="Experiment Results")

layout = html.Div(
    children=[
        html.H1(children="This is the experiment results page"),
        html.Div(children="This is the experiment results content"),
    ]
)
