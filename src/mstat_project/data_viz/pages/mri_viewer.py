import dash
from dash import html

dash.register_page(module=__name__, name="MRI Viewer")

layout = html.Div(
    children=[
        html.H1(children="This is the mri viewer page"),
        html.Div(children="This is the mri viewer content"),
    ]
)
