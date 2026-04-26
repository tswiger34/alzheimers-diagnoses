import dash
from dash import html

dash.register_page(module=__name__, name="Image Metadata")

layout = html.Div(
    children=[
        html.H1(children="This is the image metadata page"),
        html.Div(children="This is the image metadata content"),
    ]
)
