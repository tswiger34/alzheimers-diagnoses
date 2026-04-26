import dash
from dash import html

dash.register_page(module=__name__, name="Patient Cohort")

layout = html.Div(
    children=[
        html.H1(children="This is the patient demographics page"),
        html.Div(children="This is the patient demographics content"),
    ]
)
