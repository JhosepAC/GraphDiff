import dash
from dash import dcc, html, Input, Output, callback
from utils import modelo_oscilador_armonico

dash.register_page(
    __name__,
    path='/4',
    name='Edo-4'
)

layout = html.Div(className='Pages', children=[

    html.Div(className='div_parametros', children=[

        html.H2('PARÁMETROS'),

        html.Div(className='div_flex_container', children=[
            
            html.Div(className='div_flex_item', children=[
                html.H3('Masa del Objeto (m)'),
                dcc.Input(type='number', value=1, id='mass', step=0.1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Constante del Resorte (k)'),
                dcc.Input(type='number', value=1, id='spring', step=0.1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Posición Inicial (x0)'),
                dcc.Input(type='number', value=1, id='x0', step=0.1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Velocidad Inicial (v0)'),
                dcc.Input(type='number', value=0, id='v0', step=0.1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tiempo'),
                dcc.Input(type='number', value=20, id='time', step=1, min=0)
            ]),
        ])
    ]),

    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DE OSCILADOR ARMÓNICO SIMPLE', style={'textAlign': 'center'}),
        
        html.Div(className='grafica', children=[
            dcc.Loading(type='default', children=dcc.Graph(id='figura_osc'))
        ])
    ])
])

@callback(
    Output('figura_osc', 'figure'),
    Input('mass', 'value'),
    Input('spring', 'value'),
    Input('x0', 'value'),
    Input('v0', 'value'),
    Input('time', 'value'),
)
def grafica_osc(m, k, x0, v0, t):
    fig = modelo_oscilador_armonico(m, k, x0, v0, t)
    return fig
