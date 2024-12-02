import dash
from dash import dcc, html, Input, Output, callback
from utils import modelo_sir_estocastico

dash.register_page(
    __name__,
    path='/5',
    name='Edo-5'
)

layout = html.Div(className='Pages', children=[
    html.Div(className='div_parametros', children=[
        html.H2('PARÁMETROS'),
        html.Div(className='div_flex_container', children=[
            html.Div(className='div_flex_item', children=[
                html.H3('Susceptibles Iniciales'),
                dcc.Input(type='number', value=1000, id='sus_ini', step=1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Infectados Iniciales'),
                dcc.Input(type='number', value=10, id='inf_ini', step=1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Recuperados Iniciales'),
                dcc.Input(type='number', value=0, id='rec_ini', step=1, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tasa de Infección (β)'),
                dcc.Input(type='number', value=0.3, id='inf_rate', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tasa de Recuperación (γ)'),
                dcc.Input(type='number', value=0.1, id='rec_rate', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tasa de Mortalidad (δ)'),
                dcc.Input(type='number', value=0.02, id='mort_rate', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Volatilidad Susceptibles (σ_S)'),
                dcc.Input(type='number', value=0.01, id='vol_S', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Volatilidad Infectados (σ_I)'),
                dcc.Input(type='number', value=0.01, id='vol_I', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Volatilidad Recuperados (σ_R)'),
                dcc.Input(type='number', value=0.01, id='vol_R', step=0.01, min=0)
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tiempo'),
                dcc.Input(type='number', value=50, id='time', step=1, min=0)
            ]),
        ])
    ]),

    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DE EPIDEMIA SIR ESTOCÁSTICO', style={'textAlign': 'center'}),
        html.Div(className='grafica', children=[
            dcc.Loading(type='default', children=dcc.Graph(id='figura_sir_estocastico'))
        ])
    ])
])

@callback(
    Output('figura_sir_estocastico', 'figure'),
    Input('sus_ini', 'value'),
    Input('inf_ini', 'value'),
    Input('rec_ini', 'value'),
    Input('inf_rate', 'value'),
    Input('rec_rate', 'value'),
    Input('mort_rate', 'value'),
    Input('vol_S', 'value'),
    Input('vol_I', 'value'),
    Input('vol_R', 'value'),
    Input('time', 'value'),
)
def grafica_sir_estocastico(S0, I0, R0, beta, gamma, delta, sigma_S, sigma_I, sigma_R, t):
    fig = modelo_sir_estocastico(S0, I0, R0, beta, gamma, delta, sigma_S, sigma_I, sigma_R, t)
    return fig
