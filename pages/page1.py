#############################
# 
# LIBRERIAS
#
#############################
import dash 
from dash import dcc, html, Input, Output, callback
from utils import *

dash.register_page(
    __name__,
    path='/',
    name='Edo-1'
)

#############################
# 
# Layout HTML
#
#############################

layout = html.Div(className='Pages', children=[

    html.Div(className='div_parametros', children=[
        
        html.H2('PARÁMETROS'),

        html.Div(className='div_flex_container', children=[

            html.Div(className='div_flex_item', children=[
                html.H3('Población Inicial'),
                dcc.Input(type='number', value=10, id='pob_ini')
            ]),

            html.Div(className='div_flex_item', children=[
                html.H3('Tiempo Inicial'),
                dcc.Input(type='number', value=0, id='time_ini')
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tiempo Final'),
                dcc.Input(type='number', value=60, id='time_fin')
            ]),

            html.Div(className='div_flex_item', children=[
                html.H3('Tasa de Crecimiento'),
                dcc.Input(max=5, type='number', value=0.15, id='r', step=0.01),
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Capacidad de Carga'),
                dcc.Input(type='number', value=150, id='K'),
            ])
        ]),

        html.Div(className='div_flex_container', children=[
            html.Div(className='div_flex_item', children=[
                html.H3('Campo Vectorial'),
                dcc.Input(type='number', value=15, id='mallado'),
            ]),
            html.Div(className='div_flex_item', children=[
                html.H3('Tamaño del Vector'),
                dcc.Input(type='number', value=1, id='size_vec'),
            ])
        ]),

        # Botón
        html.Div(className='div_button', children=[
            html.Div([
                html.Button('Campo Vectorial', id='toggle-button', n_clicks=0, className='toggle-button')
            ]),
        ]),

    ]),

    html.Div(className='div_grafica', children=[
        html.H2('GRÁFICA DE CRECIMIENTO LOGÍSTICO', style={'textAlign': 'center'}),

        html.Div(className='grafica', children=[
            dcc.Loading(type='default', children=dcc.Graph(id='figura_1'))
        ])
    ])
])


#############################
# 
# Callback
#
#############################

@callback(
    Output('figura_1', 'figure'),
    Input('pob_ini', 'value'),
    Input('time_ini', 'value'),
    Input('time_fin', 'value'),
    Input('r', 'value'),
    Input('K', 'value'),
    Input('mallado', 'value'),
    Input('size_vec', 'value'),
    Input('toggle-button', 'n_clicks')  # Input adicional para activar/desactivar la malla
)

def grafica_edo1(P0, t_i, t_f, r, k, mallado, size_vec, n_clicks):
    # Si 'mostrar_malla' es False, pasamos None para 'cant'
    show_vector_field = (n_clicks % 2 == 1)
    
    # Llamada a la función ecuacion_logistica
    fig = ecuacion_logistica(k, P0, r, t_i, t_f, mallado, size_vec, show_vector_field)
    return fig