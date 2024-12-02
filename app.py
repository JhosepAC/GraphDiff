from dash import Dash, html, dcc 
import dash

app = Dash(
    __name__, 
    use_pages=True, 
    suppress_callback_exceptions=True
)

app.layout = html.Div(children=[
    # Primer Div
    html.Div(className='header', children=[
        html.Img(className='logo', src='assets/images/logo.png'),
        html.Div(className='div_flex_column', children=[
            html.H1('Interfaz Gráfica', className='main_title'),
            html.H3('Salvador Mesias Damián Navarro', className='main_subtitle')
        ])
    ]),
    
    # Segundo Div
    html.Div(className='contenedor_navegacion', children=[
        dcc.Link(html.Button('Modelo 01', className='boton edo_1'), href='/'),
        dcc.Link(html.Button('Modelo 02', className='boton edo_2'), href='/2'),
        dcc.Link(html.Button('Modelo 03', className='boton edo_3'), href='/3'),
        dcc.Link(html.Button('Modelo 04', className='boton edo_4'), href='/4'),
    ]),
                      
    dash.page_container 
])

# Expone el servidor Flask interno
server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=5000)
