import numpy as np
import plotly.graph_objects as go # Gráfica
import plotly.figure_factory as ff # Mallado de vectores
from scipy.integrate import odeint

def ecuacion_logistica(K: float, P0: float, r: float, t0: float, t: float, cant: float, scale: float, show_vector_field: bool):
    """
    Retorna una gráfica de la ecuación logística con su campo vectorial.

    Parámetros:
    -------
    - K: Capacidad de carga.
    - P0: Población Inicial.
    - r: Tasa de crecimiento poblacional.
    - t0: Tiempo inicial.
    - t: Tiempo final.
    - cant: Las particiones para el eje temporal y espacial. Si es None, no dibuja el campo vectorial.
    - scale: Tamaño del vector del campo vectorial.
    - show_vector_field: Booleano para mostrar el campo vectorial.
    """

    # Rango de P y t
    P_values = np.linspace(0, K + 5, cant)  # Puedes usar un valor fijo para más estabilidad
    t_values = np.linspace(0, t, cant)

    # Crear la malla de puntos (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Definir la EDO
    dP_dt = r * P * (1 - P / K)

    # Solución exacta de la Ecuación Logística
    funcion = K * P0 * np.exp(r * t_values) / (P0 * np.exp(r * t_values) + (K - P0) * np.exp(r * t0))

    # Crear la figura
    fig = go.Figure()

    # Campo vectorial: dP/dt (componente vertical)
    if show_vector_field:  # Solo crea el campo de vectores si cant no es None
        U = np.ones_like(T)  # Componente en t (horizontal)
        V = dP_dt           # Componente en P (vertical)

        # Crear el campo de vectores con Plotly
        quiver = ff.create_quiver(
            T, P, U, V,
            scale=scale,
            line=dict(color='black', width=1),
            showlegend=False
        )
        fig.add_traces(quiver.data)

    # Crear la función logística
    fig.add_trace(
        go.Scatter(
            x=t_values,
            y=funcion,
            line=dict(color='blue'),
            name='Ecuación Logística'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, t],
            y=[K, K],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Capacidad de carga'
        )
    )

    # Etiquetas para la gráfica
    fig.update_layout(
        title={
            'text': 'Campo de vectores de dP/dt = rP(1 - P/k)',
            'x': 0.5,
            'y': 0.92,
            'xanchor': 'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    # Contorno a la gráfica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=False
    )

    return fig


def lotka_volterra_model(X0, Y0, alpha, beta, delta, gamma, t):
    """
    Devuelve una gráfica del modelo de Lotka-Volterra con sus curvas de población.

    Parámetros:
    -------
    - X0: Población inicial de presas.
    - Y0: Población inicial de depredadores.
    - alpha: Tasa de crecimiento de presas.
    - beta: Tasa de depredación.
    - delta: Tasa de crecimiento de depredadores.
    - gamma: Tasa de mortalidad de depredadores.
    - t: Duración de la simulación.
    """

    # Ecuaciones de Lotka-Volterra
    def model(populations, t, alpha, beta, delta, gamma):
        X, Y = populations
        dXdt = alpha * X - beta * X * Y
        dYdt = delta * X * Y - gamma * Y
        return [dXdt, dYdt]

    # Solución del modelo
    initial_conditions = [X0, Y0]
    time_points = np.linspace(0, t, 100)  # Generate 100 time points
    solution = odeint(model, initial_conditions, time_points, args=(alpha, beta, delta, gamma))
    X, Y = solution.T

    # Crear la figura
    fig = go.Figure()

    # Agrergar la población de presas
    fig.add_trace(go.Scatter(x=time_points, y=X, mode='lines', name='Presa (X)', line=dict(color='blue')))

    # Agregar la población de depredadores
    fig.add_trace(go.Scatter(x=time_points, y=Y, mode='lines', name='Depredador (Y)', line=dict(color='red')))


    # Actualizar el diseño
    fig.update_layout(
        title={
            'text': 'Modelo Lotka-Volterra',
            'x': 0.5,
            'y': 0.92,
            'xanchor': 'center'
        },
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    # Contorno a la grafica
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=True
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        linecolor='green',
        gridcolor='gray',
        showgrid=True
    )

    return fig

def modelo_sir(S0, I0, R0, beta, gamma, t):
    """
    Devuelve una gráfica del modelo SIR con sus curvas epidémicas.

    Parámetros:
    -------
    - S0: Población susceptible inicial.
    - I0: Población infectada inicial.
    - R0: Población recuperada inicial.
    - beta: Tasa de infección.
    - gamma: Tasa de recuperación.
    - t: Duración de la epidemia.
    """
    #  Ecuaciones del modelo SIR
    def sir_equations(population, t, beta, gamma):
        S, I, R = population
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    initial_conditions = [S0, I0, R0]
    time_points = np.linspace(0, t, 100)
    solution = odeint(sir_equations, initial_conditions, time_points, args=(beta, gamma))
    S, I, R = solution.T

    # Crear la figura
    fig = go.Figure()

    # Población susceptible
    fig.add_trace(go.Scatter(x=time_points, y=S, mode='lines', name='Susceptibles (S)', line=dict(color='blue')))
    
    # Infected population
    fig.add_trace(go.Scatter(x=time_points, y=I, mode='lines', name='Infectados (I)', line=dict(color='red')))
    
    # Recovered population
    fig.add_trace(go.Scatter(x=time_points, y=R, mode='lines', name='Recuperados (R)', line=dict(color='green')))

    # Actualizar el diseño
    fig.update_layout(
        title='Modelo de Epidemia SIR',
        xaxis_title='Tiempo (t)',
        yaxis_title='Población',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    fig.update_xaxes(mirror=True, showline=True, linecolor='green', gridcolor='gray', showgrid=True)
    fig.update_yaxes(mirror=True, showline=True, linecolor='green', gridcolor='gray', showgrid=True)

    return fig

def modelo_oscilador_armonico(m, k, x0, v0, t):
    """
    Devuelve una gráfica del modelo de oscilador armónico simple.

    Parámetros:
    -------
    - m: Masa del objeto.
    - k: Constante del resorte.
    - x0: Posición inicial.
    - v0: Velocidad inicial.
    - t: Duración de la simulación.
    """
    # Ecuaciones del oscilador armónico
    omega = np.sqrt(k / m)
    time_points = np.linspace(0, t, 100)

    # Solución exacta del oscilador armónico
    x = x0 * np.cos(omega * time_points) + (v0 / omega) * np.sin(omega * time_points)

    # Crear la figura
    fig = go.Figure()

    # Posición del oscilador
    fig.add_trace(go.Scatter(x=time_points, y=x, mode='lines', name='Posición (x)', line=dict(color='blue')))

    # Actualizar el diseño
    fig.update_layout(
        title='Modelo de Oscilador Armónico Simple',
        xaxis_title='Tiempo (t)',
        yaxis_title='Posición (x)',
        width=800,
        template='plotly_white',
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    fig.update_xaxes(mirror=True, showline=True, linecolor='green', gridcolor='gray', showgrid=True)
    fig.update_yaxes(mirror=True, showline=True, linecolor='green', gridcolor='gray', showgrid=True)

    return fig
