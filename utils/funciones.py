import numpy as np
from scipy.integrate import solve_ivp, odeint
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Global Constants
DEFAULT_TIME_POINTS = 200
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 500
DEFAULT_TEMPLATE = 'plotly_white'

class ModelValidator:
    """Validar parámetros para modelos matemáticos"""
    @staticmethod
    def validate_positive_params(*params):
        """Asegúrese de que todos los parámetros sean positivos"""
        if any(param <= 0 for param in params):
            raise ValueError("Todos los parámetros deben ser positivos")

    @staticmethod
    def validate_population(total_population):
        """Asegurarse de que la población total sea positiva"""
        if total_population <= 0:
            raise ValueError("La población total debe ser positiva")

def plot_model(time_points, curves, title, x_label='Time', y_label='Population'):
    """
    Crea una gráfica con las curvas especificadas.

    Parámetros:
    -----------
    time_points : np.array, puntos de tiempo
    curves : list, lista de curvas
    title : str, título de la gráfica
    x_label : str, etiqueta del eje x
    y_label : str, etiqueta del eje y
    """
    fig = go.Figure()
    
    for curve in curves:
        fig.add_trace(go.Scatter(
            x=time_points, 
            y=curve['data'], 
            mode='lines', 
            name=curve['name'], 
            line=dict(color=curve['color'], width=2)
        ))
    
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'y': 0.92, 'xanchor': 'center'},
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        template=DEFAULT_TEMPLATE,
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )
    
    fig.update_xaxes(
        mirror=True, showline=True, 
        linecolor='green', gridcolor='gray', showgrid=False
    )
    fig.update_yaxes(
        mirror=True, showline=True, 
        linecolor='green', gridcolor='gray', showgrid=False
    )
    
    return fig

def ecuacion_logistica(K: float, P0: float, r: float, t0: float, t: float, 
                       cant: int = 50, scale: float = 1.0, 
                       show_vector_field: bool = True):
    """
    Simula y visualiza un modelo de ecuación logística.

    Parámetros:
    -----------
    K : float, capacidad de carga
    P0 : float, población inicial
    r : float, tasa de crecimiento
    t0 : float, tiempo inicial
    t : float, tiempo total de simulación
    cant : int, cantidad de puntos
    scale : float, escala del campo vectorial
    show_vector_field : bool, mostrar campo vectorial
    """
    ModelValidator.validate_positive_params(K, P0, r)
    
    # Range of P and t
    P_values = np.linspace(0, K + 5, cant)
    t_values = np.linspace(0, t, cant)

    # Create point mesh (P, t)
    T, P = np.meshgrid(t_values, P_values)

    # Define ODE
    dP_dt = r * P * (1 - P / K)

    # Exact solution of Logistic Equation
    funcion = K * P0 * np.exp(r * t_values) / (P0 * np.exp(r * t_values) + (K - P0) * np.exp(r * t0))

    # Create figure
    fig = go.Figure()

    # Vector field
    if show_vector_field:
        U = np.ones_like(T)  # Component in t (horizontal)
        V = dP_dt  # Component in P (vertical)

        quiver = ff.create_quiver(
            T, P, U, V,
            scale=scale,
            line=dict(color='black', width=1),
            showlegend=False
        )
        fig.add_traces(quiver.data)

    # Add curves
    curves = [
        {'data': funcion, 'name': 'Ecuación logística', 'color': 'blue'},
        {'data': np.full_like(t_values, K), 'name': 'Capacidad de carga', 'color': 'red'}
    ]
    
    fig.add_traces([
        go.Scatter(
            x=t_values, 
            y=curve['data'], 
            mode='lines', 
            name=curve['name'], 
            line=dict(
                color=curve['color'], 
                dash='dash' if 'capacity' in curve['name'].lower() else 'solid'
            )
        ) for curve in curves
    ])

    fig.update_layout(
        title='Campo vectorial de dP/dt = rP(1 - P/k)',
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=800,
        template=DEFAULT_TEMPLATE,
        margin=dict(l=10, r=10, t=90, b=0),
        legend=dict(orientation='h', y=1.1)
    )

    return fig

def lotka_volterra(X0: float, Y0: float, 
                   alpha: float, beta: float, 
                   delta: float, gamma: float, 
                   t: float):
    """
    Simula y visualiza un modelo de Lotka-Volterra determinista.

    Parámetros:
    -----------
    X0 : float, población inicial de presas
    Y0 : float, población inicial de depredadores
    alpha : float, tasa de crecimiento de presas
    beta : float, tasa de depredación
    delta : float, tasa de crecimiento de depredadores
    gamma : float, tasa de mortalidad de depredadores
    t : float, tiempo total de simulación
    """
    ModelValidator.validate_positive_params(X0, Y0, alpha, beta, delta, gamma)

    def model(populations, t, alpha, beta, delta, gamma):
        X, Y = populations
        dXdt = alpha * X - beta * X * Y
        dYdt = delta * X * Y - gamma * Y
        return [dXdt, dYdt]

    initial_conditions = [X0, Y0]
    time_points = np.linspace(0, t, DEFAULT_TIME_POINTS)
    solution = odeint(model, initial_conditions, time_points, args=(alpha, beta, delta, gamma))
    X, Y = solution.T

    curves = [
        {'data': X, 'name': 'Presa (X)', 'color': 'blue'},
        {'data': Y, 'name': 'Depredador (Y)', 'color': 'red'}
    ]

    return plot_model(
        time_points, 
        curves, 
        'Modelo Lotka-Volterra', 
        'Tiempo (t)', 
        'Población (P)'
    )

def modelo_sir(S0: float, I0: float, R0: float, 
               beta: float, gamma: float, t: float):
    """
    Simula y visualiza un modelo SIR determinista.

    Parámetros:
    -----------
    S0 : float, población susceptible inicial
    I0 : float, población infectada inicial
    R0 : float, población recuperada inicial
    beta : float, tasa de infección
    gamma : float, tasa de recuperación
    t : float, tiempo total de simulación
    """
    total_population = S0 + I0 + R0
    ModelValidator.validate_population(total_population)
    ModelValidator.validate_positive_params(beta, gamma)

    r0_value = beta / gamma

    def sir_dynamics(t, y, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / total_population
        dIdt = beta * S * I / total_population - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    solution = solve_ivp(
        sir_dynamics, 
        [0, t], 
        [S0, I0, R0],
        args=(beta, gamma),
        dense_output=True
    )
    
    time_points = np.linspace(0, t, DEFAULT_TIME_POINTS)
    solution_interpolated = solution.sol(time_points)
    S, I, R = solution_interpolated

    curves = [
        {'data': S, 'name': 'Susceptibles (S)', 'color': 'blue'},
        {'data': I, 'name': 'Infectados (I)', 'color': 'red'},
        {'data': R, 'name': 'Recuperados (R)', 'color': 'green'}
    ]
    
    fig = plot_model(
        time_points, 
        curves, 
        f'Modelo SIR (R0: {r0_value:.2f})'
    )

    fig.add_annotation(
        x=0.02, y=0.98, 
        text=f'R0: {r0_value:.2f}', 
        showarrow=False, 
        xref='paper', 
        yref='paper'
    )

    return fig

def modelo_oscilador_armonico(m, k, x0, v0, t):
    """
    Simula y visualiza un modelo de oscilador armónico simple.

    Parámetros:
    -----------
    m : float, masa del oscilador
    k : float, constante del resorte
    x0 : float, posición inicial
    v0 : float, velocidad inicial
    t : float, tiempo total de simulación
    """
    # Harmonic oscillator equations
    omega = np.sqrt(k / m)
    time_points = np.linspace(0, t, 100)

    # Exact solution of the harmonic oscillator
    x = x0 * np.cos(omega * time_points) + (v0 / omega) * np.sin(omega * time_points)

    # Create figure
    fig = go.Figure()

    # Oscillator position
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=x, 
        mode='lines', 
        name='Position (x)', 
        line=dict(color='blue')
    ))

    # Update layout
    fig.update_layout(
        title='Modelo de oscilador armónico simple',
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

def modelo_sir_estocastico(S0, I0, R0, beta, gamma, delta, sigma_S, sigma_I, sigma_R, t):
    """
    Simula y visualiza un modelo SIR estocástico.

    Parámetros:
    -----------
    S0 : float, población susceptible inicial
    I0 : float, población infectada inicial
    R0 : float, población recuperada inicial
    beta : float, tasa de infección
    gamma : float, tasa de recuperación
    delta : float, tasa de mortalidad
    sigma_S : float, volatilidad de la población susceptible
    sigma_I : float, volatilidad de la población infectada
    sigma_R : float, volatilidad de la población recuperada
    t : float, tiempo total de simulación
    """
    # Validación de parámetros
    total_population = S0 + I0 + R0
    if total_population <= 0 or beta <= 0 or gamma <= 0 or delta <= 0:
        raise ValueError("Parámetros inválidos")

    def sir_model_stochastic(t, y, beta, gamma, delta, sigma_S, sigma_I, sigma_R):
        S, I, R = y
        dSdt = -beta * S * I / total_population + sigma_S * np.random.normal()
        dIdt = beta * S * I / total_population - (delta + gamma) * I + sigma_I * np.random.normal()
        dRdt = gamma * I - delta * R + sigma_R * np.random.normal()
        return [dSdt, dIdt, dRdt]

    # Resolver ecuaciones diferenciales estocásticas usando el método de Euler
    initial_conditions = [S0, I0, R0]
    time_points = np.linspace(0, t, 200)
    results = np.zeros((len(time_points), 3))
    results[0] = initial_conditions

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i - 1]
        S, I, R = results[i - 1]
        dS, dI, dR = sir_model_stochastic(time_points[i], [S, I, R], beta, gamma, delta, sigma_S, sigma_I, sigma_R)
        results[i] = [S + dS * dt, I + dI * dt, R + dR * dt]

    # Extraer los resultados para cada compartimento
    S, I, R = results[:, 0], results[:, 1], results[:, 2]

    # Crear figura con Plotly
    fig = go.Figure()

    # Añadir curvas con diferentes estilos
    fig.add_trace(go.Scatter(
        x=time_points, y=S, mode='lines', 
        name='Susceptibles (S)', 
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=I, mode='lines', 
        name='Infectados (I)', 
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=R, mode='lines', 
        name='Recuperados (R)', 
        line=dict(color='green', width=2)
    ))

    # Actualizar diseño con información adicional
    fig.update_layout(
        title='Modelo SIR Estocástico',
        xaxis_title='Tiempo (t)',
        yaxis_title='Población (P)',
        width=900,
        height=500,
        template='plotly_white'
    )

    fig.update_xaxes(mirror=True, showline=True, linecolor='gray', gridcolor='lightgray')
    fig.update_yaxes(mirror=True, showline=True, linecolor='gray', gridcolor='lightgray')

    return fig