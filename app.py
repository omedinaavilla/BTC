from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from consultaspostgre import obtener_datos
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

# Obtener datos de PostgreSQL
df = obtener_datos()

# Asegurar que la columna de fecha sea de tipo datetime
df['fecha'] = pd.to_datetime(df['fecha'])

external_stylesheets = [
    {
        'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
        'rel': 'stylesheet',
    }
]
# Crear aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.title = "Bitcoin Dashboard"

# Definir tema de colores
COLORS = {
    'background': '#1a1a2e',
    'text': '#ffffff',
    'primary': '#fcbf49',
    'secondary': '#f77f00',
    'accent': '#d62828',
    'sidebar': '#16213e',
    'card': '#0f3460',
    'bitcoin': '#f7931a',
    'confidence_area': 'rgba(0, 0, 255, 0.2)',
    'danger': '#d62828',
    'success': '#28a745',
    'highlight': '#9d4edd',    # AGREGADO: Color para volatilidad
    'tertiary': '#06ffa5',     # AGREGADO: Color para RSI
}


# Crear funciones para los gráficos de EDA
def create_line_chart():
    """Crear gráfico de línea para los precios de Bitcoin"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fecha'], y=df['cierre'], mode='lines', name='Precio de cierre',
                             line=dict(color=COLORS['bitcoin'], width=2)))
    
    fig.update_layout(
        title='Evolución del Precio de Bitcoin',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_volume_chart():
    """Crear gráfico de volumen de transacciones"""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['fecha'], y=df['volumen'], name='Volumen',
                         marker_color=COLORS['secondary']))
    
    fig.update_layout(
        title='Volumen de Transacciones de Bitcoin',
        xaxis_title='Fecha',
        yaxis_title='Volumen',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_candlestick_chart():
    """Crear gráfico de velas (candlestick)"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['fecha'],
        open=df['apertura'],
        high=df['maximo'],
        low=df['minimo'],
        close=df['cierre'],
        increasing_line_color=COLORS['accent'],
        decreasing_line_color=COLORS['primary']
    )])
    
    fig.update_layout(
        title='Gráfico de Velas de Bitcoin',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_returns_chart():
    """Crear gráfico de rendimientos diarios"""
    # Calcular rendimientos diarios
    returns_df = df.copy()
    returns_df['rendimiento'] = returns_df['cierre'].pct_change() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_df['fecha'], y=returns_df['rendimiento'], mode='lines',
                             name='Rendimiento diario',
                             line=dict(color=COLORS['secondary'], width=1)))
    
    fig.update_layout(
        title='Rendimientos Diarios de Bitcoin (%)',
        xaxis_title='Fecha',
        yaxis_title='Rendimiento (%)',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_diff_series_plot():
    """Visualiza la serie original y su primera diferencia"""
    serie = df.set_index('fecha')['cierre'].asfreq('D').ffill()
    serie_diff = serie.diff().dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=serie.index, y=serie,
        mode='lines',
        name='Serie Original',
        line=dict(color=COLORS['bitcoin'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=serie_diff.index, y=serie_diff,
        mode='lines',
        name='Primera Diferencia',
        line=dict(color=COLORS['secondary'], dash='dot')
    ))

    fig.update_layout(
        title='Serie Original vs. Primera Diferencia',
        xaxis_title='Fecha',
        yaxis_title='Valor',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right', yanchor='bottom'),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

#FUNCIONES NUEVAS
def create_correlation_heatmap():
    """Crear matriz de correlación entre variables numéricas"""
    # Verificar qué columnas existen realmente en el DataFrame
    numeric_cols = ['apertura', 'maximo', 'minimo', 'cierre', 'volumen']
    # Filtrar solo las columnas que existen
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    corr = df[available_cols].corr()
    
    fig = px.imshow(
        corr, 
        text_auto=True, 
        color_continuous_scale='RdBu_r', 
        title="Matriz de Correlación"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_volatility_chart():
    """Crear gráfico de la volatilidad móvil (desviación estándar de rendimientos)"""
    df_vol = df.copy()
    df_vol['returns'] = df_vol['cierre'].pct_change()
    df_vol['volatility_30d'] = df_vol['returns'].rolling(30).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_vol['fecha'], 
        y=df_vol['volatility_30d'],
        mode='lines', 
        name='Volatilidad 30 días',
        line=dict(color=COLORS['highlight'], width=2)  # Ahora usa el color correcto
    ))
    
    fig.update_layout(
        title='Volatilidad Móvil (30 días)',
        xaxis_title='Fecha',
        yaxis_title='Desviación estándar de rendimientos',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_box_plot():
    """Crear box plots para las variables numéricas de Bitcoin"""
    fig = go.Figure()

    # Verificar qué columnas existen en tu DataFrame
    possible_variables = ['apertura', 'maximo', 'minimo', 'cierre', 'volumen', 'cierre_ajustado']
    variables = [var for var in possible_variables if var in df.columns]
    
    for var in variables:
        fig.add_trace(go.Box(
            y=df[var],
            name=var.replace('_', ' ').title(),  # Mejora el formato del nombre
            marker_color=COLORS['primary'],
            boxmean='sd'  # incluye la media y desviación estándar
        ))

    fig.update_layout(
        title='Distribución de Variables Numéricas de Bitcoin',
        yaxis_title='Valor',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def create_rsi_chart(window=14):
    """Crear gráfico del RSI (Relative Strength Index)"""
    df_rsi = df.copy()
    delta = df_rsi['cierre'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Evitar división por cero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df_rsi['RSI'] = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_rsi['fecha'], 
        y=df_rsi['RSI'],
        mode='lines', 
        name=f'RSI {window} días',
        line=dict(color=COLORS['tertiary'], width=2)  # Ahora usa el color correcto
    ))

    # Líneas de referencia para sobrecompra y sobreventa
    fig.add_hline(
        y=70, 
        line_dash="dash", 
        line_color=COLORS['danger'], 
        annotation_text="Sobrecompra (70)", 
        annotation_position="top left"
    )
    fig.add_hline(
        y=30, 
        line_dash="dash", 
        line_color=COLORS['success'], 
        annotation_text="Sobreventa (30)", 
        annotation_position="bottom left"
    )

    fig.update_layout(
        title=f'Índice RSI ({window} días)',
        xaxis_title='Fecha',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# FUNCIÓN ADICIONAL RECOMENDADA: Media móvil
def create_moving_average_chart(short_window=20, long_window=50):
    """Crear gráfico con medias móviles"""
    df_ma = df.copy()
    df_ma[f'MA_{short_window}'] = df_ma['cierre'].rolling(window=short_window).mean()
    df_ma[f'MA_{long_window}'] = df_ma['cierre'].rolling(window=long_window).mean()

    fig = go.Figure()
    
    # Precio de cierre
    fig.add_trace(go.Scatter(
        x=df_ma['fecha'], 
        y=df_ma['cierre'],
        mode='lines',
        name='Precio de Cierre',
        line=dict(color=COLORS['bitcoin'], width=2)
    ))
    
    # Media móvil corta
    fig.add_trace(go.Scatter(
        x=df_ma['fecha'], 
        y=df_ma[f'MA_{short_window}'],
        mode='lines',
        name=f'MA {short_window} días',
        line=dict(color=COLORS['secondary'], width=1)
    ))
    
    # Media móvil larga
    fig.add_trace(go.Scatter(
        x=df_ma['fecha'], 
        y=df_ma[f'MA_{long_window}'],
        mode='lines',
        name=f'MA {long_window} días',
        line=dict(color=COLORS['primary'], width=1)
    ))

    fig.update_layout(
        title=f'Precio de Bitcoin con Medias Móviles ({short_window} y {long_window} días)',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_heatmap_chart():
    """Crear mapa de calor de variaciones diarias por semana"""
    heat_df = df.copy()
    heat_df['fecha'] = pd.to_datetime(heat_df['fecha'])
    heat_df['pct_change'] = heat_df['cierre'].pct_change() * 100
    heat_df['semana'] = heat_df['fecha'].dt.isocalendar().week
    heat_df['anio'] = heat_df['fecha'].dt.year
    heat_df['dia_semana'] = heat_df['fecha'].dt.dayofweek  # 0 = lunes, 6 = domingo

    # Filtramos un año específico, por ejemplo 2023 (ajústalo según tus datos)
    heat_df = heat_df[heat_df['anio'] == 2023]

    fig = px.density_heatmap(
        heat_df,
        x='dia_semana',
        y='semana',
        z='pct_change',
        color_continuous_scale='RdYlGn',
        labels={'pct_change': '% cambio'},
        nbinsx=7,
        nbinsy=52,
        title='Mapa de Calor de Variación Diaria (% Cambio)'
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(7)),
        ticktext=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    )

    fig.update_layout(
        xaxis_title='Día de la semana',
        yaxis_title='Semana del año',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


#____________________________________________________-


def create_acf_pacf_plot():
    """Gráficos ACF y PACF antes y después de la primera diferencia con mejoras visuales"""
    serie = df['cierre']
    serie_diff = serie.diff().dropna()

    # Calcular ACF y PACF
    acf_orig = acf(serie, nlags=40)
    pacf_orig = pacf(serie, nlags=40, method='ywm')
    acf_diff = acf(serie_diff, nlags=40)
    pacf_diff = pacf(serie_diff, nlags=40, method='ywm')

    # Intervalos de confianza (usar largo de la serie correspondiente)
    conf_int_orig = 1.96 / np.sqrt(len(serie.dropna()))
    conf_int_diff = 1.96 / np.sqrt(len(serie_diff))

    # Crear figura con 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "<b>ACF - Serie Original</b>",
            "<b>PACF - Serie Original</b>",
            "<b>ACF - Primera Diferencia</b>",
            "<b>PACF - Primera Diferencia</b>"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )


    # Función auxiliar para agregar barras al gráfico
    def add_bars_and_conf(values, row, col, color, conf_int):
        x = list(range(len(values)))
        
        # Agregar área sombreada para los intervalos de confianza
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[conf_int] * len(x) + [-conf_int] * len(x),
            fill='toself',
            fillcolor=COLORS['confidence_area'],
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Agregar barras
        fig.add_trace(go.Bar(
            x=x,
            y=values,
            marker_color=color,
            marker_line_width=1,
            marker_line_color='rgba(255, 255, 255, 0.3)',
            showlegend=False,
            hovertemplate='Lag: %{x}<br>Valor: %{y:.4f}<extra></extra>'
        ), row=row, col=col)
        
        # Agregar líneas de intervalo de confianza
        fig.add_trace(go.Scatter(
            x=[0, max(x)],
            y=[conf_int, conf_int],
            mode='lines',
            line=dict(dash='dash', color=COLORS['accent'], width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=[0, max(x)],
            y=[-conf_int, -conf_int],
            mode='lines',
            line=dict(dash='dash', color=COLORS['accent'], width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Agregar línea en y=0
        fig.add_trace(go.Scatter(
            x=[0, max(x)],
            y=[0, 0],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

    # Añadir trazos para cada subplot
    add_bars_and_conf(acf_orig, 1, 1, COLORS['primary'], conf_int_orig)
    add_bars_and_conf(pacf_orig, 1, 2, COLORS['secondary'], conf_int_orig)
    add_bars_and_conf(acf_diff, 2, 1, COLORS['primary'], conf_int_diff)
    add_bars_and_conf(pacf_diff, 2, 2, COLORS['secondary'], conf_int_diff)

    # Layout final
    fig.update_layout(
        height=800,
        title={
            'text': '<b>Autocorrelación: Serie Original vs. Primera Diferencia</b>',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Mejorar ejes
    fig.update_xaxes(
        title_text='Lag',
        gridcolor='rgba(255, 255, 255, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.5)',
        tickmode='linear', 
        tick0=0,
        dtick=5
    )
    
    fig.update_yaxes(
        title_text='Correlación',
        gridcolor='rgba(255, 255, 255, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.5)',
        range=[-1.1, 1.1]  # Fijar rango para mejor comparación
    )

    return fig

def prepare_prophet_data(df):
    """Prepara los datos para Prophet con regresores adicionales"""
    # Renombrar columnas para Prophet
    prophet_df = df[["fecha", "cierre", "volumen", "maximo", "minimo"]].copy()
    prophet_df.columns = ["ds", "y", "volume", "high", "low"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
    prophet_df.dropna(inplace=True)
    
    # Filtrar datos desde 2022 para mejor performance
    prophet_df = prophet_df[prophet_df["ds"] >= "2023-01-01"]
    
    # Calcular regresores adicionales
    prophet_df['volatility'] = prophet_df['y'].rolling(window=7).std()
    prophet_df['hl_range'] = (prophet_df['high'] - prophet_df['low']) / prophet_df['y']
    prophet_df['log_volume'] = np.log(prophet_df['volume'] + 1)
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    prophet_df['rsi'] = calculate_rsi(prophet_df['y'])
    prophet_df['ma_ratio'] = prophet_df['y'] / prophet_df['y'].rolling(window=20).mean()
    prophet_df.dropna(inplace=True)
    
    # Normalizar regresores
    regresores = ['volatility', 'hl_range', 'log_volume', 'rsi', 'ma_ratio']
    scalers = {}
    regresores_scaled = []
    
    for col in regresores:
        scaler = StandardScaler()
        scaled_col = f"{col}_scaled"
        prophet_df[scaled_col] = scaler.fit_transform(prophet_df[[col]])
        scalers[col] = scaler
        regresores_scaled.append(scaled_col)
    
    return prophet_df, regresores_scaled, scalers

def create_prophet_forecast_chart(df):
    """Crea pronóstico con Prophet y retorna figura y métricas"""
    
    # Preparar datos
    prophet_df, regresores_scaled, scalers = prepare_prophet_data(df)
    
    # Configurar modelo Prophet
    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.6,
        interval_width=0.95,
        seasonality_prior_scale=20.0,
        holidays_prior_scale=20.0
    )
    
    # Agregar regresores
    for col in regresores_scaled:
        modelo.add_regressor(col)
    
    # Preparar datos para el modelo
    columnas_modelo = ['ds', 'y'] + regresores_scaled
    df_model = prophet_df[columnas_modelo].copy()
    
    # División train/test para evaluación
    test_size = 60  # Últimos 60 días para test
    if len(df_model) >= 120:
        train = df_model.iloc[:-test_size]
        test = df_model.iloc[-test_size:]
        
        # Entrenar modelo
        modelo.fit(train)
        
        # Crear futuro para test
        futuro_test = test[['ds'] + regresores_scaled].copy()
        pronostico_test = modelo.predict(futuro_test)
        
        # Calcular métricas
        y_true = test["y"].values
        y_pred = pronostico_test["yhat"].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE
        valid_mask = y_true != 0
        mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
        
        # Coverage
        y_lower = pronostico_test["yhat_lower"].values
        y_upper = pronostico_test["yhat_upper"].values
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100
        
    else:
        # Si no hay suficientes datos, usar todos para entrenar
        train = df_model
        mae = rmse = r2 = mape = coverage = 0
    
    # Entrenar modelo final con todos los datos
    modelo_final = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.6,
        interval_width=0.95,
        seasonality_prior_scale=20.0,
        holidays_prior_scale=20.0
    )
    
    for col in regresores_scaled:
        modelo_final.add_regressor(col)
    
    modelo_final.fit(df_model)
    
    # Crear pronóstico futuro (30 días)
    futuro = modelo_final.make_future_dataframe(periods=30)
    
    # Completar regresores para el futuro
    futuro = futuro.merge(
        df_model[['ds'] + regresores_scaled], 
        on='ds', 
        how='left'
    )
    
    # Rellenar valores faltantes con promedios recientes
    for col in regresores_scaled:
        last_val = df_model[col].tail(7).mean()
        futuro[col] = futuro[col].fillna(last_val)
    
    # Generar pronóstico
    pronostico = modelo_final.predict(futuro)
    
    # Corrección de sesgo
    bias_correction = df_model['y'].tail(30).mean() * 0.01
    pronostico['yhat'] += bias_correction
    pronostico['yhat_lower'] += bias_correction
    pronostico['yhat_upper'] += bias_correction
    
    # Crear gráfico
    fig = go.Figure()
    
    # Datos históricos
    historical_data = pronostico[pronostico['ds'] <= df_model['ds'].max()]
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=df_model['y'],
        mode='lines',
        name='Datos Reales',
        line=dict(color=COLORS['bitcoin'], width=2)
    ))
    
    # Pronóstico histórico
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['yhat'],
        mode='lines',
        name='Ajuste del Modelo',
        line=dict(color=COLORS['primary'], width=1, dash='dot'),
        opacity=0.7
    ))
    
    # Pronóstico futuro
    future_data = pronostico[pronostico['ds'] > df_model['ds'].max()]
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat'],
        mode='lines',
        name='Pronóstico Futuro',
        line=dict(color=COLORS['danger'], width=3)
    ))
    
    # Intervalos de confianza
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(214, 39, 40, 0.2)',
        name='Intervalo de Confianza (95%)'
    ))
    
    # Si hay datos de test, mostrarlos
    if len(df_model) >= 120:
        fig.add_trace(go.Scatter(
            x=test['ds'],
            y=test['y'],
            mode='lines',
            name='Test Real',
            line=dict(color=COLORS['success'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=pronostico_test['ds'],
            y=pronostico_test['yhat'],
            mode='lines',
            name='Predicción Test',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'Prophet - Pronóstico Bitcoin - RMSE: ${rmse:.2f} | MAPE: {mape:.1f}%',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        height=600
    )
    
    return fig, rmse, mae, r2, mape, coverage

def create_prophet_components_chart(df):
    """Crea gráfico de componentes del modelo Prophet"""
    
    prophet_df, regresores_scaled, scalers = prepare_prophet_data(df)
    
    # Configurar y entrenar modelo
    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.6,
        interval_width=0.95
    )
    
    for col in regresores_scaled:
        modelo.add_regressor(col)
    
    columnas_modelo = ['ds', 'y'] + regresores_scaled
    df_model = prophet_df[columnas_modelo].copy()
    
    modelo.fit(df_model)
    
    # Generar pronóstico
    futuro = modelo.make_future_dataframe(periods=30)
    futuro = futuro.merge(df_model[['ds'] + regresores_scaled], on='ds', how='left')
    
    for col in regresores_scaled:
        last_val = df_model[col].tail(7).mean()
        futuro[col] = futuro[col].fillna(last_val)
    
    pronostico = modelo.predict(futuro)
    
    # Crear subplots para componentes
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Tendencia', 'Estacionalidad Semanal'],
        vertical_spacing=0.08
    )
    
    # Tendencia
    fig.add_trace(go.Scatter(
        x=pronostico['ds'],
        y=pronostico['trend'],
        mode='lines',
        name='Tendencia',
        line=dict(color=COLORS['primary'])
    ), row=1, col=1)
    
    # Estacionalidad semanal
    fig.add_trace(go.Scatter(
        x=pronostico['ds'],
        y=pronostico['weekly'],
        mode='lines',
        name='Semanal',
        line=dict(color=COLORS['secondary'])
    ), row=2, col=1)
    
     
    # Estacionalidad anual (¡CAMBIO AQUÍ!)
  
    
    fig.update_layout(
        title='Componentes del Modelo Prophet',
        template='plotly_dark',
        height=800,
        showlegend=False
    )
    
    return fig


def create_residuals_analysis_chart(df):
    """Crea análisis de residuos del modelo Prophet incluyendo prueba de Ljung-Box"""
    
    prophet_df, regresores_scaled, scalers = prepare_prophet_data(df)

    columnas_modelo = ['ds', 'y'] + regresores_scaled
    df_model = prophet_df[columnas_modelo].copy()
    
    test_size = 60
    if len(df_model) >= 120:
        train = df_model.iloc[:-test_size]
        test = df_model.iloc[-test_size:]
        
        modelo = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.6,
            interval_width=0.95,
            seasonality_mode='multiplicative'
        )
        for col in regresores_scaled:
            modelo.add_regressor(col)
        modelo.fit(train)
        
        futuro_test = test[['ds'] + regresores_scaled].copy()
        pronostico_test = modelo.predict(futuro_test)
        residuos = test["y"].values - pronostico_test["yhat"].values

        # === PRUEBA DE LJUNG-BOX ===
        ljungbox_result = acorr_ljungbox(residuos, lags=[10], return_df=True)
        lb_stat = ljungbox_result['lb_stat'].values[0]
        lb_pvalue = ljungbox_result['lb_pvalue'].values[0]
        texto_ljungbox = f"Ljung-Box (lag=10): Q={lb_stat:.2f}, p-valor={lb_pvalue:.4f}"

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Predicción vs Real', 'Residuos en el Tiempo', 
                            'Distribución de Residuos', 'Q-Q Plot'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(go.Scatter(
            x=test['y'], y=pronostico_test['yhat'],
            mode='markers', name='Predicciones',
            marker=dict(color=COLORS['primary'])
        ), row=1, col=1)
        
        min_val = min(test['y'].min(), pronostico_test['yhat'].min())
        max_val = max(test['y'].max(), pronostico_test['yhat'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Línea Perfecta',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=test['ds'], y=residuos,
            mode='markers+lines', name='Residuos',
            marker=dict(color=COLORS['danger'])
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)

        fig.add_trace(go.Histogram(
            x=residuos, nbinsx=20, name='Distribución',
            marker=dict(color=COLORS['secondary'])
        ), row=2, col=1)

        sorted_residuals = np.sort(residuos)
        n = len(sorted_residuals)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles, y=sorted_residuals,
            mode='markers', name='Q-Q Plot',
            marker=dict(color=COLORS['success'])
        ), row=2, col=2)

        fig.add_annotation(
            text=texto_ljungbox,
            xref='paper', yref='paper',
            x=0.5, y=-0.15, showarrow=False,
            font=dict(size=14, color="white"),
            align='center'
        )

        fig.update_layout(
            title='Análisis de Residuos - Modelo Prophet',
            template='plotly_dark',
            height=750,
            showlegend=False
        )
        
        return fig

    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Datos insuficientes para análisis de residuos<br>Se requieren al menos 120 días de datos",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="white"),
            showarrow=False
        )
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title='Análisis de Residuos'
        )
        return fig

def create_stats_card():
    """Crear tarjeta con estadísticas descriptivas"""
    latest_price = df['cierre'].iloc[-1]
    avg_price = df['cierre'].mean()
    max_price = df['cierre'].max()
    min_price = df['cierre'].min()
    volatility = df['cierre'].pct_change().std() * 100  # Volatilidad como desviación estándar de rendimientos
    
    return html.Div([
        html.H4("Estadísticas del Bitcoin", className="stats-title"),
        html.Div([
            html.Div([
                html.P("Último precio:", className="stat-label"),
                html.H3(f"${latest_price:,.2f}", className="stat-value"),
            ], className="stat-item"),
            html.Div([
                html.P("Precio promedio:", className="stat-label"),
                html.H3(f"${avg_price:,.2f}", className="stat-value"),
            ], className="stat-item"),
            html.Div([
                html.P("Precio máximo:", className="stat-label"),
                html.H3(f"${max_price:,.2f}", className="stat-value"),
            ], className="stat-item"),
            html.Div([
                html.P("Precio mínimo:", className="stat-label"),
                html.H3(f"${min_price:,.2f}", className="stat-value"),
            ], className="stat-item"),
            html.Div([
                html.P("Volatilidad diaria:", className="stat-label"),
                html.H3(f"{volatility:.2f}%", className="stat-value"),
            ], className="stat-item"),
        ], className="stats-container")
    ], className="stats-card")

def create_histogram():
    """Crear histograma de precios de cierre"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['cierre'],
        nbinsx=30,
        marker_color=COLORS['primary']
    ))
    
    fig.update_layout(
        title='Distribución de Precios de Cierre de Bitcoin',
        xaxis_title='Precio (USD)',
        yaxis_title='Frecuencia',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_moving_averages():
    """Crear gráfico con medias móviles"""
    # Calcular medias móviles
    ma_df = df.copy()
    ma_df['MA50'] = ma_df['cierre'].rolling(window=50).mean()
    ma_df['MA200'] = ma_df['cierre'].rolling(window=200).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ma_df['fecha'], y=ma_df['cierre'], mode='lines', name='Precio de cierre',
                             line=dict(color=COLORS['bitcoin'], width=2)))
    fig.add_trace(go.Scatter(x=ma_df['fecha'], y=ma_df['MA50'], mode='lines', name='Media Móvil 50 días',
                             line=dict(color=COLORS['primary'], width=1.5)))
    fig.add_trace(go.Scatter(x=ma_df['fecha'], y=ma_df['MA200'], mode='lines', name='Media Móvil 200 días',
                             line=dict(color=COLORS['accent'], width=1.5)))
    
    fig.update_layout(
        title='Análisis Técnico: Medias Móviles',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# Estilos CSS para la aplicación
external_stylesheets = [
    {
        'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
        'rel': 'stylesheet',
    }
]

# CSS personalizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Estilos personalizados */
            body {
                margin: 0;
                padding: 0;
                font-family: "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
                background-color: #1a1a2e;
                color: #ffffff;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .stats-card {
                background-color: #0f3460;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            .stats-title {
                text-align: center;
                margin-top: 0;
                margin-bottom: 20px;
                color: #f7931a;
                font-weight: 600;
            }
            
            .stats-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
            }
            
            .stat-item {
                flex: 1 1 18%;
                min-width: 120px;
                text-align: center;
                margin: 10px;
                padding: 15px;
                background-color: #16213e;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            
            .stat-item:hover {
                transform: translateY(-5px);
            }
            
            .stat-label {
                margin: 0;
                color: #fcbf49;
                font-size: 14px;
            }
            
            .stat-value {
                margin: 5px 0 0;
                font-size: 20px;
                font-weight: bold;
            }
            
            /* Mejoras para los tabs y gráficos */
            .dash-tab {
                transition: background-color 0.3s ease;
            }
            
            .dash-tab:hover {
                background-color: rgba(252, 191, 73, 0.2);
            }
            
            .dash-graph {
                transition: all 0.3s ease;
            }
            
            .dash-graph:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            }
            
            /* Mejoras visuales para móviles */
            @media (max-width: 768px) {
                .stats-container {
                    flex-direction: column;
                }
                
                .stat-item {
                    margin: 5px 0;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Estilo personalizado
app_style = {
    'backgroundColor': COLORS['background'],
    'color': COLORS['text'],
    'fontFamily': '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
    'minHeight': '100vh',
}

header_style = {
    'backgroundColor': COLORS['card'],
    'padding': '20px',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'borderRadius': '10px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
}

tab_style = {
    'backgroundColor': COLORS['sidebar'],
    'color': COLORS['text'],
    'padding': '12px 16px',
    'borderRadius': '5px 5px 0 0',
    'borderBottom': f'2px solid {COLORS["background"]}',
    'marginRight': '2px',
}

tab_selected_style = {
    'backgroundColor': COLORS['primary'],
    'color': COLORS['background'],
    'padding': '12px 16px',
    'borderRadius': '5px 5px 0 0',
    'fontWeight': 'bold',
    'borderBottom': 'none',
}

content_style = {
    'padding': '20px',
}

card_style = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '10px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
}


# Antes: fig, rmse, aic, bic, mae, r2 = create_arima_forecast_chart(df)
# Ahora:

fig, rmse, mae, r2, mape, coverage = create_prophet_forecast_chart(df)


# Diseño de la aplicación
app.layout = html.Div(style=app_style, children=[
    # Header
    html.Div(style=header_style, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}, children=[
            html.I(className="fab fa-bitcoin", style={
                'fontSize': '32px',
                'color': COLORS['bitcoin'],
                'marginRight': '10px'
            }),
            html.H1("Dashboard Bitcoin", style={'margin': 0})
        ])
    ]),
    
    # Tabs principales
    dcc.Tabs(id="main-tabs", value="tab-7", style={'marginBottom': '20px'}, children=[
        dcc.Tab(label="1. Introducción", value="tab-1", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="2. Contexto", value="tab-2", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="3. Planteamiento del Problema", value="tab-3", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="4. Objetivos y Justificación", value="tab-4", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="5. Marco Teórico", value="tab-5", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="6. Metodología", value="tab-6", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="7. Resultados y Análisis Final", value="tab-7", style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label="8. Conclusiones", value="tab-8", style=tab_style, selected_style=tab_selected_style),
    ]),
    
    # Contenido de las pestañas
    html.Div(id="tab-content", style=content_style)
])

# Callback para actualizar el contenido de la pestaña seleccionada
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value")
)
def render_content(tab):
    if tab == "tab-1":
        return html.Div(style=card_style, children=[
    # Imagen principal centrada
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
        html.Img(src="https://imagenes.20minutos.es/files/image_990_556/uploads/imagenes/2025/05/21/europapress-6738895-filed-05-march-2022-berlin-coin-bearing-the-logo-of-the-bitcoin.jpeg", 
                 style={'height': '250px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
    ]),
    
    # Título principal con icono
    html.H2("📘 Introducción al Análisis de Bitcoin", 
            style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2E86AB'}),
    
    # Texto introductorio mejorado
    html.Div(style={'marginBottom': '30px'}, children=[
        html.P("En la última década, Bitcoin ha revolucionado el panorama financiero global desde su creación en 2009, emergiendo como el primer activo digital descentralizado con capitalización de mercado superior a $1 billón.", 
              style={'textAlign': 'justify'}),
        
        html.P("A diferencia de los activos tradicionales, Bitcoin opera en un entorno descentralizado 24/7, con una volatilidad sin precedentes (70-100% anual) que presenta desafíos únicos para inversores e investigadores. Su naturaleza descentralizada le otorga propiedades únicas como escasez programada y resistencia a la censura, pero también lo convierte en un activo altamente sensible a eventos globales.", 
              style={'textAlign': 'justify'}),
        
        html.P("Este dashboard ofrece una visión integral del comportamiento de Bitcoin desde una perspectiva de ciencia de datos. A través de visualizaciones interactivas y modelos como Prophet, buscamos comprender su evolución histórica y anticipar tendencias futuras con fundamento técnico.", 
              style={'textAlign': 'justify'}),
        
        html.P("Dirigido tanto a principiantes como a analistas experimentados, este proyecto va más allá de la simple observación de precios, profundizando en el contexto, componentes estructurales y dinámica diaria del mercado de criptomonedas.", 
              style={'textAlign': 'justify'}),
    ]),
    
    # Imagen contextual
    html.Div(style={'textAlign': 'center', 'margin': '25px 0'}, children=[
    html.Img(
        src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTGM5tjK-rQc_TLtygdCEqNNDfoTrXVZ5tJwRi7l1edbFNzdjRxbas8CTFt0RYgd3-CDQ&usqp=CAU",
        style={
            'width': '80%', 
            'maxWidth': '700px', 
            'borderRadius': '10px', 
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'objectFit': 'cover',  # Para mantener proporciones
            'maxHeight': '400px'   # Controla altura máxima
        }
    ),
    html.P("Bitcoin: La revolución financiera digital", 
          style={'marginTop': '10px', 'fontStyle': 'italic', 'color': '#6c757d'})
]),
    
    # Sección de fuentes de datos
  # Sección de fuentes de datos
html.H3("Fuente de Datos", style={'marginTop': '30px', 'marginBottom': '15px', 'color': '#2E86AB', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
html.Div(style={
    'display': 'flex', 
    'backgroundColor': '#ffffff',
    'padding': '12px 20px',
    'borderRadius': '10px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
    'maxWidth': '600px',
    'margin': 'auto'
}, children=[
    html.Img(src="https://upload.wikimedia.org/wikipedia/commons/8/8f/Yahoo%21_Finance_logo_2021.png", 
             style={'height': '50px', 'marginRight': '20px'}),
    html.Div(children=[
        html.P("Datos históricos diarios desde 2014 obtenidos vía API de Yahoo Finance", style={
            'margin': '0', 
            'fontWeight': 'bold',
            'color': '#000000'
        }),
        html.P("Incluyendo precios de apertura, cierre, máximos, mínimos y volumen transaccionado", style={
            'margin': '5px 0 0', 
            'color': '#333333'
        })
    ])
]),

    
    # Tabla de variables clave
    html.H3("Variables Clave para el Análisis", style={'marginTop': '20px', 'marginBottom': '15px', 'color': '#2E86AB', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
    html.Div(style={'overflowX': 'auto'}, children=[
        html.Table(style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '30px'}, children=[
            html.Thead(html.Tr([
                html.Th("Variable", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#2E86AB', 'color': 'white'}),
                html.Th("Descripción", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#2E86AB', 'color': 'white'}),
                html.Th("Unidad", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#2E86AB', 'color': 'white'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Close", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'}),
                    html.Td("Precio de cierre diario - principal variable objetivo", style={'border': '1px solid #ddd', 'padding': '10px'}),
                    html.Td("USD", style={'border': '1px solid #ddd', 'padding': '10px'})
                ]),
                html.Tr([
                    html.Td("Volatility", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'}),
                    html.Td("Desviación estándar de rendimientos diarios (medida de riesgo)", style={'border': '1px solid #ddd', 'padding': '10px'}),
                    html.Td("%", style={'border': '1px solid #ddd', 'padding': '10px'})
                ]),
                html.Tr([
                    html.Td("Volume", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'}),
                    html.Td("Cantidad total de Bitcoin transaccionada durante el día", style={'border': '1px solid #ddd', 'padding': '10px'}),
                    html.Td("BTC", style={'border': '1px solid #ddd', 'padding': '10px'})
                ]),
                html.Tr([
                    html.Td("Returns", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'}),
                    html.Td("Cambio porcentual diario en el precio de cierre", style={'border': '1px solid #ddd', 'padding': '10px'}),
                    html.Td("%", style={'border': '1px solid #ddd', 'padding': '10px'})
                ]),
                html.Tr([
                    html.Td("Market Sentiment", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'}),
                    html.Td("Indicadores derivados de análisis de noticias y redes sociales", style={'border': '1px solid #ddd', 'padding': '10px'}),
                    html.Td("Index", style={'border': '1px solid #ddd', 'padding': '10px'})
                ])
            ])
        ])
    ]),
    
    # Cierre conceptual
    html.Div(children=[
        html.P("A través de este análisis multidimensional, buscamos descifrar los patrones ocultos en la aparente aleatoriedad del mercado de criptomonedas, proporcionando herramientas para una toma de decisiones más informada en este fascinante ecosistema financiero.", 
              style={'textAlign': 'center', 'margin': '0', 'fontStyle': 'italic'})
    ])
])
        
    elif tab == "tab-2":
        return html.Div(style=card_style, children=[
            html.H2("Volatilidad del Mercado de Criptomonedas", style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.P("El mercado de criptomonedas se ha consolidado como uno de los espacios más dinámicos y disruptivos del sistema financiero moderno. "
                "Su expansión ha estado impulsada por factores como la innovación tecnológica, la desintermediación financiera, la búsqueda de nuevas formas "
                "de inversión y la creciente digitalización de la economía global."),

            html.P("En particular, Bitcoin, como pionero de las criptomonedas, ha sido protagonista de múltiples ciclos de euforia y corrección, donde su valor ha "
                "oscilado de forma dramática en cortos periodos de tiempo. Estas fluctuaciones han sido motivadas por noticias regulatorias, movimientos "
                "institucionales, eventos globales y decisiones técnicas como los halvings o bifurcaciones de red."),

            html.P("La volatilidad, entendida como la magnitud y frecuencia de los cambios en el precio, es una de las características más distintivas de este mercado. "
                "Para los analistas y traders, representa una oportunidad para obtener beneficios en movimientos rápidos, pero también conlleva un alto riesgo "
                "para quienes no gestionan adecuadamente su exposición."),

            html.P("Comprender esta volatilidad no solo es clave para diseñar estrategias de inversión robustas, sino también para construir modelos predictivos que "
                "sean capaces de adaptarse a contextos altamente cambiantes. En este dashboard, se abordará esta problemática a través del uso de herramientas de "
                "análisis de series temporales que permitirán explorar y anticipar dichos movimientos con mayor claridad."),

            html.Img(
                src="https://www.criptonoticias.com/wp-content/uploads/2023/02/volatilidad-bitcoin-disparada-2023-750x375.jpg",
                style={'width': '70%', 'margin': 'auto', 'display': 'block', 'marginTop': '30px', 'borderRadius': '10px'}
            )
            
            
        ])
        
    elif tab == "tab-3":
        return html.Div(style=card_style, children=[
            html.H2("Planteamiento del Problema", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
                html.Img(
                    src="https://www.accountingweb.co.uk/sites/default/files/styles/inline_banner/public/istock-1345848884.jpg",
                    style={'maxWidth': '100%', 'height': 'auto', 'borderRadius': '8px'}
                )
            ]),

            html.P("El precio de Bitcoin se caracteriza por una alta volatilidad e incertidumbre, lo que representa un reto constante para "
                "inversionistas y analistas que buscan comprender y anticipar sus movimientos. Los métodos tradicionales de análisis financiero "
                "no siempre logran capturar la complejidad del mercado de criptomonedas, que responde a múltiples factores dinámicos y no lineales."),

            html.P("Frente a este contexto, surge la necesidad de aplicar modelos más flexibles y adaptativos que permitan generar predicciones precisas "
                "y útiles en escenarios cambiantes."),

            html.P([
                html.B("Pregunta problema: "),
                "¿Es posible construir un modelo que prediga de forma confiable el precio de cierre de Bitcoin a corto plazo, integrando visualizaciones "
                "claras y comprensibles para distintos tipos de usuarios?"
            ])
    ])

    elif tab == "tab-4":
        return html.Div(style=card_style, children=[
            # Header con gradiente y mejor jerarquía
            html.Div([
                html.H2("Objetivos y Justificación", 
                    style={
                        'textAlign': 'center', 
                        'marginBottom': '30px',
                        'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                        'backgroundClip': 'text',
                        'WebkitBackgroundClip': 'text',
                        'color': 'transparent',
                        'fontSize': '2.2rem',
                        'fontWeight': 'bold'
                    })
            ], style={'marginBottom': '40px'}),

            # Objetivo General con tarjeta destacada
            html.Div([
                    html.Div([
                        html.H3("📌 Objetivo General", 
                            style={
                                'color': '#2c3e50', 
                                'marginBottom': '15px',
                                'fontSize': '1.4rem',
                                'fontWeight': '600'
                            }),
                        html.P("Desarrollar un dashboard interactivo que permita analizar, visualizar y predecir el precio de cierre diario de Bitcoin, "
                            "aplicando modelos de series temporales como Prophet y técnicas de análisis exploratorio para una comprensión más profunda "
                            "de su comportamiento histórico y proyección futura.",
                            style={
                                'fontSize': '1.05rem',
                                'lineHeight': '1.6',
                                'color': '#495057',
                                'textAlign': 'justify'
                            })
                    ], style={
                        'backgroundColor': '#FFF8E1',  # un amarillo pastel claro
                        'padding': '25px',
                        'borderRadius': '12px',
                        'border': '1px solid #FFECB3',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                        'marginBottom': '35px'
                    })
                ])
,

            # Objetivos Específicos con mejor diseño de lista
            html.Div([
    html.H3("Objetivos Específicos", 
                style={
                    'color': "#c38a06", 
                    'marginBottom': '20px',
                    'fontSize': '1.4rem',
                    'fontWeight': '600'
                }),
            html.Div([
                # Repite para cada objetivo
                *[
                    html.Div([
                        html.Div(str(i + 1), style={
                            'backgroundColor': COLORS['primary'],
                            'color': COLORS['background'],
                            'borderRadius': '50%',
                            'width': '25px',
                            'height': '25px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'fontSize': '0.9rem',
                            'fontWeight': 'bold',
                            'marginRight': '12px',
                            'flexShrink': '0'
                        }),
                        html.P(texto, style={
                            'margin': '0',
                            'lineHeight': '1.5',
                            'color': '#495057'
                        })
                    ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '15px'})
                    for i, texto in enumerate([
                        "Realizar un análisis exploratorio exhaustivo de la serie de tiempo del precio de Bitcoin, identificando patrones, tendencias y volatilidad.",
                        "Aplicar el modelo Prophet incorporando regresores externos como volumen, RSI, y volatilidad histórica.",
                        "Evaluar el desempeño predictivo del modelo usando métricas como MAE, RMSE, R² y MAPE.",
                        "Descomponer el modelo en componentes interpretables como tendencia, estacionalidad y efecto de variables externas.",
                        "Presentar los resultados en una interfaz clara, dinámica y comprensible para todo tipo de usuario."
                    ])
                ]
            ], style={
                'backgroundColor': '#FFFDE7',  # otro tono suave amarillo pastel
                'padding': '25px',
                'borderRadius': '12px',
                'border': '1px solid #FFF59D',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'marginBottom': '35px'
            })
        ]),

            # Justificación con diseño de columnas y elementos visuales
            html.Div([
                html.H3("💡 Justificación del Proyecto", 
                    style={
                        'color': COLORS['primary'], 
                        'marginBottom': '25px',
                        'fontSize': '1.4rem',
                        'fontWeight': '600',
                        'textAlign': 'center'
                    }),
                
                # Tres tarjetas de justificación
                html.Div([
                    # Tarjeta 1: Innovación Técnica
                    html.Div([
                        html.Div("🚀", style={
                            'fontSize': '2.5rem',
                            'textAlign': 'center',
                            'marginBottom': '15px'
                        }),
                        html.H4("Innovación Técnica", style={
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'marginBottom': '15px',
                            'fontSize': '1.1rem'
                        }),
                        html.P("Bitcoin representa una nueva clase de activo financiero con comportamiento no tradicional. Su análisis requiere enfoques adaptativos "
                            "que integren estadísticas, visualización interactiva y modelado predictivo.",
                            style={
                                'fontSize': '0.95rem',
                                'lineHeight': '1.5',
                                'textAlign': 'justify',
                                'color': '#495057'
                            })
                    ], style={
                        'backgroundColor': "#F2D0DE",
                        'padding': '20px',
                        'borderRadius': '12px',
                        'border': '1px solid #bbdefb',
                        'height': '100%',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),
                    
                    # Tarjeta 2: Valor Práctico
                    html.Div([
                        html.Div("📊", style={
                            'fontSize': '2.5rem',
                            'textAlign': 'center',
                            'marginBottom': '15px'
                        }),
                        html.H4("Valor Práctico", style={
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'marginBottom': '15px',
                            'fontSize': '1.1rem'
                        }),
                        html.P("Este proyecto no solo tiene valor académico al integrar ciencia de datos con finanzas, sino también valor práctico, ya que puede servir "
                            "como herramienta de consulta para inversionistas, analistas, o cualquier usuario interesado en comprender mejor el ecosistema cripto.",
                            style={
                                'fontSize': '0.95rem',
                                'lineHeight': '1.5',
                                'textAlign': 'justify',
                                'color': '#495057'
                            })
                    ], style={
                        'backgroundColor': "#f3e5f5",
                        'padding': '20px',
                        'borderRadius': '12px',
                        'border': '1px solid #ce93d8',
                        'height': '100%',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }),
                    
                    # Tarjeta 3: Tecnología Moderna
                    html.Div([
                        html.Div("⚡", style={
                            'fontSize': '2.5rem',
                            'textAlign': 'center',
                            'marginBottom': '15px'
                        }),
                        html.H4("Tecnología Moderna", style={
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'marginBottom': '15px',
                            'fontSize': '1.1rem'
                        }),
                        html.P("El uso de Dash permite una presentación moderna e interactiva de los resultados, transformando análisis complejos en visualizaciones "
                            "intuitivas, accesibles y actualizables. La elección de Prophet responde a su capacidad para trabajar con estacionalidades múltiples.",
                            style={
                                'fontSize': '0.95rem',
                                'lineHeight': '1.5',
                                'textAlign': 'justify',
                                'color': '#495057'
                            })
                    ], style={
                        'backgroundColor': '#e8f5e8',
                        'padding': '20px',
                        'borderRadius': '12px',
                        'border': '1px solid #a5d6a7',
                        'height': '100%',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))',
                    'gap': '20px',
                    'marginTop': '20px'
                })
           ])
    ])
        
    elif tab == "tab-5":
        return html.Div(style=card_style, children=[
            # Header principal
            html.H2("📚 Marco Teórico y Fundamentos", 
                style={
                    'textAlign': 'center', 
                    'marginBottom': '30px',
                    'color': COLORS['primary'],
                    'fontSize': '2.2rem',
                    'fontWeight': 'bold'
                }),

            # Sección: Series Temporales Financieras
            html.Div([
                html.H3("Series Temporales en Finanzas Digitales", 
                    style={'color': COLORS['primary'], 'fontSize': '1.4rem', 'marginBottom': '15px'}),
                html.Div([
                    html.P("Una serie temporal financiera representa la evolución cronológica de un activo, donde cada observación "
                        "está correlacionada con valores pasados. En criptomonedas como Bitcoin, esta dependencia temporal es "
                        "especialmente marcada debido a la alta volatilidad y patrones de comportamiento del mercado.",
                        style={'marginBottom': '15px', 'lineHeight': '1.6', 'color': COLORS['text']}),
                    html.P("A diferencia de los activos tradicionales, Bitcoin opera 24/7, generando datos continuos que requieren "
                        "enfoques adaptativos para capturar sus patrones únicos de estacionalidad y cambios estructurales.",
                        style={'lineHeight': '1.6', 'color': COLORS['text']})
                ], style={
                    'backgroundColor': COLORS['sidebar'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'border': f'1px solid {COLORS["primary"]}',
                    'marginBottom': '30px'
                })
            ]),

            # Componentes implementados en el modelo
            html.Div([
                html.H3("Componentes del Modelo Prophet Implementado", 
                    style={'color': COLORS['primary'], 'fontSize': '1.4rem', 'marginBottom': '20px'}),
                
                # Grid de componentes
                html.Div([
                    # Tendencia
                    html.Div([
                        html.Div(style={'fontSize': '2rem', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.H4("Tendencia g(t)", style={'color': COLORS['secondary'], 'textAlign': 'center', 'fontSize': '1.1rem'}),
                        html.P("Movimiento direccional de largo plazo. Configurado con changepoint_prior_scale=0.6 "
                            "para detectar cambios estructurales en el precio de Bitcoin.",
                            style={'fontSize': '0.9rem', 'textAlign': 'center', 'color': COLORS['text']})
                    ], style={
                        'backgroundColor': COLORS['background'],
                        'padding': '15px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORS["secondary"]}'
                    }),
                    
                    # Estacionalidad
                    html.Div([
                        html.Div(style={'fontSize': '2rem', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.H4("Estacionalidad s(t)", style={'color': COLORS['secondary'], 'textAlign': 'center', 'fontSize': '1.1rem'}),
                        html.P("Patrones cíclicos regulares. Implementada estacionalidad semanal  y anual con "
                            "seasonality_prior_scale=20.0 para capturar patrones de trading.",
                            style={'fontSize': '0.9rem', 'textAlign': 'center', 'color': COLORS['text']})
                    ], style={
                        'backgroundColor': COLORS['background'],
                        'padding': '15px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORS["secondary"]}'
                    }),
                    
                    # Regresores
                    html.Div([
                        html.Div(style={'fontSize': '2rem', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.H4("Regresores Externos", style={'color': COLORS['secondary'], 'textAlign': 'center', 'fontSize': '1.1rem'}),
                        html.P("Variables explicativas normalizadas: volatilidad, RSI, ratio máximo-mínimo, "
                            "volumen logarítmico y ratio de media móvil.",
                            style={'fontSize': '0.9rem', 'textAlign': 'center', 'color': COLORS['text']})
                    ], style={
                        'backgroundColor': COLORS['background'],
                        'padding': '15px',
                        'borderRadius': '10px',
                        'border': f'1px solid {COLORS["secondary"]}'
                    })
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
                    'gap': '15px',
                    'marginBottom': '30px'
                })
            ]),

            # Ecuación del modelo implementado
            html.Div([
                html.H3("Formulación Matemática del Modelo", 
                    style={'color': COLORS['primary'], 'fontSize': '1.4rem', 'marginBottom': '15px'}),
                html.Div([
                    html.P("El modelo Prophet implementado sigue la estructura:", 
                        style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '1.1rem', 'color': COLORS['text']}),
                    html.Div([
                        html.P("y(t) = g(t) + s(t) + Σβᵢxᵢ(t) + εₜ", 
                            style={
                                'fontWeight': 'bold', 
                                'textAlign': 'center', 
                                'fontSize': '1.3rem',
                                'color': COLORS['bitcoin'],
                                'marginBottom': '20px'
                            }),
                    ], style={
                        'backgroundColor': COLORS['sidebar'],
                        'padding': '20px',
                        'borderRadius': '8px',
                        'border': f'2px solid {COLORS["primary"]}',
                        'marginBottom': '20px'
                    }),
                    
                    html.Div([
                        html.P("g(t): Tendencia con detección automática de cambios estructurales", 
                            style={'color': COLORS['text']}),
                        html.P("s(t): Estacionalidad semanal + anual (daily_seasonality=False)", 
                            style={'color': COLORS['text']}),
                        html.P("Σβᵢxᵢ(t): Suma de efectos de regresores normalizados", 
                            style={'color': COLORS['text']}),
                        html.P("εₜ: Error aleatorio con distribución normal", 
                            style={'color': COLORS['text']})
                    ], style={'fontSize': '1rem', 'lineHeight': '1.8'})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '30px',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                })
            ]),

            # Regresores técnicos implementados
            html.Div([
                html.H3("Indicadores Técnicos como Regresores", 
                    style={'color': COLORS['primary'], 'fontSize': '1.4rem', 'marginBottom': '20px'}),
                
                html.Div([
                    # Volatilidad
                    html.Div([
                        html.H4("Volatilidad Rolling (7 días)", 
                            style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                        html.P("Desviación estándar móvil que captura la incertidumbre del mercado. "
                            "Normalizada con StandardScaler para evitar dominancia de escala.",
                            style={'fontSize': '0.95rem', 'color': COLORS['text']})
                    ], style={'marginBottom': '15px'}),
                    
                    # RSI
                    html.Div([
                        html.H4("RSI (Relative Strength Index)", 
                            style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                        html.P("Oscilador de momento (ventana=14) que identifica condiciones de sobrecompra/sobreventa. "
                            "Calculado como: RSI = 100 - (100 / (1 + RS)), donde RS = Promedio_Ganancias / Promedio_Pérdidas",
                            style={'fontSize': '0.95rem', 'color': COLORS['text']})
                    ], style={'marginBottom': '15px'}),
                    
                    # Ratio HL
                    html.Div([
                        html.H4("High-Low Range Ratio", 
                            style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                        html.P("Medida de volatilidad intradía: (Máximo - Mínimo) / Precio_Cierre. "
                            "Indica la amplitud de movimiento relativo en cada sesión.",
                            style={'fontSize': '0.95rem', 'color': COLORS['text']})
                    ], style={'marginBottom': '15px'}),
                    
                    # Volumen
                    html.Div([
                        html.H4("Volumen Logarítmico", 
                            style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                        html.P("Log(Volumen + 1) para normalizar la alta variabilidad del volumen de trading. "
                            "Indicador de interés y liquidez del mercado.",
                            style={'fontSize': '0.95rem', 'color': COLORS['text']})
                    ], style={'marginBottom': '15px'}),
                    
                    # MA Ratio
                    html.Div([
                        html.H4("Moving Average Ratio", 
                            style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                        html.P("Precio_Actual / Media_Móvil_20días. Señala si el precio está por encima o debajo "
                            "de su tendencia reciente, útil para identificar momentum.",
                            style={'fontSize': '0.95rem', 'color': COLORS['text']})
                    ])
                ], style={
                    'backgroundColor': COLORS['sidebar'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '30px'
                })
            ]),

            # Metodología de evaluación
            html.Div([
                html.H3("Metodología de Evaluación Implementada", 
                    style={'color': COLORS['primary'], 'fontSize': '1.4rem', 'marginBottom': '15px'}),
                
                html.Div([
                    html.Div([
                        html.H4("Validación Temporal", 
                            style={'color': COLORS['secondary']}),
                        html.P("División train/test con los últimos 60 días como conjunto de prueba, "
                            "respetando el orden cronológico de los datos financieros.",
                            style={'color': COLORS['text']})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.H4("Métricas de Performance", 
                            style={'color': COLORS['secondary']}),
                        html.Ul([
                            html.Li("MAE (Error Absoluto Medio): Promedio de errores absolutos", 
                                style={'color': COLORS['text']}),
                            html.Li("RMSE (Raíz del Error Cuadrático Medio): Penaliza errores grandes", 
                                style={'color': COLORS['text']}),
                            html.Li("R² (Coeficiente de Determinación): Varianza explicada por el modelo", 
                                style={'color': COLORS['text']}),
                            html.Li("MAPE (Error Porcentual Absoluto Medio): Error relativo promedio", 
                                style={'color': COLORS['text']}),
                            html.Li("Coverage: % de valores reales dentro del intervalo de confianza del 95%", 
                                style={'color': COLORS['text']})
                        ])
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.H4("Ajustes Técnicos", 
                            style={'color': COLORS['secondary']}),
                        html.Ul([
                            html.Li("Corrección de sesgo: +1% del promedio de los últimos 30 días", 
                                style={'color': COLORS['text']}),
                            html.Li("Filtrado temporal: Datos desde 2023 para mejor performance", 
                                style={'color': COLORS['text']}),
                            html.Li("interval_width=0.95: Intervalos de confianza del 95%", 
                                style={'color': COLORS['text']}),
                            html.Li("Rellenado de regresores futuros con promedios de los últimos 7 días", 
                                style={'color': COLORS['text']})
                        ])
                    ])
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                })
            ])
    ])

    elif tab == "tab-6":
        return html.Div(style=card_style, children=[
            html.H2("Metodología", style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.H3("Enfoque General", style={'color': '#fcbf49'}),
            html.P("La metodología aplicada en este proyecto se basa en una combinación de análisis exploratorio de datos (EDA), ingeniería de características, "
                "modelado con Prophet y visualización interactiva mediante Dash. Se busca no solo pronosticar el precio de Bitcoin, sino también entender "
                "sus componentes clave y comunicar los resultados de forma clara y visual."),

            html.H3("1. Adquisición y Preprocesamiento de Datos", style={'color': '#fcbf49'}),
            html.Ul([
                html.Li("Los datos se obtuvieron desde Yahoo Finance y se agregaron a una base PostgreSQL previamente construida con información histórica del mercado de Bitcoin."),
                html.Li("Se convirtieron las fechas al formato datetime y se verificó la consistencia de los precios."),
                html.Li("Se aplicó relleno para valores faltantes y se eliminaron registros inválidos o inconsistentes."),
                html.Li("Se filtró la serie para trabajar principalmente desde 2023, por ser el rango más reciente y relevante."),
            ]),

            html.H3("2. Ingeniería de Características", style={'color': '#fcbf49'}),
            html.P("Para mejorar el poder predictivo del modelo, se calcularon varios indicadores técnicos y de volatilidad que se usaron como regresores externos en Prophet:"),
            html.Ul([
                html.Li("Volatilidad diaria (rolling std de 7 días)."),
                html.Li("Rango alto-bajo (normalizado por cierre)."),
                html.Li("RSI (Relative Strength Index, 14 periodos)."),
                html.Li("Razón del precio sobre su media móvil (MA ratio)."),
                html.Li("Volumen en logaritmo (para normalización).")
            ]),
            html.P("Estas variables fueron escaladas mediante `StandardScaler` para asegurar una correcta convergencia del modelo."),

            html.H3("3. Análisis Exploratorio (EDA)", style={'color': '#fcbf49'}),
            html.Ul([
                html.Li("Visualización de la evolución del precio a lo largo del tiempo."),
                html.Li("Distribución del precio y de los rendimientos diarios."),
                html.Li("Gráficos de velas para observar dinámica intradiaria."),
                html.Li("Estacionalidad semanal y análisis de volumen."),
                html.Li("Cálculo de medias móviles y análisis de tendencia."),
            ]),

            html.H3("4. Modelado con Prophet", style={'color': '#fcbf49'}),
            html.P("Se entrenó un modelo Prophet con estacionalidad semanal activada y múltiples regresores externos. Se evaluó su desempeño con un conjunto de prueba de 60 días."),
            html.Ul([
                html.Li("El modelo genera una predicción de 30 días hacia adelante."),
                html.Li("Se ajustó la tendencia y los intervalos de confianza."),
                html.Li("Se aplicó una corrección de sesgo con base en los últimos valores promedio."),
                html.Li("Se analizaron los componentes del modelo: tendencia, estacionalidad y efecto de regresores."),
            ]),

            html.H3("5. Evaluación del Modelo", style={'color': '#fcbf49'}),
            html.P("El rendimiento se evaluó usando métricas como:"),
            html.Ul([
                html.Li("MAE (Error Absoluto Medio)."),
                html.Li("RMSE (Raíz del Error Cuadrático Medio)."),
                html.Li("MAPE (Error Porcentual Absoluto Medio)."),
                html.Li("R² (Coeficiente de determinación)."),
                html.Li("Coverage: porcentaje de valores reales dentro del intervalo de confianza."),
            ]),

            html.H3("6. Visualización Interactiva", style={'color': '#fcbf49'}),
            html.P("Se utilizó Dash (framework de Python) para desarrollar una interfaz web con pestañas, gráficos dinámicos y análisis detallado."),
            html.Ul([
                html.Li("Gráficos en tiempo real usando Plotly."),
                html.Li("Diseño oscuro para facilitar la visualización prolongada."),
                html.Li("Pestañas temáticas para separar análisis, resultados y teoría."),
            ])
        ])

    elif tab == "tab-7":
        # Tab de Resultados con subtabs
        return html.Div([
            html.H2("Resultados y Análisis Final", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            dcc.Tabs(id="results-tabs", children=[
                dcc.Tab(label="a. Estadísticas BTC", value="tab-eda1", style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.Div(style=card_style, children=[
                            html.H3("Estadísticas Descriptivas", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            create_stats_card()
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Evolución del Precio", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_line_chart())
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Volumen de Transacciones", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_volume_chart())
                        ]),
                        
                        html.Div(style=card_style, children=[
                                html.H3("Mapa de Calor de Variaciones Diarias", style={'textAlign': 'center', 'marginBottom': '15px'}),
                                dcc.Graph(figure=create_heatmap_chart())
                            ]),

                        html.Div(style=card_style, children=[
                            html.H3("Distribución de Variables (Box Plot)", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_box_plot())
                        ])
                    ])
                ]),
                        
                dcc.Tab(label="b. EDA Avanzado", value="tab-eda2", style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.Div(style=card_style, children=[
                            html.H3("Gráfico de Velas (Candlestick)", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_candlestick_chart())
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Rendimientos Diarios", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_returns_chart())
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Matriz de Correlación", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_correlation_heatmap()),
                            html.P("La matriz de correlación muestra las relaciones lineales entre las diferentes variables de precio y volumen de Bitcoin.")
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Distribución de Precios", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_histogram())
                        ])
                    ])
                ]),

                # NUEVA PESTAÑA: Análisis Técnico
                dcc.Tab(label="c. Análisis Técnico", value="tab-technical", style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.Div(style=card_style, children=[
                            html.H3("Volatilidad Móvil (30 días)", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_volatility_chart()),
                            html.P("La volatilidad móvil muestra la variabilidad del precio de Bitcoin en ventanas de 30 días. Períodos de alta volatilidad suelen coincidir con eventos significativos del mercado.")
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Índice RSI (Relative Strength Index)", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_rsi_chart()),
                            html.P("El RSI es un oscilador de momentum que mide la velocidad y magnitud de los cambios de precio. Valores por encima de 70 indican sobrecompra, mientras que valores por debajo de 30 indican sobreventa.")
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Medias Móviles", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_moving_average_chart()),
                            html.P("Las medias móviles suavizan las fluctuaciones de precio para identificar tendencias. Cuando el precio está por encima de las medias móviles, sugiere una tendencia alcista.")
                        ])
                    ])
                ]),
                
                dcc.Tab(label="d. Análisis Prophet", value="tab-prophet-analysis", style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.Div(style=card_style, children=[
                            html.H3("Componentes del Modelo Prophet", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_prophet_components_chart(df)),
                            html.P("Prophet descompone la serie temporal en tendencia, estacionalidades y efectos de regresores externos. Esto permite entender mejor los patrones subyacentes en el precio de Bitcoin.")
                        ]),
                        
                        html.Div(style=card_style, children=[
                            html.H3("Interpretación de Componentes", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.P("El modelo Prophet identifica los siguientes patrones en Bitcoin:"),
                            html.Ul([
                                html.Li("Tendencia: Captura el comportamiento a largo plazo del precio, incluyendo cambios estructurales y puntos de inflexión."),
                                html.Li("Estacionalidad Semanal: Identifica patrones recurrentes durante los días de la semana, común en mercados financieros."),
                                html.Li("Estacionalidad Diaria: Captura variaciones intradiarias en el precio debido a horarios de mayor actividad."),
                                html.Li("Regresores Externos: Incorpora volatilidad, volumen, RSI y otros indicadores técnicos que mejoran la precisión.")
                            ])
                        ])
                    ])
                ]),
                
                dcc.Tab(
                    label="e. Modelo Prophet - Pronóstico", 
                    value="viz-prophet-model", 
                    style=tab_style, 
                    selected_style=tab_selected_style, 
                    children=[
                        html.Div(style=card_style, children=[
                            html.H3("Pronóstico Bitcoin con Prophet", style={'textAlign': 'center'}),
                            dcc.Graph(figure=fig),
                            html.P("Este gráfico muestra el pronóstico del precio de Bitcoin para los próximos 30 días utilizando el modelo Prophet. Se incluyen intervalos de confianza del 95% y el ajuste histórico del modelo."),
                            html.P("Prophet es especialmente efectivo para Bitcoin ya que maneja automáticamente cambios de tendencia y incorpora múltiples estacionalidades junto con regresores externos como volatilidad y volumen.")
                        ])
                    ]
                ),
                
                dcc.Tab(
                    label="f. Métricas del Modelo", 
                    value="tab-prophet-metrics", 
                    style=tab_style, 
                    selected_style=tab_selected_style, 
                    children=[
                        html.Div(style=card_style, children=[
                            html.H3("Métricas de Rendimiento - Prophet", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.P("El modelo Prophet ha sido evaluado con datos de prueba para determinar su precisión predictiva. A continuación se presentan las métricas principales:"),
                            html.Div([
                                html.Div([
                                    html.H4("Error Cuadrático Medio (RMSE)"),
                                    html.P(f"${rmse:.2f}"),
                                    html.Small("Penaliza más los errores grandes")
                                ], className="model-metric"),
                                html.Div([
                                    html.H4("Error Absoluto Medio (MAE)"),
                                    html.P(f"${mae:.2f}"),
                                    html.Small("Promedio de errores absolutos")
                                ], className="model-metric"),
                            
                                html.Div([
                                    html.H4("Error Porcentual Absoluto Medio (MAPE)"),
                                    html.P(f"{mape:.2f}%"),
                                    html.Small("Error relativo promedio")
                                ], className="model-metric"),
                                html.Div([
                                    html.H4("Cobertura del Intervalo"),
                                    html.P(f"{coverage:.1f}%"),
                                    html.Small("% de valores reales dentro del intervalo de confianza")
                                ], className="model-metric"),
                            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'}),
                            html.Br(),
                            html.P("Prophet ofrece varias ventajas sobre ARIMA para Bitcoin:"),
                            html.Ul([
                                html.Li("Manejo automático de estacionalidades múltiples"),
                                html.Li("Robustez ante valores atípicos"),
                                html.Li("Incorporación natural de regresores externos"),
                                html.Li("Intervalos de confianza más realistas"),
                                html.Li("Mayor interpretabilidad de componentes")
                            ])
                        ])
                    ]
                ),

                dcc.Tab(
                    label="g. Análisis de Residuos", 
                    value="tab-prophet-residuals", 
                    style=tab_style, 
                    selected_style=tab_selected_style, 
                    children=[
                        html.Div(style=card_style, children=[
                            html.H3("Análisis de Residuos - Prophet", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            dcc.Graph(figure=create_residuals_analysis_chart(df)),
                            html.P("El análisis de residuos permite evaluar la calidad del ajuste del modelo:"),
                            html.Ul([
                                html.Li("Predicción vs Real: Muestra qué tan cerca están las predicciones de los valores reales"),
                                html.Li("Residuos en el Tiempo: Identifica patrones temporales no capturados por el modelo"),
                                html.Li("Distribución de Residuos: Evalúa si los errores siguen una distribución normal"),
                                html.Li("Q-Q Plot: Compara la distribución de residuos con una distribución normal teórica")
                            ])
                        ])
                    ]
                ),
            ])
        ])
            
    elif tab == "tab-8":
        return html.Div(style=card_style, children=[
            html.H2("📌 Conclusiones", style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.P("El análisis del precio de cierre diario de Bitcoin realizado en este proyecto permitió obtener una visión profunda sobre "
                "la naturaleza volátil, cíclica y estructurada de este activo digital. A través del uso del modelo Prophet, combinado con regresores "
                "externos y visualizaciones interactivas, se logró capturar no solo el comportamiento histórico de la serie temporal, sino también "
                "generar pronósticos informativos y comprensibles."),

            html.P("Los resultados mostraron que la incorporación de variables como el volumen, la volatilidad y el RSI mejora significativamente la "
                "precisión del modelo y permite explicar parte de las fluctuaciones del mercado. Las métricas de evaluación obtenidas (MAE, RMSE, MAPE, R²) "
                "demuestran un buen desempeño predictivo en el corto plazo."),

            html.P("Además, el análisis de componentes descompuestos del modelo revela patrones estacionales consistentes (especialmente a nivel semanal), "
                "así como puntos de cambio en la tendencia del precio. Esto aporta valor no solo desde un enfoque predictivo, sino también desde una perspectiva "
                "explicativa del comportamiento del mercado."),

            html.P("En conclusión, la aplicación de técnicas modernas de análisis de series temporales como Prophet, integradas en una plataforma visual e interactiva, "
                "representa una herramienta poderosa para estudiar activos financieros emergentes como Bitcoin. Este tipo de soluciones puede ser útil tanto para investigadores "
                "académicos como para profesionales del sector financiero, democratizando el acceso a modelos estadísticos complejos a través de una experiencia intuitiva.")
        ])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)