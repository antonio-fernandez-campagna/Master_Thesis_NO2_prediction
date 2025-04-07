import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import seaborn as sns
import altair as alt
import calplot
import joblib
import os
from pygam import LinearGAM, s, f, te
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==========================================
# CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================

@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """
    Carga y preprocesa los datos con caché de Streamlit.
    
    Returns:
        DataFrame: Datos de NO2, tráfico y meteorología con fecha convertida a datetime.
    """
    df = pd.read_parquet('data/more_processed/no2_with_traffic_and_meteo_one_station.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def preprocessing(df):
    """
    Preprocesamiento de los datos: creación de variables cíclicas para tiempo.
    
    Args:
        df (DataFrame): DataFrame original.
        
    Returns:
        DataFrame: DataFrame con variables cíclicas añadidas.
    """
    df = df.copy()
    
    # Crear variables cíclicas para variables temporales (hora, mes, día)
    # Esto permite capturar la naturaleza periódica del tiempo
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)

    return df


def split_data(df, fecha_fin_training):
    """
    Divide los datos en conjuntos de entrenamiento y prueba según fechas.
    
    Args:
        df (DataFrame): DataFrame completo.
        train_date (datetime): Fecha límite para entrenamiento.
        test_date (datetime): Fecha inicio para prueba.
        
    Returns:
        tuple: (train_df, test_df) DataFrames de entrenamiento y prueba.
    """
    train_df = df[df['fecha'] < fecha_fin_training]
    test_df = df[df['fecha'] >= fecha_fin_training]
    return train_df, test_df


COLUMNS_FOR_OUTLIERS = [
    'no2_value',         # concentración de NO₂ (target, revisar cuidadosamente)
    'intensidad',        # intensidad del tráfico
    'carga',             # carga del tráfico
    'ocupacion',         # ocupación
    'vmed',              # velocidad media
    'd2m', 't2m',        # temperatura del punto de rocío y temperatura 2m
    'ssr', 'ssrd',       # shortwave radiation
    'u10', 'v10',        # componentes del viento
    'sp',                # presión
    'tp',                # precipitación
]

import pandas as pd

def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Elimina outliers usando el método del IQR (interquartile range)."""
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns:
            Q1 = filtered_df[col].quantile(0.25)
            Q3 = filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df


def remove_outliers_zscore(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Elimina outliers usando el método del z-score (valores con desviación estándar alta)."""
    from scipy.stats import zscore
    filtered_df = df.copy()
    zscores = filtered_df[columns].apply(zscore)
    condition = (zscores.abs() < threshold).all(axis=1)
    return filtered_df[condition]


def remove_outliers_quantiles(df: pd.DataFrame, columns: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Elimina outliers basados en percentiles extremos."""
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns:
            lower = filtered_df[col].quantile(lower_q)
            upper = filtered_df[col].quantile(upper_q)
            filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
    return filtered_df


def filter_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    if method == 'iqr':
        return remove_outliers_iqr(df, COLUMNS_FOR_OUTLIERS)
    elif method == 'zscore':
        return remove_outliers_zscore(df, COLUMNS_FOR_OUTLIERS)
    elif method == 'quantiles':
        return remove_outliers_quantiles(df, COLUMNS_FOR_OUTLIERS)
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa: 'iqr', 'zscore' o 'quantiles'.")
    
def convertir_unidades_legibles(df):
    """
    Convierte las variables meteorológicas del ERA5 a unidades más interpretables.
    
    Conversiones:
    - Temperaturas: Kelvin a Celsius
    - Radiación: J/m² a W/m²
    - Viento: m/s a km/h
    - Presión: Pa a hPa
    - Precipitación: m a mm
    """
    df = df.copy()
    
    # Temperatura y punto de rocío (K -> °C)
    df['d2m'] = df['d2m'] - 273.15  # Punto de rocío a 2m
    df['t2m'] = df['t2m'] - 273.15  # Temperatura a 2m
    
    # Radiación solar (J/m² -> W/m²)
    df['ssr'] = df['ssr'] / 3600    # Radiación de onda corta neta
    df['ssrd'] = df['ssrd'] / 3600  # Radiación de onda corta descendente
    
    # Componentes del viento (m/s -> km/h)
    df['u10'] = df['u10'] * 3.6     # Componente U del viento a 10m
    df['v10'] = df['v10'] * 3.6     # Componente V del viento a 10m
    
    # Calcular velocidad y dirección del viento
    df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['wind_direction'] = (270 - np.arctan2(df['v10'], df['u10']) * 180/np.pi) % 360
    
    # Presión superficial (Pa -> hPa)
    df['sp'] = df['sp'] / 100
    
    # Precipitación (m -> mm)
    df['tp'] = df['tp'] * 1000
    
    return df

# Añadir un diccionario con metadatos de las variables
VARIABLE_METADATA = {
    'd2m': {'name': 'Punto de Rocío', 'unit': '°C', 'typical_range': (-10, 30)},
    't2m': {'name': 'Temperatura', 'unit': '°C', 'typical_range': (-5, 40)},
    'ssr': {'name': 'Radiación Solar Neta', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'ssrd': {'name': 'Radiación Solar Descendente', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'wind_speed': {'name': 'Velocidad del Viento', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_direction': {'name': 'Dirección del Viento', 'unit': '°', 'typical_range': (0, 360)},
    'sp': {'name': 'Presión Superficial', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'tp': {'name': 'Precipitación Total', 'unit': 'mm', 'typical_range': (0, 50)}
}

def estandarizar_variables(df, columnas):
    """
    Estandariza las variables seleccionadas y guarda los parámetros de escalado.
    """
    df = df.copy()
    scaler_dict = {}
    
    for col in columnas:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scaler_dict[col] = scaler
        
    return df, scaler_dict

def desescalar_variables(df, scaler_dict):
    df = df.copy()
    for col, scaler in scaler_dict.items():
        df[col] = scaler.inverse_transform(df[[col]])
    return df

def escalar_target(y):
    """
    Escala la variable objetivo y devuelve el scaler para posterior uso.
    """
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    return y_scaled, scaler


# ==========================================
# ENTRENAMIENTO DEL MODELO
# ==========================================

def gam_train(X_train, y_train, feature_names, sensor_seleccionado, outlier_type, preprocessing_box, scaler_dict, scaler_target):
    """
    Entrenamiento del modelo GAM con variables escaladas.
    """
    # Crear modelo GAM con splines para cada variable
    # s() indica que se usará una función suave (spline) para modelar la relación
    gam = LinearGAM(
        s(0) + s(1) +                  # Hour cyclical terms
        s(2) + s(3) +                  # Month cyclical terms
        s(4) + s(5) +                  # Day cyclical terms
        s(6, n_splines=5) +           # intensidad
        s(7, n_splines=5) +           # carga
        s(8, n_splines=5) +           # ocupacion
        s(9, n_splines=5) +           # vmed
        s(10, n_splines=5) +          # d2m (2m dewpoint temperature)
        s(11, n_splines=5) +          # t2m (2m temperature)
        s(12, n_splines=5) +          # ssr (surface solar radiation)
        s(13, n_splines=5) +          # ssrd (surface solar radiation downwards)
        s(14, n_splines=5) +          # u10 (10m u-wind component)
        s(15, n_splines=5) +          # v10 (10m v-wind component)
        s(16, n_splines=5) +          # sp (surface pressure)
        s(17, n_splines=5)            # tp (total precipitation)
    )

    # Ajustar el modelo
    print("Entrenando modelo...")
    gam.fit(X_train, y_train)
    gam.feature_names = feature_names  # Guardar nombres de características para interpretación

    print("Modelo entrenado correctamente")

    # Guardar modelo y scalers juntos
    model_info = {
        'model': gam,
        'feature_names': feature_names,
        'scaler_dict': scaler_dict,
        'scaler_target': scaler_target,
        'variable_metadata': VARIABLE_METADATA
    }
    
    file_name = f'data/models/gam_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_box}.pkl'
    joblib.dump(model_info, file_name)
    print(f"Modelo guardado en {file_name}")



def gam_evaluate(model, X_test, y_test):
    """
    Evaluación del modelo mediante error cuadrático medio.
    
    Args:
        model: Modelo GAM entrenado.
        X_test (DataFrame): Variables predictoras para prueba.
        y_test (Series): Variable objetivo para prueba.
        
    Returns:
        float: Error cuadrático medio.
    """
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


# ==========================================
# VISUALIZACIÓN DE EFECTOS DEL MODELO
# ==========================================

def show_gam_model_partial_dependence(model, feature_names):
    """
    Muestra las gráficas de dependencia parcial del modelo para cada variable.
    Esta función usa el método integrado del GAM para mostrar cómo cada variable
    afecta la predicción, manteniendo las demás constantes.
    
    DIFERENCIA PRINCIPAL: Muestra el efecto de cada variable de forma independiente,
    directamente usando los métodos del modelo GAM.
    
    Args:
        model: Modelo GAM entrenado.
        feature_names (list): Lista de nombres de características.
    """
    # Crear un subtítulo en Streamlit
    st.subheader("Gráficas de Dependencia Parcial")
    st.markdown("""
    Estas gráficas muestran cómo cada variable afecta la predicción de NO₂, 
    manteniendo todas las demás variables constantes. El eje Y representa el 
    efecto en la predicción.
    """)
    
    # Crear una disposición en cuadrícula con columnas
    cols = 4  # Número de columnas en la cuadrícula
    rows = (len(feature_names) + cols - 1) // cols  # Calcular filas necesarias
    
    # Crear la cuadrícula
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(feature_names):
                feature = feature_names[idx]
                with columns[col]:
                    fig, ax = plt.figure(figsize=(5, 4)), plt.gca()
                    # Generar una cuadrícula de valores para la variable actual
                    XX = model.generate_X_grid(term=idx)
                    
                    # Manejar ambos casos: cuando el método devuelve tupla o valor único
                    try:
                        # Obtener dependencia parcial y intervalos de confianza
                        pdep, confi = model.partial_dependence(term=idx, X=XX, width=0.95)
                        ax.plot(XX[:, idx], pdep)
                        ax.fill_between(XX[:, idx], confi[:, 0], confi[:, 1], alpha=0.2)
                    except ValueError:
                        # Solo dependencia parcial sin intervalos de confianza
                        pdep = model.partial_dependence(term=idx, X=XX)
                        ax.plot(XX[:, idx], pdep)
                    
                    ax.set_title(f'Dependencia Parcial: {feature}')
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)  # Línea horizontal en y=0
                    st.pyplot(fig)
                    plt.close(fig)


def get_partial_effects(model, feature_names, values, sin_name=None, cos_name=None):
    """
    Calcula el efecto parcial para variables específicas o pares de variables cíclicas.
    
    DIFERENCIA PRINCIPAL: Esta función permite combinar los efectos de las variables
    cíclicas (sin/cos) para mostrar su efecto conjunto, o calcular efectos para 
    valores específicos de una variable.
    
    Args:
        model: Modelo GAM entrenado.
        feature_names (list): Lista de nombres de características.
        values (array): Valores para los que calcular el efecto.
        sin_name (str, optional): Nombre de la variable seno para variables cíclicas.
        cos_name (str, optional): Nombre de la variable coseno para variables cíclicas.
        
    Returns:
        list: Lista de efectos parciales calculados.
    """
    effects = []
    for val in values:
        # Inicializar vector de ceros para todas las variables
        XX = np.zeros((1, len(feature_names)))

        # Caso 1: Variables cíclicas (par seno/coseno)
        if sin_name and cos_name:
            # Convertir valor a representación seno/coseno
            sin_val = np.sin(2 * np.pi * val / len(values))
            cos_val = np.cos(2 * np.pi * val / len(values))
            
            # Asignar valores seno/coseno a las posiciones correspondientes
            XX[0, feature_names.index(sin_name)] = sin_val
            XX[0, feature_names.index(cos_name)] = cos_val
            
            # Calcular efectos parciales para ambas componentes y sumarlos
            sin_eff = model.partial_dependence(term=feature_names.index(sin_name), X=XX)
            cos_eff = model.partial_dependence(term=feature_names.index(cos_name), X=XX)
            effects.append(float(sin_eff + cos_eff))

        # Caso 2: Variable simple (no cíclica)
        elif sin_name:
            XX[0, feature_names.index(sin_name)] = val
            eff = model.partial_dependence(term=feature_names.index(sin_name), X=XX)
            effects.append(float(eff))

    return effects


def plot_effect(x_vals, y_vals, title, xlabel, ylabel, feature_name=None, xticks=None):
    """
    Función mejorada para graficar efectos con unidades correctas.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals)
    
    if feature_name in VARIABLE_METADATA:
        metadata = VARIABLE_METADATA[feature_name]
        xlabel = f"{metadata['name']} ({metadata['unit']})"
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True)
    
    # Añadir información sobre el rango típico si está disponible
    if feature_name in VARIABLE_METADATA:
        range_min, range_max = VARIABLE_METADATA[feature_name]['typical_range']
        ax.axvspan(range_min, range_max, alpha=0.1, color='green', label='Rango típico')
        ax.legend()
    
    st.pyplot(fig)
    plt.close(fig)


def show_temporal_effects(model, feature_names):
    """
    Muestra efectos temporales y otros efectos relevantes en NO₂.
    Esta función utiliza get_partial_effects para crear visualizaciones específicas.
    
    Args:
        model: Modelo GAM entrenado.
        feature_names (list): Lista de nombres de características.
    """
    st.write("## Efectos Temporales y Variables Clave en NO₂")
    st.markdown("""
    Estas gráficas muestran cómo diferentes factores afectan los niveles de NO₂.
    Para variables cíclicas como hora, día y mes, se combinan los efectos de las 
    componentes seno y coseno para mostrar el impacto real.
    """)

    # Crear 3 columnas para organizar las visualizaciones
    col1, col2, col3 = st.columns(3)

    with col1:
        # Efecto de la hora del día
        st.write("#### Efecto de la Hora del Día")
        hours = np.arange(0, 24)
        hour_effects = get_partial_effects(model, feature_names, hours, sin_name='hour_sin', cos_name='hour_cos')
        plot_effect(hours, hour_effects, 'Efecto de la Hora del Día en NO₂', 'Hora', 'Efecto en NO₂', xticks=range(0, 24, 2))

        # Efecto de la intensidad de tráfico
        st.write("#### Efecto de la Intensidad de Tráfico")
        intensidad_range = np.linspace(0, 1500, 100)
        intensidad_effects = get_partial_effects(model, feature_names, intensidad_range, sin_name='intensidad')
        plot_effect(intensidad_range, intensidad_effects, 'Efecto de la Intensidad en NO₂', 'Intensidad', 'Efecto en NO₂')

    with col2:
        # Efecto del día de la semana
        st.write("#### Efecto del Día de la Semana (0=Lunes, 6=Domingo)")
        days = np.arange(0, 7)
        day_effects = get_partial_effects(model, feature_names, days, sin_name='day_sin', cos_name='day_cos')
        plot_effect(days, day_effects, 'Efecto del Día de la Semana en NO₂', 'Día de la Semana', 'Efecto en NO₂', xticks=range(0, 7))

        # Efecto de la temperatura
        st.write("#### Efecto de la Temperatura")
        t2m_range = np.linspace(-5, 40, 100)
        t2m_effects = get_partial_effects(model, feature_names, t2m_range, sin_name='t2m')
        plot_effect(t2m_range, t2m_effects, 'Efecto de la Temperatura en NO₂', 'Temperatura (°C)', 'Efecto en NO₂')

    with col3:
        # Efecto del mes
        st.write("#### Efecto del Mes")
        months = np.arange(1, 13)
        month_effects = get_partial_effects(model, feature_names, months, sin_name='month_sin', cos_name='month_cos')
        plot_effect(months, month_effects, 'Efecto del Mes en NO₂', 'Mes', 'Efecto en NO₂', xticks=range(1, 13))
        
        # Efecto de la velocidad media
        st.write("#### Efecto de la Velocidad Media")
        vmed_range = np.linspace(0, 100, 100)
        vmed_effects = get_partial_effects(model, feature_names, vmed_range, sin_name='vmed')
        plot_effect(vmed_range, vmed_effects, 'Efecto de la Velocidad Media en NO₂', 'Velocidad (km/h)', 'Efecto en NO₂')

    # Sección adicional con más visualizaciones
    st.write("## Efectos Meteorológicos Adicionales")
    col1, col2 = st.columns(2)
    
    with col1:
        # Efecto de la precipitación
        st.write("#### Efecto de la Precipitación")
        tp_range = np.linspace(0, 50, 100)  # Ajustar según rango de datos
        tp_effects = get_partial_effects(model, feature_names, tp_range, sin_name='tp')
        plot_effect(tp_range, tp_effects, 'Efecto de la Precipitación en NO₂', 'Precipitación (mm)', 'Efecto en NO₂')
    
    with col2:
        # Efecto de la radiación solar
        st.write("#### Efecto de la Radiación Solar")
        ssr_range = np.linspace(0, 1000, 100)  # Ajustar según rango de datos
        ssr_effects = get_partial_effects(model, feature_names, ssr_range, sin_name='ssr')
        plot_effect(ssr_range, ssr_effects, 'Efecto de la Radiación Solar en NO₂', 'Radiación (J/m²)', 'Efecto en NO₂')


def show_gam_model_stats(model, X_test, y_test, feature_names, scaler_target):
    """
    Muestra las estadísticas del modelo y visualizaciones de efectos.
    
    Args:
        model: Modelo GAM entrenado.
        X_test (DataFrame): Variables predictoras para prueba.
        y_test (Series): Variable objetivo para prueba.
        feature_names (list): Lista de nombres de características.
        scaler_target: Scaler para desescalar la variable objetivo.
    """
    # Calcular y mostrar métricas de evaluación
    y_pred = model.predict(X_test)
    y_pred_descaled = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_descaled))
    r2 = r2_score(y_test, y_pred_descaled)
    
    # Crear panel de métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f} µg/m³", "Menor es mejor")
    col2.metric("R² Score", f"{r2:.3f}", "Más cercano a 1 es mejor")
    col3.metric("MAE", f"{np.mean(np.abs(y_test - y_pred_descaled)):.2f} µg/m³", "Menor es mejor")
    
    # Mostrar distribución de residuos
    st.write("### Distribución de Residuos")
    residuals = y_test - y_pred_descaled
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma de residuos
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title('Histograma de Residuos')
    ax[0].set_xlabel('Residuo (µg/m³)')
    
    # Gráfico Q-Q de residuos
    sm.qqplot(residuals, line='45', ax=ax[1], fit=True)
    ax[1].set_title('Gráfico Q-Q de Residuos')
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Mostrar visualizaciones de efectos
    tabs = st.tabs(["Efectos Temporales y Clave", "Dependencias Parciales Detalladas"])
    
    with tabs[0]:
        show_temporal_effects(model, feature_names)
    
    with tabs[1]:
        show_gam_model_partial_dependence(model, feature_names)


# ==========================================
# INTERFAZ DE USUARIO DE STREAMLIT
# ==========================================

def training_page():
    """Página principal de entrenamiento y análisis del modelo."""
    
    # st.markdown("""
    # Esta aplicación permite entrenar y analizar modelos GAM (Generalized Additive Models) 
    # para estudiar los factores que influyen en los niveles de NO₂.
    # """)
    
    # Sección para analizar modelo existente
    st.header("GAM Análisis sobre NO2 - Configuración")
    
    # Modelo por defecto a cargar
    default_model_name = 'gam_model'
    model_path = 'data/models/' + default_model_name
    file_name = ""
    extension = ".pkl"

    # Cargar datos
    df = cargar_datos_trafico_y_meteo()
    
    # Configuración de fechas y sensor
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Selector de sensor
        sensor_seleccionado = st.selectbox(
            "Selecciona un sensor de NO₂",
            df['id_no2'].unique(),
            index=2
        )
        
        # Filtrar por sensor
        df_sensor = df[df['id_no2'] == sensor_seleccionado]
        
        # Obtener rango de fechas disponibles
        fecha_min = df_sensor["fecha"].min().date()
        fecha_max = df_sensor["fecha"].max().date()

        # Selectores de fechas para división de datos
        fecha_inicio = st.date_input(
            "Fecha de inicio para entrenamiento", 
            fecha_min, 
            min_value=fecha_min, 
            max_value=fecha_max,
            key="fecha_inicio_key"  
        )

        fecha_fin_training = st.date_input(
            "Fecha de inicio para evaluación", 
            '2024-01-01', 
            min_value=fecha_min, 
            max_value=fecha_max,
            key="fecha_fin_key"  
        )

    with config_col2:
        
        # Convertir a datetime para comparación
        fecha_inicio_dt = pd.to_datetime(fecha_inicio)
        fecha_fin_training_dt = pd.to_datetime(fecha_fin_training)

        # select box para qué tipo filtrado de outlier se quiere aplicar.
        outlier_type = st.selectbox(
            "Tipo de filtrado de outliers",
            ["zscore", "iqr", "quantiles","none"]
        )
        
        # select box para saber si aplicar preprocessing, con esto, se crearan 
        # el sin y cos de las variables temporales. Esto sirve para que el modelo
        # pueda entender las variables temporales y no lo vea como valores discretos.
        preprocessing_box = st.selectbox(
            "Preprocesamiento",
            ["sin_cos", "none"]
        )
    
    # Selector de variables
    st.subheader("Selección de Variables")
    
    # Mostrar categorías de variables
    var_categories = {
        "Variables Temporales": ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos'],
        "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
        "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp']
    }
    
    # Crear tabs para categorías de variables
    var_tabs = st.tabs(list(var_categories.keys()))
    
    selected_features = []
    for i, (category, vars_list) in enumerate(var_categories.items()):
        with var_tabs[i]:
            selected_in_category = st.multiselect(
                f"Variables de {category}",
                vars_list,
                default=vars_list
            )
            selected_features.extend(selected_in_category)

    if outlier_type == "none":
        df_sensor = df_sensor
    else:
        df_sensor = filter_outliers(df_sensor, outlier_type)

    if preprocessing_box == "sin_cos":
        df_sensor = preprocessing(df_sensor)

    df_sensor = convertir_unidades_legibles(df_sensor)
    
    # Variable objetivo
    target = 'no2_value'
    
    # Dividir datos
    train_df, test_df = split_data(df_sensor, fecha_fin_training_dt)
    
    # Preparar conjuntos de entrenamiento y prueba
    X_train = train_df[selected_features].copy()
    y_train = train_df[target].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[target].copy()

    # Escalar las variables
    X_train, scaler_dict = estandarizar_variables(X_train, selected_features)
    X_test = X_test.copy()
    for col in selected_features:
        X_test[col] = scaler_dict[col].transform(X_test[[col]])
    
    # Escalar el target
    y_train_scaled, scaler_target = escalar_target(y_train)

    full_model_path  = model_path + file_name + "_" + sensor_seleccionado + "_" + outlier_type + "_" + preprocessing_box + extension
    
    if os.path.exists(full_model_path):
        model_info = joblib.load(full_model_path)
        model = model_info['model']
        scaler_dict = model_info['scaler_dict']
        scaler_target = model_info['scaler_target']
        
        if st.button("Analizar Modelo con Datos Seleccionados"):
            with st.spinner("Analizando modelo..."):
                # Desescalar para las visualizaciones
                X_test_original = desescalar_variables(X_test.copy(), scaler_dict)
                show_gam_model_stats(model, X_test_original, y_test, selected_features, scaler_target)
    else:
        st.info(f"No se encontró el modelo {full_model_path}. Por favor, entrene un nuevo modelo.")
        # Botón para entrenar nuevo modelo
        if st.button("Entrenar Nuevo Modelo", key="train_button"):
            with st.spinner("Entrenando modelo... Este proceso puede tardar varios minutos."):
                X_train, scaler_dict = estandarizar_variables(X_train, selected_features)
                X_test = X_test.copy()
                for col in selected_features:
                    X_test[col] = scaler_dict[col].transform(X_test[[col]])
                y_train_scaled, scaler_target = escalar_target(y_train)
                gam_train(X_train, y_train_scaled, selected_features, sensor_seleccionado, outlier_type, preprocessing_box, scaler_dict, scaler_target)
            st.success("¡Modelo entrenado correctamente!")
    

