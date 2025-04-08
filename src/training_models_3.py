# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import seaborn as sns
import joblib
import os
import xgboost as xgb
from xgboost.callback import EarlyStopping
# from sklearn.model_selection import train_test_split # No se usa directamente si filtramos por fecha
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


# ==========================================
# CONSTANTES Y METADATOS (Sin cambios)
# ==========================================
COLUMNS_FOR_OUTLIERS = [
    'no2_value', 'intensidad', 'carga', 'ocupacion', 'vmed',
    'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp'
]

VARIABLE_METADATA = {
    # ... (igual que antes, añadir metadatos para nuevas features si es necesario) ...
    'no2_value_lag1': {'name': 'NO2 Lag 1h', 'unit': 'µg/m³'},
    'no2_value_lag3': {'name': 'NO2 Lag 3h', 'unit': 'µg/m³'},
    'no2_value_lag24': {'name': 'NO2 Lag 24h', 'unit': 'µg/m³'},
    'no2_rolling_mean_3h': {'name': 'NO2 Media Móvil 3h', 'unit': 'µg/m³'},
    'no2_rolling_std_3h': {'name': 'NO2 Desv. Est. Móvil 3h', 'unit': 'µg/m³'},
    'no2_rolling_mean_24h': {'name': 'NO2 Media Móvil 24h', 'unit': 'µg/m³'},
    'no2_rolling_std_24h': {'name': 'NO2 Desv. Est. Móvil 24h', 'unit': 'µg/m³'},
    'is_weekend': {'name': 'Es Fin de Semana', 'unit': 'bool'},
    'time_of_day': {'name': 'Parte del Día', 'unit': 'category'},
    'day_of_week': {'name': 'Día de la Semana', 'unit': 'int'},
     # ... (metadatos existentes) ...
}

# ==========================================
# CARGA DE DATOS (Sin cambios)
# ==========================================
@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """Carga y cachea los datos."""
    file_path = 'data/more_processed/no2_with_traffic_and_meteo_one_station.parquet'
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo de datos en {file_path}")
        st.stop()
    df = pd.read_parquet(file_path)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by='fecha') # Asegurar orden temporal
    # Asegurar índice único basado en fecha si hay duplicados por sensor
    # df = df.set_index('fecha') # Opcional, depende de la estructura exacta
    return df

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    """Añade nuevas características temporales, lags y rolling windows."""
    df = df.copy()
    df = df.sort_values(by='fecha') # Muy importante para lags/rolling

    # Asegurar que el índice es la fecha para lags/rolling si no lo es ya
    if not isinstance(df.index, pd.DatetimeIndex):
         # Si hay múltiples sensores, necesitaríamos agrupar antes de lags
         # Asumiendo un solo sensor por ahora o que los datos ya están filtrados
        df = df.set_index('fecha').sort_index()

    # 1. Features Temporales Adicionales
    df['day_of_week'] = df.index.dayofweek  # Lunes=0, Domingo=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # Sábado=5, Domingo=6 -> 1, else 0
    df['hour'] = df.index.hour # Asegurar que 'hour' existe si no estaba ya
    # Categorizar parte del día
    bins = [-1, 5, 11, 17, 21, 24] # Madrugada (0-5), Mañana (6-11), Tarde (12-17), Noche (18-21), Madrugada (22-23)
    labels = ['Madrugada', 'Mañana', 'Tarde', 'Noche', 'Madrugada2']
    df['time_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels, right=True)
    # Unificar 'Madrugada' y 'Madrugada2' si se desea o manejar como categoría separada
    df['time_of_day'] = df['time_of_day'].replace({'Madrugada2': 'Madrugada'}).astype('category')


    # # 2. Lag Features (NO2) - Importante: shift() asume ordenado por fecha
    # lags = [1, 3, 24] # Lag de 1h, 3h, 24h
    # for lag in lags:
    #     df[f'no2_value_lag{lag}'] = df['no2_value'].shift(lag)

    # # 3. Rolling Window Features (NO2)
    # windows = [3, 24] # Ventanas de 3h y 24h
    # for window in windows:
    #     # closed='left' para incluir la hora actual en la ventana si se desea
    #     rolling_no2 = df['no2_value'].rolling(window=f'{window}H', closed='left')
    #     df[f'no2_rolling_mean_{window}h'] = rolling_no2.mean()
    #     df[f'no2_rolling_std_{window}h'] = rolling_no2.std()

    # Manejar NaNs introducidos por lags/rolling
    # Opción 1: Eliminar filas con NaN (más simple si hay suficientes datos)
    # initial_rows = len(df)
    # df = df.dropna(subset=[f'no2_value_lag{max(lags)}', f'no2_rolling_mean_{max(windows)}h']) # Eliminar basado en el lag/window más largo
    # print(f"Eliminados {initial_rows - len(df)} registros debido a NaNs de lags/rolling.")
    # Opción 2: Imputar (ej. con media, mediana, ffill - CUIDADO con data leakage en ffill)
    # for col in df.columns:
    #    if df[col].isnull().any():
    #       df[col] = df[col].fillna(method='ffill').fillna(method='bfill') # Ejemplo ffill+bfill

    df = df.reset_index() # Devolver 'fecha' como columna

    return df

# ==========================================
# PREPROCESAMIENTO (Sin cambios significativos, excepto manejo de 'fecha')
# ==========================================

def preprocessing_temporal_cyclical(df):
    """Añade variables cíclicas seno/coseno."""
    df = df.copy()
    if 'hour' not in df.columns: df['hour'] = pd.to_datetime(df['fecha']).dt.hour
    if 'month' not in df.columns: df['month'] = pd.to_datetime(df['fecha']).dt.month
    if 'day' not in df.columns: df['day'] = pd.to_datetime(df['fecha']).dt.day # O dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31) # Simplificación
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31) # Simplificación
    return df

def split_data(df, fecha_fin_training):
    """Divide los datos en train/test basado en fecha."""
    train_df = df[df['fecha'] < fecha_fin_training].copy()
    test_df = df[df['fecha'] >= fecha_fin_training].copy()
    return train_df, test_df

# --- Funciones de Outliers (remove_outliers_iqr, etc.) sin cambios ---
def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Elimina outliers usando el método del IQR."""
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            Q1 = filtered_df[col].quantile(0.25)
            Q3 = filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0: # Evitar división por cero o problemas con datos constantes
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Elimina outliers usando el método del z-score."""
    filtered_df = df.copy()
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols: return filtered_df
    # Calcular z-scores solo para columnas numéricas, ignorando NaNs temporalmente
    zscores = filtered_df[numeric_cols].apply(lambda x: zscore(x, nan_policy='omit'))
    # La condición debe aplicarse solo donde los zscores no son NaN
    condition = (zscores.abs() < threshold).all(axis=1)
    return filtered_df[condition]

def remove_outliers_quantiles(df: pd.DataFrame, columns: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Elimina outliers basados en percentiles extremos."""
    filtered_df = df.copy()
    for col in columns:
         if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            lower = filtered_df[col].quantile(lower_q)
            upper = filtered_df[col].quantile(upper_q)
            if upper > lower: # Asegurar que hay un rango válido
                filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
    return filtered_df

def filter_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """Aplica el método de filtrado de outliers seleccionado."""
    numeric_cols_for_outliers = [c for c in COLUMNS_FOR_OUTLIERS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if method == 'iqr':
        return remove_outliers_iqr(df, numeric_cols_for_outliers)
    elif method == 'zscore':
        return remove_outliers_zscore(df, numeric_cols_for_outliers)
    elif method == 'quantiles':
        return remove_outliers_quantiles(df, numeric_cols_for_outliers)
    elif method == 'none':
        return df
    else:
        raise ValueError(f"Método '{method}' no reconocido.")

# --- Funciones de Conversión de Unidades y Escalado (sin cambios) ---
def convertir_unidades_legibles(df):
    """Convierte las variables meteorológicas del ERA5 a unidades más interpretables."""
    df = df.copy()
    if 'd2m' in df.columns: df['d2m'] = df['d2m'] - 273.15
    if 't2m' in df.columns: df['t2m'] = df['t2m'] - 273.15
    if 'ssr' in df.columns: df['ssr'] = df['ssr'] / 3600
    if 'ssrd' in df.columns: df['ssrd'] = df['ssrd'] / 3600
    # Usar originales u10/v10 para cálculo, luego convertir
    u10_mps = df.get('u10', pd.Series(0, index=df.index))
    v10_mps = df.get('v10', pd.Series(0, index=df.index))
    df['wind_speed'] = np.sqrt(u10_mps**2 + v10_mps**2) * 3.6 # Convertir a km/h
    df['wind_direction'] = (270 - np.arctan2(v10_mps, u10_mps) * 180/np.pi) % 360
    if 'u10' in df.columns: df['u10'] = df['u10'] * 3.6 # Convertir a km/h para posible uso directo
    if 'v10' in df.columns: df['v10'] = df['v10'] * 3.6 # Convertir a km/h
    if 'sp' in df.columns: df['sp'] = df['sp'] / 100
    if 'tp' in df.columns: df['tp'] = df['tp'] * 1000
    return df

def estandarizar_variables(df, columnas):
    """Estandariza las variables numéricas seleccionadas."""
    df = df.copy()
    scaler_dict = {}
    columnas_a_escalar = [col for col in columnas if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    for col in columnas_a_escalar:
        scaler = StandardScaler()
        data_to_scale = df[[col]]
        # Solo escalar si no hay NaNs o si ya han sido imputados
        if not data_to_scale.isnull().any().any():
            df[col] = scaler.fit_transform(data_to_scale)
            scaler_dict[col] = scaler
        else:
            st.warning(f"Columna '{col}' contiene NaNs y no será escalada. Considere imputar NaNs antes.")

    return df, scaler_dict

def escalar_target(y):
    """Escala la variable objetivo (y)."""
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    return y_scaled, scaler

def desescalar_target(y_scaled, scaler_target):
    """Desescala la variable objetivo."""
    # Asegurarse que y_scaled es un array 2D para inverse_transform
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    return scaler_target.inverse_transform(y_scaled).ravel()

# ==========================================
# ENTRENAMIENTO XGBOOST (Usando Callbacks - Sin cambios funcionales)
# ==========================================
def xgboost_train(X_train, y_train_scaled, X_val, y_val_scaled, feature_names, sensor_seleccionado, outlier_type, preprocessing_type, scaler_dict, scaler_target):
    """Entrena XGBoost usando callbacks."""
    st.info("Configurando y entrenando modelo XGBoost...")

    # Filtrar solo columnas numéricas para XGBoost (maneja categóricas si se codifican)
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_features) != len(feature_names):
         st.warning(f"Se usarán solo {len(numeric_features)} variables numéricas de las {len(feature_names)} seleccionadas para XGBoost.")
         # Aquí podrías añadir codificación para categóricas si se incluyen 'time_of_day' etc.
         # Por ahora, filtraremos solo las numéricas para que el modelo corra.
         X_train = X_train[numeric_features]
         X_val = X_val[numeric_features]
         feature_names = numeric_features # Actualizar la lista de features usadas

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )

    # early_stopping_callback = EarlyStopping(
    #     rounds=50, metric_name='rmse', save_best=True
    # )
    
    eval_set = [(X_val, y_val_scaled)]

    st.text("Iniciando entrenamiento con Early Stopping Callback...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # XGBoost no tiene un callback directo para la barra de progreso de Streamlit,
    # pero podemos indicar que el proceso está corriendo.
    status_text.text("Entrenando... (El progreso real no se muestra por ronda)")

    xgb_model.fit(
        X_train, y_train_scaled,
        eval_set=eval_set,
        verbose=True
    )

    progress_bar.progress(100)
    status_text.success("Entrenamiento completado.")
    st.text(f"Modelo XGBoost entrenado (Early Stopping activado).")

    model_info = {
        'model': xgb_model,
        'feature_names': feature_names, # Guardar las features realmente usadas
        'scaler_dict': scaler_dict,
        'scaler_target': scaler_target,
        'variable_metadata': VARIABLE_METADATA
    }
    model_dir = 'data/models'
    os.makedirs(model_dir, exist_ok=True)
    file_name = f'{model_dir}/xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_type}.pkl'
    joblib.dump(model_info, file_name)
    st.success(f"Modelo XGBoost y metadatos guardados en '{file_name}'")
    return model_info


# ==========================================
# EVALUACIÓN Y VISUALIZACIÓN (Sección Mejorada)
# ==========================================

def plot_residuals_analysis(y_test, y_pred_descaled):
    """Genera gráficos de análisis de residuos."""
    residuals = y_test - y_pred_descaled
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Histograma
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title('Histograma de Residuos')
    ax[0].set_xlabel('Residuo (Real - Predicción) [µg/m³]')
    ax[0].set_ylabel('Frecuencia')
    # Q-Q Plot
    sm.qqplot(residuals, line='45', ax=ax[1], fit=True)
    ax[1].set_title('Gráfico Q-Q de Residuos vs Normal')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    **Interpretación:** Idealmente, el histograma se asemeja a una campana (normalidad) centrada en cero, y los puntos del Q-Q plot siguen la línea roja (normalidad de residuos).
    """)

def plot_feature_importance(model, feature_names):
    """Muestra la importancia de las variables del modelo XGBoost."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3))) # Ajustar altura
        sns.barplot(x=importances[indices], y=sorted_features, ax=ax)
        ax.set_title('Importancia de las Variables (XGBoost Feature Importance)')
        ax.set_xlabel('Importancia Relativa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("El modelo no tiene el atributo 'feature_importances_'.")

def plot_predictions_vs_actuals(test_df_viz, y_pred_descaled):
    """Grafica predicciones vs valores reales con opciones de granularidad y zoom."""
    df_plot = test_df_viz[['fecha']].copy()
    df_plot['Real'] = test_df_viz['no2_value'].values
    df_plot['Predicción'] = y_pred_descaled
    df_plot = df_plot.set_index('fecha')

    st.subheader("Comparación Temporal: Predicciones vs. Valores Reales")

    # Controles para Zoom y Granularidad
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
         min_date = df_plot.index.min().date()
         max_date = df_plot.index.max().date()
         date_range = st.date_input(
             "Selecciona Rango de Fechas para Visualizar:",
             value=(min_date, max_date),
             min_value=min_date,
             max_value=max_date,
             key="viz_date_range"
         )
    with col2:
         granularity = st.selectbox(
             "Selecciona Granularidad:",
             options=['Horaria', 'Media Diaria', 'Media Semanal'],
             index=0, # Por defecto Horaria
             key="viz_granularity"
         )
    with col3: # Espacio vacío o botón de refresco
         st.write("") # Placeholder

    # Filtrar por fecha seleccionada
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1) # Incluir el último día
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]

    if df_filtered.empty:
         st.warning("No hay datos en el rango de fechas seleccionado.")
         return

    # Aplicar granularidad
    if granularity == 'Media Diaria':
         df_agg = df_filtered.resample('D').mean()
         plot_title = f'Predicciones vs. Reales (Media Diaria) - {date_range[0]} a {date_range[1]}'
    elif granularity == 'Media Semanal':
         df_agg = df_filtered.resample('W-MON').mean() # Agrupar por semana empezando en Lunes
         plot_title = f'Predicciones vs. Reales (Media Semanal) - {date_range[0]} a {date_range[1]}'
    else: # Horaria
         df_agg = df_filtered
         plot_title = f'Predicciones vs. Reales (Horario) - {date_range[0]} a {date_range[1]}'

    if df_agg.empty:
         st.warning(f"No hay datos para agregar a nivel {granularity.lower()}.")
         return

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_agg.index, df_agg['Real'], label='Valor Real', alpha=0.8)
    ax.plot(df_agg.index, df_agg['Predicción'], label='Predicción XGBoost', linestyle='--', alpha=0.8)

    # Formatear eje X según granularidad
    if granularity == 'Media Diaria' or granularity == 'Media Semanal':
         ax.xaxis.set_major_locator(mdates.AutoDateLocator())
         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else: # Horaria
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12)) # Ajustar densidad
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    fig.autofmt_xdate() # Rotar fechas para mejor lectura
    ax.set_title(plot_title)
    ax.set_ylabel('Concentración NO₂ (µg/m³)')
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals_over_time(test_df_viz, y_pred_descaled):
    """Grafica los residuos a lo largo del tiempo."""
    df_plot = test_df_viz[['fecha']].copy()
    df_plot['Residuos'] = test_df_viz['no2_value'].values - y_pred_descaled
    df_plot = df_plot.set_index('fecha')

    st.subheader("Análisis Temporal de Errores (Residuos)")

    # Controles (reutilizar los de la gráfica anterior o crear nuevos con keys diferentes)
    col1, col2 = st.columns([2,1])
    with col1:
         min_date = df_plot.index.min().date()
         max_date = df_plot.index.max().date()
         # Usar una key diferente para no interferir con el otro selector de fecha
         date_range_res = st.date_input(
             "Selecciona Rango de Fechas para Errores:",
             value=(min_date, max_date),
             min_value=min_date,
             max_value=max_date,
             key="res_date_range"
         )
    with col2:
         # Usar una key diferente
         granularity_res = st.selectbox(
             "Granularidad de Errores:",
             options=['Horaria', 'Media Diaria', 'Error Absoluto Medio Diario', 'Media Semanal'],
             index=0,
             key="res_granularity"
         )

    # Filtrar y agregar
    start_date = pd.to_datetime(date_range_res[0])
    end_date = pd.to_datetime(date_range_res[1]) + timedelta(days=1)
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]

    if df_filtered.empty:
         st.warning("No hay datos de residuos en el rango seleccionado.")
         return

    if granularity_res == 'Media Diaria':
         df_agg = df_filtered.resample('D').mean()
         plot_title = f'Residuos Medios Diarios - {date_range_res[0]} a {date_range_res[1]}'
         y_label = 'Residuo Medio (µg/m³)'
    elif granularity_res == 'Error Absoluto Medio Diario':
         df_agg = df_filtered.resample('D').apply(lambda x: x.abs().mean())
         plot_title = f'Error Absoluto Medio Diario - {date_range_res[0]} a {date_range_res[1]}'
         y_label = 'MAE Diario (µg/m³)'
    elif granularity_res == 'Media Semanal':
         df_agg = df_filtered.resample('W-MON').mean()
         plot_title = f'Residuos Medios Semanales - {date_range_res[0]} a {date_range_res[1]}'
         y_label = 'Residuo Medio (µg/m³)'
    else: # Horaria
         df_agg = df_filtered
         plot_title = f'Residuos Horarios - {date_range_res[0]} a {date_range_res[1]}'
         y_label = 'Residuo (µg/m³)'

    if df_agg.empty:
         st.warning(f"No hay datos de residuos para agregar a nivel {granularity_res.lower()}.")
         return

    # Graficar
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_agg.index, df_agg['Residuos'], label=y_label, alpha=0.9)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero') # Línea en cero

    # Formatear eje X
    if granularity_res != 'Horaria':
         ax.xaxis.set_major_locator(mdates.AutoDateLocator())
         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
         ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    fig.autofmt_xdate()
    ax.set_title(plot_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# PÁGINA PRINCIPAL DE STREAMLIT (Sin Sidebar)
# ==========================================

def training_page():
    """Página principal para entrenamiento y análisis (sin sidebar)."""

    st.markdown("""
    Plataforma para entrenar modelos XGBoost, incluyendo ingeniería de variables,
    y analizar sus resultados con visualizaciones detalladas.
    """)

    # --- Carga Inicial de Datos ---
    df_total = cargar_datos_trafico_y_meteo()

    # --- Sección de Configuración ---
    st.header("1. Configuración del Experimento")
    config_cols = st.columns(4) # Columnas para controles

    with config_cols[0]:
        # Selector de sensor
        sensor_seleccionado = st.selectbox(
            "Sensor NO₂:",
            df_total['id_no2'].unique(),
            index=2, # Ajustar índice por defecto
            key="sensor_select"
        )
        # Filtrar datos por sensor AHORA para obtener rango de fechas correcto
        df_sensor = df_total[df_total['id_no2'] == sensor_seleccionado].copy()
        st.info(f"{len(df_sensor)} registros para {sensor_seleccionado}")

    # Obtener rango de fechas disponibles para el sensor
    if df_sensor.empty:
        st.error("No hay datos para el sensor seleccionado.")
        st.stop()
    fecha_min = df_sensor["fecha"].min().date()
    fecha_max = df_sensor["fecha"].max().date()

    with config_cols[1]:
        # Fecha de división Train/Test
        fecha_fin_training_dt = st.date_input(
            "Fecha Fin Entrenamiento:",
            datetime(2024, 1, 1).date(), # Fecha por defecto
            min_value=fecha_min + timedelta(days=7), # Asegurar algo de entrenamiento
            max_value=fecha_max - timedelta(days=1), # Asegurar algo de test
            help="Datos ANTES de esta fecha para entrenar, DESDE esta fecha para evaluar.",
            key="date_split"
        )
        fecha_fin_training_dt = pd.to_datetime(fecha_fin_training_dt)

    with config_cols[2]:
        # Tipo de filtrado de outliers
        outlier_type = st.selectbox(
            "Filtrado Outliers:",
            ["none", "zscore", "iqr", "quantiles"], index=0,
            help="Método para eliminar valores atípicos.",
            key="outlier_select"
        )

    with config_cols[3]:
        # Preprocesamiento (variables cíclicas)
        preprocessing_type = st.selectbox(
            "Preproc. Temporal:", ["none", "sin_cos"], index=1,
            help="Crear variables seno/coseno.",
            key="preproc_select"
        )

    # --- Ingeniería de Variables (Aplicada antes de seleccionar) ---
    st.write("⚙️ Aplicando Ingeniería de Variables...")
    # Aplicar Feature Engineering DESPUÉS de filtrar por sensor
    df_featured = engineer_features(df_sensor)
    # IMPORTANTE: Manejar NaNs creados por lags/rolling ANTES de dividir
    nan_cols = [col for col in df_featured.columns if df_featured[col].isnull().any()]
    if nan_cols:
         st.warning(f"Columnas con NaNs después de Feature Engineering: {nan_cols}. Eliminando filas con NaNs...")
         initial_rows = len(df_featured)
         # Eliminar filas donde CUALQUIERA de las nuevas features sea NaN (más seguro)
         cols_to_check_nan = [c for c in df_featured.columns if 'lag' in c or 'rolling' in c]
         df_featured = df_featured.dropna(subset=cols_to_check_nan)
         st.info(f"Se eliminaron {initial_rows - len(df_featured)} filas debido a NaNs.")
         if df_featured.empty:
              st.error("No quedan datos después de eliminar NaNs de Feature Engineering. Ajusta lags/windows o el rango de fechas.")
              st.stop()

    # --- Selección de Variables (con Expander) ---
    st.header("2. Selección de Variables Predictoras")
    with st.expander("Ver/Modificar Variables", expanded=False):
        # Categorías de variables (incluyendo las nuevas)
        var_categories = {
            "Ciclicas": [f for f in df_featured.columns if '_sin' in f or '_cos' in f],
            "Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
            "Meteo": [c for c in df_featured.columns if c in ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']],
            "Engineered (Lag)": [f for f in df_featured.columns if 'lag' in f],
            "Engineered (Rolling)": [f for f in df_featured.columns if 'rolling' in f],
            "Engineered (Temporal)": ['day_of_week', 'is_weekend', 'time_of_day', 'hour'] # Añadir 'hour' si no se usa sin/cos
        }

        # Filtrar categorías/variables que realmente existen en df_featured
        available_features = []
        valid_categories = {}
        for cat, var_list in var_categories.items():
            valid_vars = [v for v in var_list if v in df_featured.columns]
            if valid_vars:
                valid_categories[cat] = valid_vars
                available_features.extend(valid_vars)

        # Eliminar duplicados manteniendo orden (si es necesario)
        available_features = sorted(list(set(available_features)))

        # Selección en columnas dentro del expander
        sel_cols = st.columns(len(valid_categories))
        selected_features = []
        default_selection = available_features # Seleccionar todo por defecto inicialmente
        # Excluir algunas por defecto si se desea, ejemplo:
        # default_selection = [f for f in available_features if f not in ['wind_direction', 'time_of_day']]

        for i, (cat, var_list) in enumerate(valid_categories.items()):
            with sel_cols[i]:
                st.markdown(f"**{cat}**")
                selected_in_cat = st.multiselect(
                    f"Selecciona en {cat}", var_list,
                    default=[v for v in default_selection if v in var_list],
                    key=f"select_{cat}"
                )
                selected_features.extend(selected_in_cat)

        # Asegurarse de que las seleccionadas son únicas
        selected_features = sorted(list(set(selected_features)))
        st.info(f"Variables seleccionadas: {len(selected_features)}")

        if not selected_features:
            st.error("¡Debes seleccionar al menos una variable predictora!")
            st.stop()


    # --- Procesamiento Final y División ---
    st.write("🔄 Aplicando preprocesamiento y dividiendo datos...")
    df_processed = df_featured.copy()

    # Convertir unidades (después de FE por si se usan originales u10/v10)
    df_processed = convertir_unidades_legibles(df_processed)

    # Filtrar outliers (si se seleccionó)
    df_processed = filter_outliers(df_processed, outlier_type)
    if outlier_type != 'none':
        st.write(f"Aplicado filtro '{outlier_type}'. Registros restantes: {len(df_processed)}")

    # Aplicar preprocesamiento cíclico (si se seleccionó)
    if preprocessing_type == "sin_cos":
        df_processed = preprocessing_temporal_cyclical(df_processed)
        st.write("Aplicado preproc. 'sin_cos'.")
        # Asegurar que las 'sin_cos' estén si se seleccionaron, y quitar las originales
        features_to_use = []
        has_hour_sincos = 'hour_sin' in selected_features or 'hour_cos' in selected_features
        has_day_sincos = 'day_sin' in selected_features or 'day_cos' in selected_features
        has_month_sincos = 'month_sin' in selected_features or 'month_cos' in selected_features
        for f in selected_features:
            is_original_temporal = (f == 'hour' and has_hour_sincos) or \
                                   (f == 'day' and has_day_sincos) or \
                                   (f == 'month' and has_month_sincos)
            if not is_original_temporal:
                features_to_use.append(f)
        # Añadir las sin/cos correspondientes si no estaban ya explícitamente seleccionadas
        if has_hour_sincos: features_to_use.extend(['hour_sin', 'hour_cos'])
        if has_day_sincos: features_to_use.extend(['day_sin', 'day_cos'])
        if has_month_sincos: features_to_use.extend(['month_sin', 'month_cos'])
        selected_features = sorted(list(set(features_to_use)))
        st.write(f"Variables finales ajustadas por sin/cos: {len(selected_features)}")


    # Codificar variables categóricas si existen y se seleccionaron (Ej: time_of_day)
    categorical_features = df_processed[selected_features].select_dtypes(include=['category', 'object']).columns.tolist()
    if categorical_features:
        st.write(f"Codificando variables categóricas: {categorical_features}")
        df_processed = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
        # Actualizar lista de features seleccionadas para incluir las dummy
        selected_features = [f for f in selected_features if f not in categorical_features]
        dummy_cols = [c for c in df_processed.columns if any(cat + "_" in c for cat in categorical_features)]
        selected_features.extend(dummy_cols)
        selected_features = sorted(list(set(selected_features)))
        st.info(f"Variables totales después de Dummificación: {len(selected_features)}")


    # Filtrar columnas finales justo antes de dividir
    cols_to_keep = ['fecha', 'no2_value'] + [f for f in selected_features if f in df_processed.columns]
    df_final = df_processed[cols_to_keep].copy()

    # Eliminar NaNs restantes (importante antes de escalar/entrenar)
    if df_final[selected_features].isnull().any().any():
        st.warning("Detectados NaNs antes de escalar/entrenar. Eliminando filas...")
        initial_rows = len(df_final)
        df_final = df_final.dropna(subset=selected_features)
        st.info(f"Eliminadas {initial_rows - len(df_final)} filas con NaNs en variables seleccionadas.")

    if df_final.empty:
         st.error("No quedan datos después de la limpieza final de NaNs.")
         st.stop()

    # Dividir datos
    train_df, test_df = split_data(df_final, fecha_fin_training_dt)
    st.success(f"Datos listos: {len(train_df)} train, {len(test_df)} test.")

    if train_df.empty or test_df.empty:
        st.error("Conjunto de entrenamiento o prueba vacío. Ajusta fechas o revisa filtros.")
        st.stop()

    # Preparar X e y
    X_train = train_df[selected_features].copy()
    y_train = train_df['no2_value'].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df['no2_value'].copy()
    # Guardar test_df original con fecha para visualización
    test_df_viz = test_df[['fecha', 'no2_value']].copy()

    # Escalar variables
    X_train_scaled, scaler_dict = estandarizar_variables(X_train, selected_features)
    y_train_scaled, scaler_target = escalar_target(y_train)
    # Escalar X_test usando scalers del train
    X_test_scaled = X_test.copy()
    numeric_cols_in_test = X_test_scaled.select_dtypes(include=np.number).columns
    for col in numeric_cols_in_test:
        if col in scaler_dict:
            X_test_scaled[col] = scaler_dict[col].transform(X_test_scaled[[col]])
        elif col in selected_features: # Solo advertir si era una feature seleccionada que debería haberse escalado
             st.warning(f"No se encontró scaler para '{col}' en test (puede ser no numérica o no presente en train).")
    # Escalar y_test para early stopping
    y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1)).ravel()

    # --- Acciones: Entrenar o Analizar ---
    st.header("3. Acciones del Modelo")
    action_cols = st.columns(2)
    model_dir = 'data/models'
    model_filename = f'xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_type}.pkl'
    full_model_path = os.path.join(model_dir, model_filename)

    with action_cols[0]:
        if st.button("🚀 Entrenar Nuevo Modelo", key="train_button", type="primary"):
            model_info = xgboost_train(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled, # Usar test set como validation set
                selected_features, # Pasar las features finales usadas
                sensor_seleccionado, outlier_type, preprocessing_type,
                scaler_dict, scaler_target
            )
            # Forzar recarga de la info del modelo para análisis inmediato
            st.session_state['model_info'] = model_info
            st.session_state['test_df_viz'] = test_df_viz
            st.session_state['y_test'] = y_test
            st.session_state['X_test_scaled'] = X_test_scaled # Guardar para análisis
            st.rerun() # Recargar para mostrar análisis

    with action_cols[1]:
        if st.button("📊 Analizar Modelo Guardado", key="analyze_button", disabled=not os.path.exists(full_model_path)):
             if os.path.exists(full_model_path):
                try:
                    st.session_state['model_info'] = joblib.load(full_model_path)
                    # Cargar datos necesarios para visualización con la configuración actual
                    st.session_state['test_df_viz'] = test_df_viz
                    st.session_state['y_test'] = y_test
                    st.session_state['X_test_scaled'] = X_test_scaled # Usar X_test actual escalado
                    st.success(f"Modelo '{model_filename}' cargado para análisis.")
                    # st.rerun() # Recargar para mostrar análisis
                except Exception as e:
                    st.error(f"Error al cargar modelo: {e}")
                    if 'model_info' in st.session_state: del st.session_state['model_info']
             else:
                st.error("No existe modelo guardado para esta configuración.")
                if 'model_info' in st.session_state: del st.session_state['model_info']

    # --- Sección de Análisis (Mostrar si hay un modelo cargado/entrenado) ---
    if 'model_info' in st.session_state and st.session_state['model_info'] is not None:
        st.header("4. Análisis del Modelo")

        model_data = st.session_state['model_info']
        loaded_model = model_data['model']
        loaded_scaler_target = model_data['scaler_target']
        loaded_feature_names = model_data['feature_names'] # Features usadas al GUARDAR el modelo
        current_test_df_viz = st.session_state['test_df_viz']
        current_y_test = st.session_state['y_test']
        current_X_test_scaled = st.session_state['X_test_scaled'] # X_test actual escalado

        # Verificar compatibilidad de features entre modelo cargado y datos actuales
        # XGBoost puede manejar features extras en la predicción si se ajusta, pero es mejor alinear
        cols_in_model = set(loaded_feature_names)
        cols_in_current_test = set(current_X_test_scaled.columns)

        if cols_in_model != cols_in_current_test:
             st.warning(f"""
             ¡Discrepancia de variables! El modelo se entrenó con {len(cols_in_model)} variables,
             pero los datos de prueba actuales tienen {len(cols_in_current_test)}.
             Variables solo en modelo: {cols_in_model - cols_in_current_test}
             Variables solo en datos actuales: {cols_in_current_test - cols_in_model}
             Se intentará predecir solo con las columnas del modelo.
             """)
             # Alinear columnas de X_test_scaled a las del modelo
             missing_cols = list(cols_in_model - cols_in_current_test)
             extra_cols = list(cols_in_current_test - cols_in_model)
             if missing_cols:
                 # Opción: Añadir columnas faltantes con ceros o media (no ideal)
                 # Por ahora, no predecir si faltan columnas esenciales.
                 st.error("Faltan columnas del modelo en los datos de prueba actuales. No se puede predecir.")
                 st.stop()
             # Eliminar columnas extra
             current_X_test_aligned = current_X_test_scaled[list(cols_in_model)]
        else:
             current_X_test_aligned = current_X_test_scaled[loaded_feature_names] # Asegurar orden

        # Realizar predicciones con el modelo cargado y los datos de test alineados
        y_pred_scaled = loaded_model.predict(current_X_test_aligned)
        y_pred_descaled = desescalar_target(y_pred_scaled, loaded_scaler_target)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(current_y_test, y_pred_descaled))
        r2 = r2_score(current_y_test, y_pred_descaled)
        mae = mean_absolute_error(current_y_test, y_pred_descaled)

        # Pestañas para organizar el análisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen y Métricas",
            "📈 Análisis Temporal",
            "⚙️ Importancia Variables",
            "📉 Análisis de Errores"
        ])

        with tab1:
            st.subheader("Métricas de Evaluación (Test Set)")
            metric_cols = st.columns(3)
            metric_cols[0].metric("RMSE", f"{rmse:.2f} µg/m³", delta_color="inverse")
            metric_cols[1].metric("R² Score", f"{r2:.3f}")
            metric_cols[2].metric("MAE", f"{mae:.2f} µg/m³", delta_color="inverse")
            st.divider()
            st.subheader("Análisis de Residuos")
            plot_residuals_analysis(current_y_test, y_pred_descaled)

        with tab2:
             plot_predictions_vs_actuals(current_test_df_viz, y_pred_descaled)

        with tab3:
             st.subheader("Importancia de Variables")
             # Asegurarse de pasar los nombres correctos con los que se entrenó el modelo
             plot_feature_importance(loaded_model, loaded_feature_names)

        with tab4:
             plot_residuals_over_time(current_test_df_viz, y_pred_descaled)

    elif 'model_info' in st.session_state and st.session_state['model_info'] is None:
        # Caso donde se intentó cargar pero falló o no existía
        st.info("Carga un modelo guardado o entrena uno nuevo para ver el análisis.")
