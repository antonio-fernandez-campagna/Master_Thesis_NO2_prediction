# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import seaborn as sns
# import altair as alt # No se usa directamente en este script adaptado
# import calplot # No se usa directamente en este script adaptado
import joblib
import os
import xgboost as xgb # <--- Import XGBoost
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore # Necesario para zscore outlier removal

# ==========================================
# CARGA Y PREPROCESAMIENTO DE DATOS (Sin cambios respecto al original)
# ==========================================

@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """
    Carga y preprocesa los datos con caché de Streamlit.

    Returns:
        DataFrame: Datos de NO2, tráfico y meteorología con fecha convertida a datetime.
    """
    # Asegúrate de que la ruta al archivo es correcta
    file_path = 'data/more_processed/no2_with_traffic_and_meteo_one_station.parquet'
    if not os.path.exists(file_path):
        st.error(f"Error: No se encontró el archivo de datos en {file_path}")
        st.stop()
    df = pd.read_parquet(file_path)
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
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31) # Simplificación, idealmente sería día del año
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31) # Simplificación, idealmente sería día del año
    return df

def split_data(df, fecha_fin_training):
    """
    Divide los datos en conjuntos de entrenamiento y prueba según fechas.

    Args:
        df (DataFrame): DataFrame completo.
        fecha_fin_training (datetime): Fecha límite para entrenamiento (inclusive).

    Returns:
        tuple: (train_df, test_df) DataFrames de entrenamiento y prueba.
    """
    train_df = df[df['fecha'] < fecha_fin_training].copy()
    test_df = df[df['fecha'] >= fecha_fin_training].copy()
    return train_df, test_df

COLUMNS_FOR_OUTLIERS = [
    'no2_value', 'intensidad', 'carga', 'ocupacion', 'vmed',
    'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp'
]

def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Elimina outliers usando el método del IQR."""
    filtered_df = df.copy()
    for col in columns:
        if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            Q1 = filtered_df[col].quantile(0.25)
            Q3 = filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Elimina outliers usando el método del z-score."""
    filtered_df = df.copy()
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        return filtered_df # No hay columnas numéricas para aplicar z-score

    zscores = filtered_df[numeric_cols].apply(zscore)
    condition = (zscores.abs() < threshold).all(axis=1)
    return filtered_df[condition]

def remove_outliers_quantiles(df: pd.DataFrame, columns: list[str], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Elimina outliers basados en percentiles extremos."""
    filtered_df = df.copy()
    for col in columns:
         if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            lower = filtered_df[col].quantile(lower_q)
            upper = filtered_df[col].quantile(upper_q)
            filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
    return filtered_df

def filter_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """Aplica el método de filtrado de outliers seleccionado."""
    if method == 'iqr':
        return remove_outliers_iqr(df, COLUMNS_FOR_OUTLIERS)
    elif method == 'zscore':
        return remove_outliers_zscore(df, COLUMNS_FOR_OUTLIERS)
    elif method == 'quantiles':
        return remove_outliers_quantiles(df, COLUMNS_FOR_OUTLIERS)
    elif method == 'none':
        return df
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa: 'iqr', 'zscore', 'quantiles' o 'none'.")

def convertir_unidades_legibles(df):
    """Convierte las variables meteorológicas del ERA5 a unidades más interpretables."""
    df = df.copy()
    if 'd2m' in df.columns: df['d2m'] = df['d2m'] - 273.15
    if 't2m' in df.columns: df['t2m'] = df['t2m'] - 273.15
    if 'ssr' in df.columns: df['ssr'] = df['ssr'] / 3600
    if 'ssrd' in df.columns: df['ssrd'] = df['ssrd'] / 3600
    if 'u10' in df.columns: df['u10_kmh'] = df['u10'] * 3.6 # Renombrar para evitar confusión
    if 'v10' in df.columns: df['v10_kmh'] = df['v10'] * 3.6 # Renombrar
    if 'u10_kmh' in df.columns and 'v10_kmh' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10_kmh']**2 + df['v10_kmh']**2)
        df['wind_direction'] = (270 - np.arctan2(df['v10_kmh'], df['u10_kmh']) * 180/np.pi) % 360
    if 'sp' in df.columns: df['sp'] = df['sp'] / 100
    if 'tp' in df.columns: df['tp'] = df['tp'] * 1000
    return df

VARIABLE_METADATA = {
    'd2m': {'name': 'Punto de Rocío', 'unit': '°C', 'typical_range': (-10, 30)},
    't2m': {'name': 'Temperatura', 'unit': '°C', 'typical_range': (-5, 40)},
    'ssr': {'name': 'Radiación Solar Neta', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'ssrd': {'name': 'Radiación Solar Descendente', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'wind_speed': {'name': 'Velocidad del Viento', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_direction': {'name': 'Dirección del Viento', 'unit': '°', 'typical_range': (0, 360)},
    'sp': {'name': 'Presión Superficial', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'tp': {'name': 'Precipitación Total', 'unit': 'mm', 'typical_range': (0, 50)},
    'intensidad': {'name': 'Intensidad Tráfico', 'unit': 'veh/h', 'typical_range': (0, 3000)}, # Ejemplo
    'carga': {'name': 'Carga Tráfico', 'unit': '%', 'typical_range': (0, 100)}, # Ejemplo
    'ocupacion': {'name': 'Ocupación Tráfico', 'unit': '%', 'typical_range': (0, 100)}, # Ejemplo
    'vmed': {'name': 'Velocidad Media Tráfico', 'unit': 'km/h', 'typical_range': (0, 120)} # Ejemplo
}

def estandarizar_variables(df, columnas):
    """Estandariza las variables seleccionadas y guarda los parámetros de escalado."""
    df = df.copy()
    scaler_dict = {}
    for col in columnas:
        if col in df.columns:
            scaler = StandardScaler()
            # Asegurarse que la columna es numérica y no tiene NaNs antes de escalar
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().any():
                 df[col] = scaler.fit_transform(df[[col]])
                 scaler_dict[col] = scaler
            else:
                st.warning(f"Columna '{col}' no es numérica o contiene NaNs. No se escalará.")
                # Podrías optar por rellenar NaNs aquí si es apropiado
                # df[col] = df[col].fillna(df[col].median()) # Ejemplo: imputar con mediana
                # df[col] = scaler.fit_transform(df[[col]])
                # scaler_dict[col] = scaler
    return df, scaler_dict

def desescalar_variables(df, scaler_dict):
    """Desescala las variables usando los scalers guardados."""
    df = df.copy()
    for col, scaler in scaler_dict.items():
         if col in df.columns:
            df[col] = scaler.inverse_transform(df[[col]])
    return df

def escalar_target(y):
    """Escala la variable objetivo y devuelve el scaler."""
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    return y_scaled, scaler

def desescalar_target(y_scaled, scaler_target):
    """Desescala la variable objetivo."""
    return scaler_target.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ==========================================
# ENTRENAMIENTO DEL MODELO XGBOOST
# ==========================================

def xgboost_train(X_train, y_train_scaled, X_val, y_val_scaled, feature_names, sensor_seleccionado, outlier_type, preprocessing_box, scaler_dict, scaler_target):
    """
    Entrena un modelo XGBoost con variables escaladas y validación temprana.

    Args:
        X_train: DataFrame de entrenamiento (features escaladas).
        y_train_scaled: Array de target de entrenamiento (escalado).
        X_val: DataFrame de validación (features escaladas).
        y_val_scaled: Array de target de validación (escalado).
        feature_names: Lista de nombres de las features.
        sensor_seleccionado: ID del sensor.
        outlier_type: Método de eliminación de outliers usado.
        preprocessing_box: Tipo de preprocesamiento temporal usado.
        scaler_dict: Diccionario con los scalers de las features.
        scaler_target: Scaler del target.

    Returns:
        dict: Información del modelo entrenado y metadatos.
    """
    print("Configurando y entrenando modelo XGBoost...")

    # Configuración del modelo XGBoost (CORREGIDO)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', # Para regresión
        n_estimators=1000,             # Número máximo de árboles (se ajustará con early stopping)
        learning_rate=0.05,            # Tasa de aprendizaje
        max_depth=7,                   # Profundidad máxima de los árboles
        subsample=0.8,                 # Fracción de muestras para entrenar cada árbol
        colsample_bytree=0.8,          # Fracción de features para entrenar cada árbol
        random_state=42,               # Para reproducibilidad
        n_jobs=-1,                     # Usar todos los cores disponibles
        eval_metric='rmse'             # <--- Métrica de evaluación definida aquí
    )

    # Entrenar el modelo con early stopping
    print("Iniciando entrenamiento...")
    # Usamos los datos de validación (test en este caso) para el early stopping
    eval_set = [(X_val, y_val_scaled)] # Solo necesitamos el de validación para early stopping
                                        # aunque pasar ambos [(X_train, y_train_scaled), (X_val, y_val_scaled)] también funciona
                                        # y permite monitorizar ambos.

    xgb_model.fit(
        X_train, y_train_scaled,
        eval_set=eval_set,
        # eval_metric ya NO va aquí
        verbose=False                # Poner True para ver el progreso del entrenamiento y la métrica
    )
    # print(f"Modelo XGBoost entrenado correctamente. Número óptimo de árboles: {xgb_model.best_iteration}")

    # Guardar modelo y metadatos asociados
    model_info = {
        'model': xgb_model,
        'feature_names': feature_names,
        'scaler_dict': scaler_dict,
        'scaler_target': scaler_target,
        'variable_metadata': VARIABLE_METADATA # Guardar metadatos para referencia
    }

    # Crear directorio si no existe
    model_dir = 'data/models'
    os.makedirs(model_dir, exist_ok=True)

    # Guardar el modelo
    file_name = f'{model_dir}/xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_box}.pkl'
    joblib.dump(model_info, file_name)
    print(f"Modelo XGBoost y metadatos guardados en {file_name}")

    return model_info


# ==========================================
# EVALUACIÓN DEL MODELO XGBOOST
# ==========================================

def show_xgboost_model_stats(model, X_test, y_test, feature_names, scaler_target):
    """
    Muestra las estadísticas del modelo XGBoost y visualizaciones básicas.

    Args:
        model: Modelo XGBoost entrenado.
        X_test (DataFrame): Variables predictoras para prueba (escaladas).
        y_test (Series): Variable objetivo para prueba (original, sin escalar).
        feature_names (list): Lista de nombres de características.
        scaler_target: Scaler para desescalar la variable objetivo.
    """
    st.header("Resultados del Modelo XGBoost")

    # Realizar predicciones con datos de test escalados
    y_pred_scaled = model.predict(X_test)

    # Desescalar las predicciones para compararlas con y_test original
    y_pred_descaled = desescalar_target(y_pred_scaled, scaler_target)

    # Calcular métricas de evaluación
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_descaled))
    r2 = r2_score(y_test, y_pred_descaled)
    mae = mean_absolute_error(y_test, y_pred_descaled)

    # Mostrar panel de métricas
    st.subheader("Métricas de Evaluación en Conjunto de Prueba")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f} µg/m³", delta_color="inverse", help="Raíz del Error Cuadrático Medio. Menor es mejor.")
    col2.metric("R² Score", f"{r2:.3f}", help="Coeficiente de Determinación. Más cercano a 1 es mejor.")
    col3.metric("MAE", f"{mae:.2f} µg/m³", delta_color="inverse", help="Error Absoluto Medio. Menor es mejor.")

    # Mostrar distribución de residuos
    st.subheader("Análisis de Residuos")
    residuals = y_test - y_pred_descaled
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma de residuos
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set_title('Histograma de Residuos')
    ax[0].set_xlabel('Residuo (Valor Real - Predicción) [µg/m³]')
    ax[0].set_ylabel('Frecuencia')

    # Gráfico Q-Q de residuos vs distribución normal teórica
    sm.qqplot(residuals, line='45', ax=ax[1], fit=True)
    ax[1].set_title('Gráfico Q-Q de Residuos')
    ax[1].set_xlabel('Cuantiles Teóricos (Normal)')
    ax[1].set_ylabel('Cuantiles de los Residuos')

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretación de Residuos:**
    - El **Histograma** idealmente debería parecerse a una campana de Gauss centrada en cero, indicando que los errores son aleatorios y no sistemáticos.
    - El **Gráfico Q-Q** compara los cuantiles de los residuos con los de una distribución normal. Si los puntos se alinean aproximadamente con la línea roja diagonal, sugiere que los residuos siguen una distribución normal, lo cual es una buena señal. Desviaciones pueden indicar problemas como heterocedasticidad o no linealidades no capturadas.
    """)

    # Información sobre interpretabilidad (SHAP)
    st.subheader("Interpretabilidad del Modelo (Avanzado)")
    st.info("""
    Para entender cómo cada variable influye en las predicciones de un modelo XGBoost, se suelen utilizar técnicas como **SHAP (SHapley Additive exPlanations)**.
    Implementar gráficos SHAP (como dependence plots o summary plots) proporcionaría información similar a las dependencias parciales de los modelos GAM.
    Esto requiere instalar la librería `shap` (`pip install shap`) y añadir código específico para calcular y visualizar los valores SHAP.
    """)


# ==========================================
# INTERFAZ DE USUARIO DE STREAMLIT (Adaptada para XGBoost)
# ==========================================

def training_page():
    """Página principal para entrenamiento y análisis del modelo XGBoost."""

    st.title("Entrenamiento y Análisis de Modelo XGBoost para NO₂")
    st.markdown("""
    Esta aplicación permite entrenar un modelo XGBoost o analizar uno existente
    para predecir los niveles de NO₂ basándose en datos de tráfico y meteorología.
    """)

    # Cargar datos
    df_total = cargar_datos_trafico_y_meteo()

    # --- Columnas de Configuración ---
    st.sidebar.header("Configuración del Entrenamiento/Análisis")

    # Selector de sensor
    sensor_seleccionado = st.sidebar.selectbox(
        "Selecciona un sensor de NO₂:",
        df_total['id_no2'].unique(),
        index=2 # Puedes ajustar el índice por defecto
    )

    # Filtrar por sensor seleccionado
    df_sensor = df_total[df_total['id_no2'] == sensor_seleccionado].copy()
    st.info(f"Datos cargados para el sensor: {sensor_seleccionado}. Total de registros: {len(df_sensor)}")

    # Obtener rango de fechas disponibles para el sensor
    if df_sensor.empty:
        st.error("No hay datos para el sensor seleccionado.")
        st.stop()
    fecha_min = df_sensor["fecha"].min().date()
    fecha_max = df_sensor["fecha"].max().date()

    # Selectores de fechas para división de datos
    fecha_fin_training_dt = st.sidebar.date_input(
        "Fecha Fin Entrenamiento (Inicio Evaluación):",
        datetime(2024, 1, 1).date(), # Fecha por defecto
        min_value=fecha_min + timedelta(days=1), # Asegurar al menos un día de entrenamiento
        max_value=fecha_max,
        help="Los datos ANTERIORES a esta fecha se usarán para entrenar, los datos EN o DESPUÉS de esta fecha para evaluar."
    )
    fecha_fin_training_dt = pd.to_datetime(fecha_fin_training_dt) # Convertir a datetime

    # Tipo de filtrado de outliers
    outlier_type = st.sidebar.selectbox(
        "Filtrado de Outliers:",
        ["none", "zscore", "iqr", "quantiles"],
        index=0, # Por defecto 'none'
        help="Método para eliminar valores atípicos antes de entrenar."
    )

    # Preprocesamiento (variables cíclicas)
    preprocessing_box = st.sidebar.selectbox(
        "Preprocesamiento Temporal:",
        ["none", "sin_cos"],
        index=1, # Por defecto 'sin_cos'
        help="Crear variables seno/coseno para hora, mes y día."
    )

    # --- Selección de Variables ---
    st.sidebar.header("Selección de Variables Predictoras")

    var_categories = {
        "Ciclicas": ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos'],
        "Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
        "Meteo Original": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp'],
        "Meteo Convertida": ['wind_speed', 'wind_direction'] # Añadidas por convertir_unidades_legibles
    }

    # Filtrar variables disponibles según preprocesamiento
    available_features = []
    if preprocessing_box == "sin_cos":
        available_features.extend(var_categories["Ciclicas"])
    else: # Si no hay sin/cos, ofrecer las originales numéricas
         available_features.extend(['hour', 'month', 'day']) # Asumiendo que existen

    available_features.extend(var_categories["Tráfico"])
    available_features.extend(var_categories["Meteo Original"])
    available_features.extend(var_categories["Meteo Convertida"])

    # Eliminar duplicados y asegurarse que existen en el df (después de preprocesamiento)
    processed_df_temp = df_sensor.copy() # Copia temporal para verificar columnas
    if preprocessing_box == "sin_cos":
        processed_df_temp = preprocessing(processed_df_temp)
    processed_df_temp = convertir_unidades_legibles(processed_df_temp)

    # Mantener solo las columnas que realmente existen en el DF procesado
    final_available_features = [f for f in available_features if f in processed_df_temp.columns]

    # Multiselect en la barra lateral
    selected_features = st.sidebar.multiselect(
         "Selecciona las variables:",
         options=final_available_features,
         default=[f for f in final_available_features if f not in ['wind_direction']] # Excluir dirección por defecto si es categórica/circular
    )

    if not selected_features:
        st.warning("Por favor, selecciona al menos una variable predictora.")
        st.stop()

    # --- Procesamiento de Datos Post-Configuración ---
    # 1. Filtrar outliers (si se seleccionó)
    df_processed = filter_outliers(df_sensor, outlier_type)
    if outlier_type != 'none':
        st.write(f"Aplicado filtro de outliers '{outlier_type}'. Registros restantes: {len(df_processed)}")

    # 2. Aplicar preprocesamiento temporal (si se seleccionó)
    if preprocessing_box == "sin_cos":
        df_processed = preprocessing(df_processed)
        st.write("Aplicado preprocesamiento 'sin_cos' para variables temporales.")

    # 3. Convertir unidades meteorológicas (siempre)
    df_processed = convertir_unidades_legibles(df_processed)
    st.write("Convertidas unidades meteorológicas a formatos legibles.")

    # 4. Comprobar NaNs antes de dividir
    if df_processed[selected_features + ['no2_value']].isnull().any().any():
        st.warning("Se detectaron valores NaN después del preprocesamiento. Rellenando con la media...")
        for col in selected_features + ['no2_value']:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median()) # O usar .mean()

    # 5. Dividir datos en entrenamiento y prueba
    train_df, test_df = split_data(df_processed, fecha_fin_training_dt)
    st.write(f"Datos divididos: {len(train_df)} para entrenamiento, {len(test_df)} para prueba.")

    if train_df.empty or test_df.empty:
        st.error("Error: El conjunto de entrenamiento o prueba está vacío. Ajusta las fechas o verifica los datos.")
        st.stop()

    # 6. Preparar X e y
    X_train = train_df[selected_features].copy()
    y_train = train_df['no2_value'].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df['no2_value'].copy()

    # 7. Escalar variables (Features y Target)
    X_train_scaled, scaler_dict = estandarizar_variables(X_train, selected_features)
    y_train_scaled, scaler_target = escalar_target(y_train)

    # Escalar X_test usando los scalers del entrenamiento
    X_test_scaled = X_test.copy()
    for col in selected_features:
        if col in scaler_dict:
            X_test_scaled[col] = scaler_dict[col].transform(X_test_scaled[[col]])
        else:
             st.warning(f"No se encontró scaler para la columna '{col}' en el conjunto de prueba. Puede que no estuviera en el entrenamiento o no fuera escalable.")
             # Decide cómo manejar esto: eliminar columna, imputar, etc. Por ahora se deja como está.

    # Escalar y_test solo para usarlo en early stopping si es necesario
    y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1)).ravel()

    st.success("Preprocesamiento y preparación de datos completados.")

    # --- Acciones: Entrenar o Analizar ---
    st.header("Acciones")

    model_dir = 'data/models'
    model_filename = f'xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_box}.pkl'
    full_model_path = os.path.join(model_dir, model_filename)

    col_action1, col_action2 = st.columns(2)

    # Botón para Entrenar
    with col_action1:
        if st.button("🚀 Entrenar Nuevo Modelo XGBoost", key="train_button"):
            if not selected_features:
                 st.error("Debes seleccionar al menos una variable predictora para entrenar.")
            else:
                with st.spinner(f"Entrenando modelo XGBoost... Esto puede tardar unos minutos. Usando {len(selected_features)} variables."):
                    # Asegurarse de que el directorio de modelos existe
                    os.makedirs(model_dir, exist_ok=True)

                    # Llamar a la función de entrenamiento
                    # Pasamos X_test_scaled y y_test_scaled como conjunto de validación para early stopping
                    model_info = xgboost_train(
                        X_train_scaled, y_train_scaled,
                        X_test_scaled, y_test_scaled, # Usar test set como validation set aquí
                        selected_features,
                        sensor_seleccionado,
                        outlier_type,
                        preprocessing_box,
                        scaler_dict,
                        scaler_target
                    )
                st.success(f"¡Modelo XGBoost entrenado y guardado en '{full_model_path}'!")
                # Mostrar resultados del modelo recién entrenado
                show_xgboost_model_stats(model_info['model'], X_test_scaled, y_test, selected_features, scaler_target)


    # Botón para Analizar Modelo Existente
    with col_action2:
        if st.button("📊 Analizar Modelo Guardado", key="analyze_button", disabled=not os.path.exists(full_model_path)):
             if os.path.exists(full_model_path):
                with st.spinner(f"Cargando y analizando modelo desde '{full_model_path}'..."):
                    try:
                        model_info = joblib.load(full_model_path)
                        # Verificar compatibilidad de features
                        if set(model_info['feature_names']) != set(selected_features):
                             st.warning("Las variables seleccionadas ahora son diferentes a las usadas cuando se guardó el modelo. Los resultados podrían no ser directamente comparables.")
                        # Evaluar el modelo cargado con los datos de prueba actuales (escalados)
                        show_xgboost_model_stats(model_info['model'], X_test_scaled, y_test, model_info['feature_names'], model_info['scaler_target'])
                    except Exception as e:
                        st.error(f"Error al cargar o analizar el modelo: {e}")
             else:
                st.error(f"No se encontró un modelo guardado en '{full_model_path}'. Por favor, entrena un modelo primero.")

    if not os.path.exists(full_model_path):
        st.info(f"No existe un modelo guardado para la configuración actual ({model_filename}). Puedes entrenar uno nuevo usando el botón 'Entrenar'.")

