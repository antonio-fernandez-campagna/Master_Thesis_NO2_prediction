import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Reutilizar algunas funciones útiles de training.py
from training import (
    cargar_datos_trafico_y_meteo,
    preprocessing,
    convertir_unidades_legibles,
    filter_outliers,
    COLUMNS_FOR_OUTLIERS,
    VARIABLE_METADATA
)

def create_time_features(df):
    """
    Crea características temporales adicionales para el modelo.
    """
    df = df.copy()
    
    # Características existentes del preprocessing
    df = preprocessing(df)
    
    # Características temporales adicionales
    df['hour'] = df['fecha'].dt.hour
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (x in range(7, 10) or x in range(17, 20)) else 0)
    df['is_weekend'] = df['fecha'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_week'] = df['fecha'].dt.weekday
    df['week_of_year'] = df['fecha'].dt.isocalendar().week
    
    # Lag features de NO2 (promedios móviles)
    df['no2_rolling_mean_3h'] = df.groupby('id_no2')['no2_value'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['no2_rolling_mean_24h'] = df.groupby('id_no2')['no2_value'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    
    # Interacciones entre variables
    df['traffic_temp_interaction'] = df['intensidad'] * df['t2m']
    df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    
    return df

# def optimize_xgboost(X_train, y_train):
#     """
#     Optimiza los hiperparámetros de XGBoost usando RandomizedSearchCV.
#     """
#     param_dist = {
#         'n_estimators': randint(100, 500),
#         'max_depth': randint(3, 10),
#         'learning_rate': uniform(0.01, 0.3),
#         'subsample': uniform(0.6, 0.4),
#         'colsample_bytree': uniform(0.6, 0.4),
#         'min_child_weight': randint(1, 7)
#     }
    
#     xgb_model = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         n_jobs=-1,
#         random_state=42
#     )
    
#     random_search = RandomizedSearchCV(
#         xgb_model,
#         param_distributions=param_dist,
#         n_iter=20,
#         cv=5,
#         random_state=42,
#         n_jobs=-1,
#         verbose=1
#     )
    
#     random_search.fit(X_train, y_train)
#     return random_search.best_estimator_


def baseline_xgboost(X_train, y_train):
    """
    Entrena un modelo XGBoost con hiperparámetros muy simples (baseline).
    """
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    xgb_model.fit(X_train, y_train)
    return xgb_model


def train_xgboost(X_train, y_train, sensor_seleccionado, outlier_type, preprocessing_box):
    """
    Entrena el modelo XGBoost y lo guarda.
    """
    # Entrenar modelo
    model = baseline_xgboost(X_train, y_train)
    
    # Guardar modelo y metadatos
    model_info = {
        'model': model,
        'feature_names': X_train.columns.tolist(),
        'variable_metadata': VARIABLE_METADATA
    }
    
    file_name = f'data/models/xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_box}.pkl'
    joblib.dump(model_info, file_name)
    print(f"Modelo guardado en {file_name}")
    
    return model

def plot_feature_importance(model, feature_names):
    """
    Visualiza la importancia de las características del modelo XGBoost.
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df.plot(kind='barh', x='feature', y='importance', ax=ax)
    ax.set_title('Importancia de las Características')
    ax.set_xlabel('Importancia')
    st.pyplot(fig)
    plt.close()

def show_model_performance(model, X_test, y_test):
    """
    Muestra las métricas de rendimiento y gráficos de diagnóstico.
    """
    y_pred = model.predict(X_test)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f} µg/m³", "Menor es mejor")
    col2.metric("R² Score", f"{r2:.3f}", "Más cercano a 1 es mejor")
    col3.metric("MAE", f"{mae:.2f} µg/m³", "Menor es mejor")
    
    # Gráficos de diagnóstico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuos
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title('Distribución de Residuos')
    ax1.set_xlabel('Residuos (µg/m³)')
    
    # Predicho vs Real
    ax2.scatter(y_test, y_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel('Valores Reales (µg/m³)')
    ax2.set_ylabel('Valores Predichos (µg/m³)')
    ax2.set_title('Predicho vs Real')
    
    st.pyplot(fig)
    plt.close()

def xgboost_page():
    """Página principal para el modelo XGBoost."""
    
    st.header("XGBoost - Análisis de NO₂")
    
    # Cargar datos
    df = cargar_datos_trafico_y_meteo()
    
    # Configuración
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        sensor_seleccionado = st.selectbox(
            "Selecciona un sensor de NO₂",
            df['id_no2'].unique(),
            index=2,
            key="xgb_sensor_select"
        )
        
        df_sensor = df[df['id_no2'] == sensor_seleccionado]
        
        fecha_min = df_sensor["fecha"].min().date()
        fecha_max = df_sensor["fecha"].max().date()
        
        fecha_inicio = st.date_input(
            "Fecha de inicio para entrenamiento",
            fecha_min,
            min_value=fecha_min,
            max_value=fecha_max,
            key="xgb_fecha_inicio"
        )
        
        fecha_fin_training = st.date_input(
            "Fecha de inicio para evaluación",
            '2024-01-01',
            min_value=fecha_min,
            max_value=fecha_max,
            key="xgb_fecha_fin"
        )
    
    with config_col2:
        outlier_type = st.selectbox(
            "Tipo de filtrado de outliers",
            ["zscore", "iqr", "quantiles", "none"],
            key="xgb_outlier_type"
        )
        
        preprocessing_box = st.selectbox(
            "Preprocesamiento",
            ["advanced", "basic", "none"],
            key="xgb_preprocessing"
        )
    
    # Preprocesamiento de datos
    if outlier_type != "none":
        df_sensor = filter_outliers(df_sensor, outlier_type)
    
    df_sensor = convertir_unidades_legibles(df_sensor)
    
    if preprocessing_box == "advanced":
        df_sensor = create_time_features(df_sensor)
    elif preprocessing_box == "basic":
        df_sensor = preprocessing(df_sensor)
    
    # División de datos
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_training_dt = pd.to_datetime(fecha_fin_training)
    
    train_df = df_sensor[df_sensor['fecha'] < fecha_fin_training_dt]
    test_df = df_sensor[df_sensor['fecha'] >= fecha_fin_training_dt]
    
    # Selección de características
    feature_cols = [col for col in df_sensor.columns if col not in ['fecha', 'id_no2', 'no2_value']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['no2_value']
    X_test = test_df[feature_cols]
    y_test = test_df['no2_value']
    
    # Entrenamiento o carga del modelo
    model_path = f'data/models/xgboost_model_{sensor_seleccionado}_{outlier_type}_{preprocessing_box}.pkl'
    
    if os.path.exists(model_path):
        model_info = joblib.load(model_path)
        model = model_info['model']
        
        if st.button("Analizar Modelo"):
            with st.spinner("Analizando modelo..."):
                show_model_performance(model, X_test, y_test)
                st.subheader("Importancia de Características")
                plot_feature_importance(model, feature_cols)
    else:
        if st.button("Entrenar Nuevo Modelo"):
            with st.spinner("Entrenando modelo... Este proceso puede tardar varios minutos."):
                model = train_xgboost(X_train, y_train, sensor_seleccionado, outlier_type, preprocessing_box)
                st.success("¡Modelo entrenado correctamente!")
                
                show_model_performance(model, X_test, y_test)
                st.subheader("Importancia de Características")
                plot_feature_importance(model, feature_cols)
