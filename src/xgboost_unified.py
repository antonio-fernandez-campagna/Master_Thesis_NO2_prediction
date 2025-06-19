"""
Módulo unificado para entrenamiento de modelos XGBoost.

Incluye tanto modelos individuales (por sensor) como globales (multi-sensor)
con una interfaz limpia y modular.
"""

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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Importar configuraciones y funciones de visualización
from xgboost_training import (
    OUTLIER_METHODS, PREPROCESSING_OPTIONS, VARIABLE_CATEGORIES, 
    VARIABLE_METADATA, COLUMNS_FOR_OUTLIERS,
    show_model_metrics, show_residual_analysis, show_feature_importance,
    show_temporal_predictions, show_residuals_over_time,
    XGBoostTrainer
)


class XGBoostUnifiedTrainer:
    """Clase unificada para entrenamiento de modelos XGBoost individuales y globales."""
    
    def __init__(self):
        self.df_master = None
        self.individual_trainer = XGBoostTrainer()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'xgb_unified_data_loaded' not in st.session_state:
            st.session_state.xgb_unified_data_loaded = False
        if 'xgb_unified_mode' not in st.session_state:
            st.session_state.xgb_unified_mode = 'individual'
    
    @st.cache_data(ttl=3600)
    def load_data(_self) -> pd.DataFrame:
        """Carga y preprocesa los datos con caché."""
        try:
            df = pd.read_parquet('data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet')
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
            return pd.DataFrame()
    
    def show_data_overview(self):
        """Muestra overview del dataset completo."""
        st.header("📊 Overview del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total registros", f"{len(self.df_master):,}")
        with col2:
            st.metric("Sensores NO₂", self.df_master['id_no2'].nunique())
        with col3:
            st.metric("Sensores tráfico", self.df_master['id_trafico'].nunique())
        with col4:
            periodo_años = (self.df_master['fecha'].max() - self.df_master['fecha'].min()).days / 365.25
            st.metric("Período", f"{periodo_años:.1f} años")
        
        # Mostrar distribución por sensor
        with st.expander("📋 Distribución de Datos por Sensor"):
            sensor_stats = self.df_master.groupby('id_no2').agg({
                'fecha': ['min', 'max', 'count'],
                'no2_value': ['mean', 'std']
            }).round(2)
            sensor_stats.columns = ['fecha_min', 'fecha_max', 'registros', 'no2_mean', 'no2_std']
            st.dataframe(sensor_stats, use_container_width=True)


def show_individual_training():
    """Interfaz para entrenamiento de modelos individuales."""
    st.subheader("🎯 Entrenamiento por Sensor Individual")
    
    trainer = XGBoostTrainer()
    
    # Cargar datos
    if not st.session_state.xgboost_data_loaded:
        trainer.df_master = trainer.load_data()
        if not trainer.df_master.empty:
            st.session_state.xgboost_data_loaded = True
    else:
        trainer.df_master = trainer.load_data()
    
    if trainer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Configuración simplificada
    col1, col2 = st.columns(2)
    
    with col1:
        sensores = sorted(trainer.df_master['id_no2'].unique())
        sensor_seleccionado = st.selectbox(
            "Sensor de NO₂", 
            sensores, 
            index=2 if len(sensores) > 2 else 0,
            key="individual_sensor"
        )
    
    with col2:
        fecha_division = st.date_input(
            "Fecha de división",
            value=pd.to_datetime('2024-01-01').date(),
            key="individual_split_date"
        )
    
    # Información del sensor seleccionado
    df_sensor = trainer.df_master[trainer.df_master['id_no2'] == sensor_seleccionado]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros totales", len(df_sensor))
    with col2:
        train_count = len(df_sensor[df_sensor['fecha'] < pd.to_datetime(fecha_division)])
        st.metric("Registros entrenamiento", train_count)
    with col3:
        test_count = len(df_sensor[df_sensor['fecha'] >= pd.to_datetime(fecha_division)])
        st.metric("Registros evaluación", test_count)
    
    # Botón de entrenamiento
    if st.button("🚀 Entrenar Modelo Individual", type="primary"):
        st.info("Redirigiendo a entrenamiento individual detallado...")
        st.info("💡 Tip: Usa la página de 'XGBoost Training' para configuración avanzada")


def show_global_training(unified_trainer):
    """Interfaz para entrenamiento de modelos globales."""
    st.subheader("🌍 Entrenamiento Global Multi-Sensor")
    
    # Configuración de sensores para train/test
    st.markdown("### Configuración de Sensores")
    
    sensores_disponibles = sorted(unified_trainer.df_master['id_no2'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Sensores para Entrenamiento**")
        sensores_train = st.multiselect(
            "Selecciona sensores para entrenar:",
            sensores_disponibles,
            default=sensores_disponibles[:-2],  # Todos menos los últimos 2
            key="global_train_sensors"
        )
    
    with col2:
        st.markdown("**🧪 Sensores para Evaluación**")
        sensores_test = st.multiselect(
            "Selecciona sensores para evaluar:",
            sensores_disponibles,
            default=sensores_disponibles[-2:],  # Los últimos 2
            key="global_test_sensors"
        )
    
    # Validaciones
    if not sensores_train:
        st.warning("⚠️ Selecciona al menos un sensor para entrenamiento")
        return
    
    if not sensores_test:
        st.warning("⚠️ Selecciona al menos un sensor para evaluación")
        return
    
    # Mostrar estadísticas de la configuración
    df_train = unified_trainer.df_master[unified_trainer.df_master['id_no2'].isin(sensores_train)]
    df_test = unified_trainer.df_master[unified_trainer.df_master['id_no2'].isin(sensores_test)]
    
    st.markdown("### 📊 Estadísticas de la Configuración")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sensores entrenamiento", len(sensores_train))
    with col2:
        st.metric("Registros entrenamiento", f"{len(df_train):,}")
    with col3:
        st.metric("Sensores evaluación", len(sensores_test))
    with col4:
        st.metric("Registros evaluación", f"{len(df_test):,}")
    
    # Configuración adicional
    col1, col2 = st.columns(2)
    
    with col1:
        outlier_method = st.selectbox(
            "Método de outliers",
            options=list(OUTLIER_METHODS.keys()),
            format_func=lambda x: OUTLIER_METHODS[x],
            index=0,
            key="global_outlier_method"
        )
    
    with col2:
        preprocessing = st.selectbox(
            "Preprocesamiento temporal",
            options=list(PREPROCESSING_OPTIONS.keys()),
            format_func=lambda x: PREPROCESSING_OPTIONS[x],
            index=0,
            key="global_preprocessing"
        )
    
    # Variables seleccionadas automáticamente
    selected_features = []
    
    # Variables temporales (según preprocesamiento)
    if preprocessing == 'sin_cos':
        temporal_vars = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
            'weekend', 'season_sin', 'season_cos'
        ]
    else:
        temporal_vars = ['hour', 'month', 'day_of_week', 'day_of_year', 'weekend']
    
    # Variables de tráfico y meteorológicas (siempre disponibles)
    traffic_vars = ['intensidad', 'carga', 'ocupacion', 'vmed']
    meteo_vars = ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']
    
    # CREAR DATASET DE MUESTRA PARA VALIDAR FEATURES
    sample_df = unified_trainer.df_master.head(100).copy()
    
    # Aplicar preprocesamiento a la muestra para verificar features
    if preprocessing == 'sin_cos':
        sample_df = unified_trainer.individual_trainer.create_cyclical_features(sample_df)
    
    sample_df = unified_trainer.individual_trainer.convert_units(sample_df)
    
    # VALIDAR Y AGREGAR SOLO FEATURES EXISTENTES
    available_temporal = [var for var in temporal_vars if var in sample_df.columns]
    available_traffic = [var for var in traffic_vars if var in sample_df.columns]
    available_meteo = [var for var in meteo_vars if var in sample_df.columns]
    
    selected_features.extend(available_temporal)
    selected_features.extend(available_traffic)
    selected_features.extend(available_meteo)
    
    # Validar que tenemos features suficientes
    if len(selected_features) < 5:
        st.error(f"❌ Muy pocas variables disponibles: {len(selected_features)}")
        st.error(f"Variables encontradas: {selected_features}")
        return
    
    # Mostrar variables seleccionadas
    with st.expander("📋 Variables del Modelo Global"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Temporales:**")
            for var in available_temporal[:7]:
                st.write(f"• {var}")
            if len(available_temporal) > 7:
                st.write(f"... y {len(available_temporal) - 7} más")
        with col2:
            st.write("**Tráfico:**")
            for var in available_traffic:
                st.write(f"• {var}")
        with col3:
            st.write("**Meteorológicas:**")
            for var in available_meteo[:7]:
                st.write(f"• {var}")
            if len(available_meteo) > 7:
                st.write(f"... y {len(available_meteo) - 7} más")
        
        st.write(f"**Total variables:** {len(selected_features)}")
        
        # Mostrar variables faltantes si las hay
        missing_temporal = [var for var in temporal_vars if var not in available_temporal]
        missing_traffic = [var for var in traffic_vars if var not in available_traffic]
        missing_meteo = [var for var in meteo_vars if var not in available_meteo]
        
        if missing_temporal or missing_traffic or missing_meteo:
            st.warning("⚠️ Variables no disponibles:")
            if missing_temporal:
                st.write(f"Temporales: {missing_temporal}")
            if missing_traffic:
                st.write(f"Tráfico: {missing_traffic}")
            if missing_meteo:
                st.write(f"Meteorológicas: {missing_meteo}")
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Entrenar Modelo Global", type="primary"):
            train_global_model(
                unified_trainer, sensores_train, sensores_test, 
                selected_features, outlier_method, preprocessing
            )
    
    with col2:
        if st.button("🔍 Analizar Modelo Existente"):
            st.info("Funcionalidad de análisis en desarrollo...")
    
    # Mostrar resultados si existen en session_state
    if 'global_model_results' in st.session_state:
        results = st.session_state.global_model_results
        
        st.divider()
        st.subheader("📊 Resultados del Modelo Global")
        
        # Mostrar métricas globales
        st.markdown("### 📈 Métricas Globales")
        show_model_metrics(results['metrics'])
        
        # Mostrar análisis por sensor
        show_global_sensor_analysis(
            results['test_df'], 
            results['model'], 
            results['selected_features'], 
            results['scaler_target'], 
            results['sensores_test']
        )


def train_global_model(unified_trainer, sensores_train, sensores_test, 
                      selected_features, outlier_method, preprocessing):
    """Entrena un modelo global con la configuración especificada."""
    
    with st.spinner("Entrenando modelo global..."):
        # Preparar datos
        df_train = unified_trainer.df_master[unified_trainer.df_master['id_no2'].isin(sensores_train)]
        df_test = unified_trainer.df_master[unified_trainer.df_master['id_no2'].isin(sensores_test)]
        
        # Aplicar preprocesamiento ANTES de seleccionar features
        if preprocessing == 'sin_cos':
            df_train = unified_trainer.individual_trainer.create_cyclical_features(df_train)
            df_test = unified_trainer.individual_trainer.create_cyclical_features(df_test)
        
        # Convertir unidades
        df_train = unified_trainer.individual_trainer.convert_units(df_train)
        df_test = unified_trainer.individual_trainer.convert_units(df_test)
        
        # VALIDAR QUE TODAS LAS FEATURES EXISTEN
        missing_features_train = [f for f in selected_features if f not in df_train.columns]
        missing_features_test = [f for f in selected_features if f not in df_test.columns]
        
        if missing_features_train or missing_features_test:
            st.error(f"❌ Features faltantes en train: {missing_features_train}")
            st.error(f"❌ Features faltantes en test: {missing_features_test}")
            return
        
        # Eliminar outliers solo en entrenamiento
        if outlier_method != 'none':
            df_train = unified_trainer.individual_trainer.remove_outliers(df_train, outlier_method)
        
        # Preparar matrices DESPUÉS de validar features
        X_train = df_train[selected_features].copy()
        y_train = df_train['no2_value'].copy()
        X_test = df_test[selected_features].copy()
        y_test = df_test['no2_value'].copy()
        
        # Limpiar NaNs ANTES del escalado
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        df_test_clean = df_test[test_mask]
        
        # Validar que tenemos datos suficientes
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("❌ No hay datos suficientes después de la limpieza")
            return
        
        # ESCALADO SIN DATA LEAKAGE - Solo usar X_train para fit
        scaler_dict = {}
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        for feature in selected_features:
            if feature in X_train.columns and pd.api.types.is_numeric_dtype(X_train[feature]):
                scaler = StandardScaler()
                # FIT solo con datos de ENTRENAMIENTO
                X_train_scaled[feature] = scaler.fit_transform(X_train[[feature]]).flatten()
                # TRANSFORM datos de TEST
                X_test_scaled[feature] = scaler.transform(X_test[[feature]]).flatten()
                scaler_dict[feature] = scaler
        
        # Escalar target solo con datos de entrenamiento
        y_train_scaled, scaler_target = unified_trainer.individual_trainer.scale_target(y_train)
        # Escalar y_test para validación (usando scaler ya entrenado)
        y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Entrenar modelo
        model = unified_trainer.individual_trainer.train_xgboost_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
        )
        
        # Evaluar modelo
        metrics = unified_trainer.individual_trainer.evaluate_model(
            model, X_test_scaled, y_test, scaler_target
        )
        
        # Guardar resultados en session_state
        st.session_state.global_model_results = {
            'model': model,
            'metrics': metrics,
            'test_df': df_test_clean,
            'sensores_train': sensores_train,
            'sensores_test': sensores_test,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'selected_features': selected_features
        }
        
        st.success("✅ Modelo global entrenado exitosamente!")
        st.info("📊 Los resultados se muestran a continuación...")


def show_global_sensor_analysis(test_df, model, selected_features, scaler_target, sensores_test):
    """Muestra análisis detallado por sensor para el modelo global."""
    
    st.subheader("🎯 Análisis por Sensor de Evaluación")
    
    # OBTENER SCALER_DICT UNA SOLA VEZ (optimización)
    scaler_dict = None
    if 'global_model_results' in st.session_state:
        scaler_dict = st.session_state.global_model_results['scaler_dict']
    
    # Calcular métricas por sensor
    sensor_metrics = []
    
    for sensor_id in sensores_test:
        sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
        
        if len(sensor_data) == 0:
            continue
        
        X_sensor = sensor_data[selected_features]
        y_sensor = sensor_data['no2_value']
        
        # APLICAR EL MISMO ESCALADO QUE EN EL ENTRENAMIENTO
        if scaler_dict is not None:
            # Escalar las features usando los mismos scalers del entrenamiento
            X_sensor_scaled = X_sensor.copy()
            for feature in selected_features:
                if feature in scaler_dict:
                    X_sensor_scaled[feature] = scaler_dict[feature].transform(X_sensor[[feature]])
            
            # Filtrar solo variables numéricas
            numeric_features = X_sensor_scaled.select_dtypes(include=[np.number]).columns.tolist()
            X_sensor_numeric = X_sensor_scaled[numeric_features]
        else:
            # Fallback si no hay scaler_dict
            numeric_features = X_sensor.select_dtypes(include=[np.number]).columns.tolist()
            X_sensor_numeric = X_sensor[numeric_features]
        
        # Predicciones
        y_pred_scaled = model.predict(X_sensor_numeric)
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_sensor, y_pred))
        r2 = r2_score(y_sensor, y_pred)
        mae = mean_absolute_error(y_sensor, y_pred)
        
        sensor_metrics.append({
            'sensor_id': sensor_id,
            'n_samples': len(sensor_data),
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'no2_mean': y_sensor.mean(),
            'no2_std': y_sensor.std()
        })
    
    sensor_metrics_df = pd.DataFrame(sensor_metrics)
    
    # Validar que tenemos métricas
    if sensor_metrics_df.empty:
        st.error("❌ No se pudieron calcular métricas por sensor")
        return
    
    # Mostrar métricas resumidas en columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "RMSE Promedio",
            f"{sensor_metrics_df['rmse'].mean():.2f} µg/m³",
            f"±{sensor_metrics_df['rmse'].std():.2f}"
        )
    
    with col2:
        st.metric(
            "R² Promedio", 
            f"{sensor_metrics_df['r2'].mean():.3f}",
            f"±{sensor_metrics_df['r2'].std():.3f}"
        )
    
    with col3:
        st.metric(
            "MAE Promedio",
            f"{sensor_metrics_df['mae'].mean():.2f} µg/m³",
            f"±{sensor_metrics_df['mae'].std():.2f}"
        )
    
    # Mostrar tabla de métricas
    with st.expander("📋 Métricas Detalladas por Sensor"):
        st.dataframe(
            sensor_metrics_df.style.format({
                'rmse': '{:.2f}',
                'r2': '{:.3f}',
                'mae': '{:.2f}',
                'no2_mean': '{:.2f}',
                'no2_std': '{:.2f}'
            }),
            use_container_width=True
        )
    
    # Guardar métricas en session_state para análisis detallado
    st.session_state.sensor_metrics_data = {
        'test_df': test_df,
        'model': model,
        'selected_features': selected_features,
        'scaler_target': scaler_target,
        'scaler_dict': scaler_dict,  # Pasar scaler_dict directamente
        'sensores_test': sensores_test,
        'sensor_metrics_df': sensor_metrics_df
    }
    
    # Selector para análisis individual
    st.markdown("### 🔍 Análisis Detallado por Sensor")
    
    sensor_seleccionado = st.selectbox(
        "Selecciona sensor para análisis detallado:",
        sensores_test,
        key="global_analysis_sensor"
    )
    
    # Usar checkbox en lugar de botón para evitar rerun
    if st.checkbox("📈 Mostrar Análisis Detallado", key="show_detailed_analysis"):
        show_detailed_sensor_analysis(test_df, model, selected_features, scaler_target, sensor_seleccionado, scaler_dict)


def show_detailed_sensor_analysis(test_df, model, selected_features, scaler_target, sensor_id, scaler_dict):
    """Muestra análisis detallado para un sensor específico."""
    
    sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
    
    if len(sensor_data) == 0:
        st.error(f"No hay datos para el sensor {sensor_id}")
        return
    
    X_sensor = sensor_data[selected_features]
    y_sensor = sensor_data['no2_value']
    
    # APLICAR EL MISMO ESCALADO QUE EN EL ENTRENAMIENTO
    if scaler_dict is not None:
        # Escalar las features usando los mismos scalers del entrenamiento
        X_sensor_scaled = X_sensor.copy()
        for feature in selected_features:
            if feature in scaler_dict:
                X_sensor_scaled[feature] = scaler_dict[feature].transform(X_sensor[[feature]])
        
        # Filtrar solo variables numéricas
        numeric_features = X_sensor_scaled.select_dtypes(include=[np.number]).columns.tolist()
        X_sensor_numeric = X_sensor_scaled[numeric_features]
    else:
        # Fallback si no hay scaler_dict
        numeric_features = X_sensor.select_dtypes(include=[np.number]).columns.tolist()
        X_sensor_numeric = X_sensor[numeric_features]
    
    # Predicciones
    y_pred_scaled = model.predict(X_sensor_numeric)
    y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    st.subheader(f"📊 Análisis Detallado - Sensor {sensor_id}")
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_sensor, y_pred))
    r2 = r2_score(y_sensor, y_pred)
    mae = mean_absolute_error(y_sensor, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse:.2f} µg/m³")
    with col2:
        st.metric("R²", f"{r2:.3f}")
    with col3:
        st.metric("MAE", f"{mae:.2f} µg/m³")
    
    # Gráficos temporales
    show_temporal_predictions(sensor_data, y_pred)


def show_info_panel():
    """Muestra panel de información sobre el módulo unificado."""
    with st.expander("ℹ️ Acerca del Módulo XGBoost Unificado", expanded=False):
        st.markdown("""
        **🎯 Entrenamiento XGBoost Unificado**
        
        Este módulo permite entrenar y comparar dos tipos de modelos:
        
        **🏠 Modelos Individuales:**
        - Un modelo por sensor
        - Especializado en patrones locales
        - Ideal para análisis específico por ubicación
        
        **🌍 Modelos Globales:**
        - Un modelo entrenado con múltiples sensores
        - Aprende patrones generales transferibles
        - Ideal para nowcasting en nuevas ubicaciones
        
        **🔬 Configuración Experimental:**
        - Selecciona sensores para entrenamiento vs evaluación
        - Evalúa transferibilidad entre ubicaciones
        - Análisis detallado por sensor individual
        
        **🚀 Aplicaciones:**
        - Comparar rendimiento individual vs global
        - Validar transferibilidad de modelos
        - Seleccionar estrategia óptima para nowcasting
        """)


def xgboost_unified_page():
    """Página principal del módulo XGBoost unificado."""
    
    st.title("🚀 XGBoost Training - Modelos Individuales vs Globales")
    
    # Panel de información
    show_info_panel()
    
    # Inicializar trainer
    unified_trainer = XGBoostUnifiedTrainer()
    
    # Cargar datos
    if not st.session_state.xgb_unified_data_loaded:
        if st.button("📊 Cargar Dataset Completo", type="primary"):
            with st.spinner("Cargando dataset completo..."):
                unified_trainer.df_master = unified_trainer.load_data()
                if not unified_trainer.df_master.empty:
                    st.session_state.xgb_unified_data_loaded = True
                    st.success("¡Dataset cargado exitosamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    unified_trainer.df_master = unified_trainer.load_data()
    
    if unified_trainer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Mostrar overview
    unified_trainer.show_data_overview()
    
    # Selector de modo
    st.header("🎯 Selecciona Tipo de Modelo")
    
    mode = st.radio(
        "Tipo de entrenamiento:",
        ["🏠 Individual (por sensor)", "🌍 Global (multi-sensor)"],
        index=0 if st.session_state.xgb_unified_mode == 'individual' else 1,
        horizontal=True
    )
    
    # Actualizar estado
    if "Individual" in mode:
        st.session_state.xgb_unified_mode = 'individual'
    else:
        st.session_state.xgb_unified_mode = 'global'
    
    st.divider()
    
    # Mostrar interfaz según el modo
    if st.session_state.xgb_unified_mode == 'individual':
        show_individual_training()
    else:
        show_global_training(unified_trainer)


if __name__ == "__main__":
    xgboost_unified_page() 