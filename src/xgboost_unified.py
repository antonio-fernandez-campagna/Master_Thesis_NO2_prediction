"""
Módulo unificado para entrenamiento de modelos XGBoost.

Incluye tanto modelos individuales (por sensor) como globales (multi-sensor)
con una interfaz limpia y modular aplicando mejores prácticas de ingeniería de software.
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
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Importar configuraciones y funciones desde el módulo base
from xgboost_training import (
    OUTLIER_METHODS, PREPROCESSING_OPTIONS, VARIABLE_CATEGORIES, 
    VARIABLE_METADATA, COLUMNS_FOR_OUTLIERS,
    show_model_metrics, show_residual_analysis, show_feature_importance,
    show_temporal_predictions, show_residuals_over_time,
    XGBoostTrainer
)


class XGBoostUnifiedTrainer(XGBoostTrainer):
    """
    Clase unificada para entrenamiento XGBoost que extiende XGBoostTrainer.
    Maneja tanto modelos individuales como globales con una interfaz consistente.
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_unified_session_state()
    
    def _initialize_unified_session_state(self):
        """Inicializa estados específicos del módulo unificado."""
        if 'xgb_unified_data_loaded' not in st.session_state:
            st.session_state.xgb_unified_data_loaded = False
        if 'xgb_unified_mode' not in st.session_state:
            st.session_state.xgb_unified_mode = 'individual'
        if 'xgb_unified_config' not in st.session_state:
            st.session_state.xgb_unified_config = {}
        if 'xgb_unified_analysis_data' not in st.session_state:
            st.session_state.xgb_unified_analysis_data = {}
    
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

    def show_variable_selection(self, key_prefix: str = "") -> List[str]:
        """
        Muestra interfaz de selección de variables reutilizable.
        
        Args:
            key_prefix: Prefijo para las claves de session_state
            
        Returns:
            Lista de variables seleccionadas
        """
        st.subheader("🔧 Selección de Variables")
        
        # Crear tabs para categorías
        var_tabs = st.tabs(list(VARIABLE_CATEGORIES.keys()))
        
        selected_features = []
        for i, (category, vars_list) in enumerate(VARIABLE_CATEGORIES.items()):
            with var_tabs[i]:
                # Filtrar variables que existen en los datos o se pueden crear
                available_vars = [var for var in vars_list if var in self.df_master.columns or 'sin' in var or 'cos' in var]
                
                # Configurar defaults específicos para cada categoría
                if category == "Variables Temporales":
                    # Para XGBoost, preferir variables no cíclicas por defecto
                    default_vars = [var for var in available_vars if not ('sin' in var or 'cos' in var)]
                else:
                    default_vars = available_vars
                
                selected_in_category = st.multiselect(
                    f"Variables de {category}",
                    available_vars,
                    default=default_vars,
                    help=f"Selecciona las variables de {category.lower()} para el modelo",
                    key=f"{key_prefix}_{category.replace(' ', '_')}_vars"
                )
                selected_features.extend(selected_in_category)
        
        return selected_features

    def show_configuration_panel(self, mode: str, key_prefix: str = "") -> Dict:
        """
        Muestra panel de configuración reutilizable.
        
        Args:
            mode: 'individual' o 'global'
            key_prefix: Prefijo para las claves de session_state
            
        Returns:
            Diccionario con la configuración seleccionada
        """
        st.subheader("⚙️ Configuración del Modelo")
        
        config = {}
        
        if mode == 'individual':
            col1, col2 = st.columns(2)
            
            with col1:
                # Selección de sensor
                sensores = sorted(self.df_master['id_no2'].unique())
                sensor_seleccionado = st.selectbox(
                    "Sensor de NO₂", 
                    sensores, 
                    index=2 if len(sensores) > 2 else 0,
                    key=f"{key_prefix}_sensor"
                )
                config['sensor'] = sensor_seleccionado
                
                # Filtrar por sensor y obtener fechas
                df_sensor = self.df_master[self.df_master['id_no2'] == sensor_seleccionado]
                fecha_min = df_sensor["fecha"].min().date()
                fecha_max = df_sensor["fecha"].max().date()
                
                # Fecha de división
                fecha_division = st.date_input(
                    "Fecha de división (entrenamiento/prueba)",
                    value=pd.to_datetime('2024-01-01').date(),
                    min_value=fecha_min,
                    max_value=fecha_max,
                    help="Los datos anteriores se usan para entrenamiento, posteriores para evaluación",
                    key=f"{key_prefix}_split_date"
                )
                config['fecha_division'] = fecha_division
                config['df_sensor'] = df_sensor
            
            with col2:
                config.update(self._show_preprocessing_options(key_prefix))
        
        else:  # mode == 'global'
            st.markdown("### 🌍 Configuración Global Multi-Sensor")
            
            # Configuración de sensores para train/test
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Sensores para Entrenamiento**")
                sensores_disponibles = sorted(self.df_master['id_no2'].unique())
                sensores_train = st.multiselect(
                    "Selecciona sensores para entrenar:",
                    sensores_disponibles,
                    default=sensores_disponibles[:-2],  # Todos menos los últimos 2
                    key=f"{key_prefix}_train_sensors"
                )
                config['sensores_train'] = sensores_train
            
            with col2:
                st.markdown("**🧪 Sensores para Evaluación**")
                sensores_test = st.multiselect(
                    "Selecciona sensores para evaluar:",
                    sensores_disponibles,
                    default=sensores_disponibles[-2:],  # Los últimos 2
                    key=f"{key_prefix}_test_sensors"
                )
                config['sensores_test'] = sensores_test
            
            # Validaciones
            if not sensores_train or not sensores_test:
                if not sensores_train:
                    st.warning("⚠️ Selecciona al menos un sensor para entrenamiento")
                if not sensores_test:
                    st.warning("⚠️ Selecciona al menos un sensor para evaluación")
                return config
            
            # Mostrar estadísticas
            self._show_global_config_stats(sensores_train, sensores_test)
            
            # Configuración adicional
            col1, col2 = st.columns(2)
            with col1:
                config.update(self._show_preprocessing_options(key_prefix))
        
        return config

    def _show_preprocessing_options(self, key_prefix: str) -> Dict:
        """Muestra opciones de preprocesamiento."""
        config = {}
        
        # Método de filtrado de outliers
        outlier_method = st.selectbox(
            "Método de filtrado de outliers",
            options=list(OUTLIER_METHODS.keys()),
            format_func=lambda x: OUTLIER_METHODS[x],
            index=0,  # none por defecto para XGBoost
            key=f"{key_prefix}_outlier_method"
        )
        config['outlier_method'] = outlier_method
        
        # Preprocesamiento temporal
        preprocessing = st.selectbox(
            "Preprocesamiento temporal",
            options=list(PREPROCESSING_OPTIONS.keys()),
            format_func=lambda x: PREPROCESSING_OPTIONS[x],
            index=0,  # sin_cos por defecto
            key=f"{key_prefix}_preprocessing"
        )
        config['preprocessing'] = preprocessing
        
        return config

    def _show_global_config_stats(self, sensores_train: List[str], sensores_test: List[str]):
        """Muestra estadísticas de configuración global."""
        df_train = self.df_master[self.df_master['id_no2'].isin(sensores_train)]
        df_test = self.df_master[self.df_master['id_no2'].isin(sensores_test)]
        
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

    def show_configuration_summary(self, config: Dict, selected_features: List[str]):
        """Muestra resumen de configuración."""
        with st.expander("📋 Resumen de Configuración"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'sensor' in config:
                    st.write(f"**Sensor:** {config['sensor']}")
                else:
                    st.write(f"**Sensores Train:** {len(config.get('sensores_train', []))}")
                    st.write(f"**Sensores Test:** {len(config.get('sensores_test', []))}")
                st.write(f"**Variables:** {len(selected_features)}")
            
            with col2:
                st.write(f"**Outliers:** {OUTLIER_METHODS[config['outlier_method']]}")
                st.write(f"**Preprocesamiento:** {PREPROCESSING_OPTIONS[config['preprocessing']]}")
            
            with col3:
                if 'fecha_division' in config:
                    st.write(f"**Fecha división:** {config['fecha_division']}")
                if 'df_sensor' in config:
                    fecha_min = config['df_sensor']["fecha"].min().date()
                    st.write(f"**Período entreno:** {fecha_min} - {config['fecha_division']}")

    def prepare_data(self, config: Dict, selected_features: List[str], mode: str) -> Dict:
        """
        Prepara los datos según la configuración especificada.
        
        Args:
            config: Configuración del modelo
            selected_features: Variables seleccionadas
            mode: 'individual' o 'global'
            
        Returns:
            Diccionario con los datos preparados
        """
        with st.spinner("Preparando datos..."):
            if mode == 'individual':
                return self._prepare_individual_data(config, selected_features)
            else:
                return self._prepare_global_data(config, selected_features)

    def _prepare_individual_data(self, config: Dict, selected_features: List[str]) -> Dict:
        """Prepara datos para entrenamiento individual."""
        df_processed = config['df_sensor'].copy()
        
        st.write("📊 Datos originales:", len(df_processed))
        
        # Aplicar transformaciones
        if config['preprocessing'] == 'sin_cos':
            df_processed = self.create_cyclical_features(df_processed)
        
        df_processed = self.convert_units(df_processed)
        
        # Dividir datos ANTES de eliminar outliers
        fecha_division_dt = pd.to_datetime(config['fecha_division'])
        train_df, test_df = self.split_data(df_processed, fecha_division_dt)
        
        st.write("📅 Datos entrenamiento (antes outliers):", len(train_df))
        st.write("📅 Datos evaluación:", len(test_df))
        
        # Eliminar outliers SOLO del conjunto de entrenamiento
        outliers_removed = 0
        if config['outlier_method'] != 'none':
            len_before = len(train_df)
            train_df = self.remove_outliers(train_df, config['outlier_method'])
            outliers_removed = len_before - len(train_df)
            st.write("🔍 Datos entrenamiento (después outliers):", len(train_df))
            st.write(f"❌ Outliers eliminados: {outliers_removed}")
        
        return {
            'train_df': train_df,
            'test_df': test_df,
            'outliers_removed': outliers_removed,
            'mode': 'individual'
        }

    def _prepare_global_data(self, config: Dict, selected_features: List[str]) -> Dict:
        """Prepara datos para entrenamiento global."""
        # Preparar datos
        df_train = self.df_master[self.df_master['id_no2'].isin(config['sensores_train'])]
        df_test = self.df_master[self.df_master['id_no2'].isin(config['sensores_test'])]
        
        st.write("📊 Datos entrenamiento originales:", len(df_train))
        st.write("📊 Datos evaluación originales:", len(df_test))
        
        # Aplicar preprocesamiento ANTES de seleccionar features
        if config['preprocessing'] == 'sin_cos':
            df_train = self.create_cyclical_features(df_train)
            df_test = self.create_cyclical_features(df_test)
        
        # Convertir unidades
        df_train = self.convert_units(df_train)
        df_test = self.convert_units(df_test)
        
        # Validar que todas las features existen
        missing_features_train = [f for f in selected_features if f not in df_train.columns]
        missing_features_test = [f for f in selected_features if f not in df_test.columns]
        
        if missing_features_train or missing_features_test:
            st.error(f"❌ Features faltantes en train: {missing_features_train}")
            st.error(f"❌ Features faltantes en test: {missing_features_test}")
            return {}
        
        # Eliminar outliers solo en entrenamiento
        outliers_removed = 0
        if config['outlier_method'] != 'none':
            len_before = len(df_train)
            df_train = self.remove_outliers(df_train, config['outlier_method'])
            outliers_removed = len_before - len(df_train)
            st.write(f"❌ Outliers eliminados: {outliers_removed}")
        
        st.write("📅 Datos entrenamiento finales:", len(df_train))
        st.write("📅 Datos evaluación finales:", len(df_test))
        
        return {
            'train_df': df_train,
            'test_df': df_test,
            'outliers_removed': outliers_removed,
            'mode': 'global'
        }

    def train_model(self, data_prep: Dict, selected_features: List[str], config: Dict) -> Dict:
        """
        Entrena el modelo con los datos preparados.
        
        Args:
            data_prep: Datos preparados
            selected_features: Variables seleccionadas
            config: Configuración del modelo
            
        Returns:
            Diccionario con el modelo entrenado y métricas
        """
        with st.spinner("Entrenando modelo XGBoost..."):
            train_df = data_prep['train_df']
            test_df = data_prep['test_df']
            
            # Preparar matrices
            X_train = train_df[selected_features].copy()
            y_train = train_df['no2_value'].copy()
            X_test = test_df[selected_features].copy()
            y_test = test_df['no2_value'].copy()
            
            # Limpiar NaNs
            train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            test_df_clean = test_df[test_mask]
            
            # Validar datos
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("❌ No hay datos suficientes después de la limpieza")
                return {}
            
            # Escalar datos
            X_train_scaled, X_test_scaled, scaler_dict = self.scale_features(X_train, X_test, selected_features)
            y_train_scaled, scaler_target = self.scale_target(y_train)
            y_test_scaled, _ = self.scale_target(y_test)
            
            # Entrenar modelo
            model = self.train_xgboost_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
            
            # Evaluar modelo
            metrics = self.evaluate_model(model, X_test_scaled, y_test, scaler_target)
            
            # Guardar modelo si es individual
            model_path = None
            if data_prep['mode'] == 'individual':
                model_path = self.save_model(
                    model, selected_features, scaler_dict, scaler_target,
                    config['sensor'], config['outlier_method'], config['preprocessing']
                )
                st.success(f"✅ Modelo entrenado y guardado en: {model_path}")
            else:
                st.success("✅ Modelo global entrenado exitosamente!")
                
            return {
                'model': model,
                'metrics': metrics,
                'test_df': test_df_clean,
                'scaler_dict': scaler_dict,
                'scaler_target': scaler_target,
                'selected_features': selected_features,
                'config': config,
                'model_path': model_path
            }

    def show_analysis_interface(self, results: Dict, config_key: str):
        """
        Muestra interfaz de análisis simplificada.
        
        Args:
            results: Resultados del modelo
            config_key: Clave para almacenar en session_state
        """
        st.header("📊 Análisis del Modelo XGBoost")
        
        # Guardar resultados en session_state
        st.session_state.xgb_unified_analysis_data[config_key] = results
        
        # Mostrar directamente todo el análisis
        self._show_analysis_content(results)

    def _show_analysis_content(self, results: Dict):
        """Muestra todo el contenido de análisis directamente."""
        # Mostrar importancia de variables
        st.subheader("🎯 Importancia de Variables")
        show_feature_importance(results['model'], results['selected_features'])
        
        # Si es modelo global, mostrar análisis por sensor
        if results['config'].get('sensores_test'):
            st.divider()
            self._show_global_sensor_analysis(results)
        
        # Si es modelo individual, mostrar análisis temporal
        elif 'sensor' in results['config']:
            st.divider()
            st.subheader("📈 Análisis Temporal del Sensor")
            show_temporal_predictions(results['test_df'], results['metrics']['y_pred'])
            
            st.divider()
            st.subheader("📉 Análisis de Residuos Temporales")
            show_residuals_over_time(results['test_df'], results['metrics']['y_pred'])
            
            st.divider()
            st.subheader("📊 Análisis de Residuos")
            show_residual_analysis(
                results['test_df']['no2_value'], 
                results['metrics']['y_pred']
            )
            
            st.divider()
            st.subheader("📈 Métricas de Evaluación")
            show_model_metrics(results['metrics'])

    def _show_global_sensor_analysis(self, results: Dict):
        """Muestra análisis detallado por sensor para modelo global."""
        st.subheader("🌍 Análisis por Sensor de Evaluación")
        
        sensores_test = results['config']['sensores_test']
        test_df = results['test_df']
        model = results['model']
        selected_features = results['selected_features']
        scaler_target = results['scaler_target']
        scaler_dict = results['scaler_dict']
        
        # Calcular métricas por sensor
        sensor_metrics = []
        sensor_predictions = {}  # Guardar predicciones para análisis detallado
        
        for sensor_id in sensores_test:
            sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
            
            if len(sensor_data) == 0:
                continue
            
            X_sensor = sensor_data[selected_features]
            y_sensor = sensor_data['no2_value']
            
            # Aplicar escalado
            X_sensor_scaled = X_sensor.copy()
            for feature in selected_features:
                if feature in scaler_dict:
                    X_sensor_scaled[feature] = scaler_dict[feature].transform(X_sensor[[feature]])
            
            # Filtrar variables numéricas
            numeric_features = X_sensor_scaled.select_dtypes(include=[np.number]).columns.tolist()
            X_sensor_numeric = X_sensor_scaled[numeric_features]
            
            # Predicciones
            y_pred_scaled = model.predict(X_sensor_numeric)
            y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Guardar predicciones para análisis detallado
            sensor_predictions[sensor_id] = {
                'sensor_data': sensor_data,
                'y_true': y_sensor,
                'y_pred': y_pred,
                'rmse': np.sqrt(mean_squared_error(y_sensor, y_pred)),
                'r2': r2_score(y_sensor, y_pred),
                'mae': mean_absolute_error(y_sensor, y_pred)
            }
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(y_sensor, y_pred))
            r2 = r2_score(y_sensor, y_pred)
            mae = mean_absolute_error(y_sensor, y_pred)
            
            # Análisis de residuos
            residuos = y_sensor - y_pred
            residuo_std = residuos.std()
            residuo_mean = residuos.mean()
            
            sensor_metrics.append({
                'sensor_id': sensor_id,
                'n_samples': len(sensor_data),
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'no2_mean': y_sensor.mean(),
                'no2_std': y_sensor.std(),
                'residuo_mean': residuo_mean,
                'residuo_std': residuo_std,
                'periodo_min': sensor_data['fecha'].min(),
                'periodo_max': sensor_data['fecha'].max()
            })
        
        if not sensor_metrics:
            st.error("❌ No se pudieron calcular métricas por sensor")
            return
        
        sensor_metrics_df = pd.DataFrame(sensor_metrics)
        
        # Mostrar métricas resumidas
        st.markdown("### 📊 Métricas Globales por Sensor")
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            best_sensor = sensor_metrics_df.loc[sensor_metrics_df['r2'].idxmax(), 'sensor_id']
            st.metric(
                "Mejor Sensor (R²)",
                f"{best_sensor}",
                f"R² = {sensor_metrics_df['r2'].max():.3f}"
            )
        
        # Gráfico de comparación entre sensores
        self._show_sensor_comparison_chart(sensor_metrics_df)
        
        # Mostrar tabla detallada
        with st.expander("📋 Métricas Detalladas por Sensor"):
            # Formatear fechas para mostrar
            display_df = sensor_metrics_df.copy()
            display_df['periodo_min'] = display_df['periodo_min'].dt.strftime('%Y-%m-%d')
            display_df['periodo_max'] = display_df['periodo_max'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df.style.format({
                    'rmse': '{:.2f}',
                    'r2': '{:.3f}',
                    'mae': '{:.2f}',
                    'no2_mean': '{:.2f}',
                    'no2_std': '{:.2f}',
                    'residuo_mean': '{:.3f}',
                    'residuo_std': '{:.2f}'
                }),
                use_container_width=True
            )
        
        # Selector para análisis individual detallado
        st.markdown("### 🔍 Análisis Detallado por Sensor Específico")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sensor_seleccionado = st.selectbox(
                "Selecciona sensor para análisis detallado:",
                sensores_test,
                key="global_xgb_analysis_sensor",
                help="Escoge un sensor para ver análisis temporal detallado y comparación de predicciones"
            )
        
        with col2:
            # Mostrar info del sensor seleccionado
            if sensor_seleccionado in sensor_predictions:
                sensor_info = sensor_predictions[sensor_seleccionado]
                st.metric(
                    f"RMSE {sensor_seleccionado}", 
                    f"{sensor_info['rmse']:.2f} µg/m³",
                    help="Error cuadrático medio para este sensor"
                )
        
        # Análisis detallado del sensor seleccionado
        if st.checkbox("📈 Mostrar Análisis Temporal Detallado", key="show_detailed_xgb_analysis"):
            if sensor_seleccionado in sensor_predictions:
                self._show_detailed_sensor_analysis(sensor_seleccionado, sensor_predictions[sensor_seleccionado])
            else:
                st.error(f"No hay datos disponibles para el sensor {sensor_seleccionado}")

    def _show_sensor_comparison_chart(self, sensor_metrics_df: pd.DataFrame):
        """Muestra gráfico de comparación entre sensores."""
        st.markdown("### 📊 Comparación de Rendimiento por Sensor")
        
        # Crear gráfico de barras comparativo
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE por sensor
        axes[0].bar(sensor_metrics_df['sensor_id'], sensor_metrics_df['rmse'], 
                   color='lightcoral', alpha=0.7)
        axes[0].set_title('RMSE por Sensor')
        axes[0].set_ylabel('RMSE (µg/m³)')
        axes[0].set_xlabel('Sensor ID')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # R² por sensor
        axes[1].bar(sensor_metrics_df['sensor_id'], sensor_metrics_df['r2'], 
                   color='lightblue', alpha=0.7)
        axes[1].set_title('R² por Sensor')
        axes[1].set_ylabel('R²')
        axes[1].set_xlabel('Sensor ID')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # MAE por sensor
        axes[2].bar(sensor_metrics_df['sensor_id'], sensor_metrics_df['mae'], 
                   color='lightgreen', alpha=0.7)
        axes[2].set_title('MAE por Sensor')
        axes[2].set_ylabel('MAE (µg/m³)')
        axes[2].set_xlabel('Sensor ID')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def _show_detailed_sensor_analysis(self, sensor_id: str, sensor_data: Dict):
        """Muestra análisis detallado para un sensor específico."""
        
        st.subheader(f"📊 Análisis Detallado - Sensor {sensor_id}")
        
        sensor_df = sensor_data['sensor_data']
        y_true = sensor_data['y_true']
        y_pred = sensor_data['y_pred']
        
        # Métricas del sensor
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{sensor_data['rmse']:.2f} µg/m³")
        with col2:
            st.metric("R²", f"{sensor_data['r2']:.3f}")
        with col3:
            st.metric("MAE", f"{sensor_data['mae']:.2f} µg/m³")
        with col4:
            # Calcular sesgo
            bias = np.mean(y_pred - y_true)
            st.metric("Sesgo", f"{bias:.2f} µg/m³")
        
        # Información del período
        st.markdown("### 📅 Información del Período de Evaluación")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Muestras", len(sensor_df))
        with col2:
            st.metric("Fecha Inicio", sensor_df['fecha'].min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Fecha Fin", sensor_df['fecha'].max().strftime('%Y-%m-%d'))
        
        # Análisis temporal específico del sensor
        st.markdown("### 📈 Análisis Temporal del Sensor")
        
        # Crear DataFrame para visualización temporal
        df_plot = sensor_df[['fecha', 'no2_value']].copy()
        df_plot['Predicción'] = y_pred
        df_plot['Residuo'] = y_true - y_pred
        df_plot = df_plot.set_index('fecha')
        
        # Controles de visualización
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "Rango de fechas:",
                value=(df_plot.index.min().date(), df_plot.index.max().date()),
                min_value=df_plot.index.min().date(),
                max_value=df_plot.index.max().date(),
                key=f"sensor_{sensor_id}_date_range"
            )
        
        with col2:
            granularity = st.selectbox(
                "Granularidad:",
                options=['Horaria', 'Media Diaria', 'Media Semanal'],
                index=1,  # Media Diaria por defecto
                key=f"sensor_{sensor_id}_granularity"
            )
        
        # Filtrar por fechas
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
        df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]
        
        if not df_filtered.empty:
            # Aplicar granularidad
            if granularity == 'Media Diaria':
                df_agg = df_filtered.resample('D').mean()
                title_suffix = '(Media Diaria)'
            elif granularity == 'Media Semanal':
                df_agg = df_filtered.resample('W-MON').mean()
                title_suffix = '(Media Semanal)'
            else:
                df_agg = df_filtered
                title_suffix = '(Horario)'
            
            # Gráfico de predicciones vs reales
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Predicciones vs Reales
            ax1.plot(df_agg.index, df_agg['no2_value'], label='Valor Real', 
                    alpha=0.8, linewidth=2, color='blue')
            ax1.plot(df_agg.index, df_agg['Predicción'], label='Predicción XGBoost', 
                    linestyle='--', alpha=0.8, linewidth=2, color='red')
            
            ax1.set_title(f'Predicciones vs Reales - Sensor {sensor_id} {title_suffix}')
            ax1.set_ylabel('Concentración NO₂ (µg/m³)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuos
            ax2.plot(df_agg.index, df_agg['Residuo'], alpha=0.8, linewidth=1.5, color='green')
            ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
            ax2.set_title(f'Residuos Temporales - Sensor {sensor_id} {title_suffix}')
            ax2.set_ylabel('Residuo (µg/m³)')
            ax2.set_xlabel('Fecha')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Formatear fechas en eje X
            if granularity != 'Horaria':
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            else:
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            
            plt.setp([ax1, ax2], xticklabels=[])  # Ocultar etiquetas del primer gráfico
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Análisis de distribución de errores para este sensor
        st.markdown("### 📊 Distribución de Errores del Sensor")
        
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de residuos
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(residuals, kde=True, ax=ax, bins=30)
            ax.set_title(f'Distribución de Residuos - Sensor {sensor_id}')
            ax.set_xlabel('Residuo (Real - Predicción) [µg/m³]')
            ax.set_ylabel('Frecuencia')
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Scatter plot predicción vs real
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Línea ideal (y=x)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Predicción Perfecta')
            
            ax.set_xlabel('Valor Real (µg/m³)')
            ax.set_ylabel('Predicción (µg/m³)')
            ax.set_title(f'Predicción vs Real - Sensor {sensor_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Estadísticas adicionales del sensor
        st.markdown("### 📈 Estadísticas Adicionales del Sensor")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Media NO₂ Real", f"{y_true.mean():.2f} µg/m³")
        with col2:
            st.metric("Std NO₂ Real", f"{y_true.std():.2f} µg/m³")
        with col3:
            st.metric("Media Residuos", f"{residuals.mean():.3f} µg/m³")
        with col4:
            st.metric("Std Residuos", f"{residuals.std():.2f} µg/m³")
        
        # Análisis por hora del día para este sensor
        if len(sensor_df) > 24:  # Solo si tenemos suficientes datos
            st.markdown("### 🕐 Análisis por Hora del Día")
            
            sensor_df_copy = sensor_df.copy()
            sensor_df_copy['hour'] = sensor_df_copy['fecha'].dt.hour
            sensor_df_copy['prediction'] = y_pred
            sensor_df_copy['residual'] = y_true - y_pred
            
            hourly_stats = sensor_df_copy.groupby('hour').agg({
                'no2_value': ['mean', 'std', 'count'],
                'prediction': ['mean', 'std'],
                'residual': ['mean', 'std']
            }).round(2)
            
            # Aplanar nombres de columnas
            hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Valores por hora
            hours = hourly_stats.index
            ax1.plot(hours, hourly_stats['no2_value_mean'], 'o-', label='Real', linewidth=2)
            ax1.plot(hours, hourly_stats['prediction_mean'], 's--', label='Predicción', linewidth=2)
            ax1.fill_between(hours, 
                           hourly_stats['no2_value_mean'] - hourly_stats['no2_value_std'],
                           hourly_stats['no2_value_mean'] + hourly_stats['no2_value_std'],
                           alpha=0.2, label='±1 Std Real')
            ax1.set_xlabel('Hora del Día')
            ax1.set_ylabel('NO₂ Promedio (µg/m³)')
            ax1.set_title(f'Patrón Horario - Sensor {sensor_id}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(0, 24, 2))
            
            # Residuos por hora
            ax2.plot(hours, hourly_stats['residual_mean'], 'o-', color='green', linewidth=2)
            ax2.fill_between(hours,
                           hourly_stats['residual_mean'] - hourly_stats['residual_std'],
                           hourly_stats['residual_mean'] + hourly_stats['residual_std'],
                           alpha=0.2, color='green')
            ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Hora del Día')
            ax2.set_ylabel('Residuo Promedio (µg/m³)')
            ax2.set_title(f'Errores por Hora - Sensor {sensor_id}')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(0, 24, 2))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Mostrar tabla de estadísticas horarias
            with st.expander(f"📋 Estadísticas Horarias Detalladas - Sensor {sensor_id}"):
                st.dataframe(hourly_stats, use_container_width=True)


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
        - Configuración completa de variables y preprocesamiento
        
        **🌍 Modelos Globales:**
        - Un modelo entrenado con múltiples sensores
        - Aprende patrones generales transferibles
        - Ideal para nowcasting en nuevas ubicaciones
        - Análisis de transferibilidad entre sensores
        
        **🔬 Características Técnicas:**
        - Selección granular de variables por categorías
        - Múltiples métodos de detección de outliers
        - Preprocesamiento temporal avanzado
        - Análisis detallado de rendimiento
        - Interfaz unificada y reutilizable
        
        **🚀 Aplicaciones:**
        - Comparar rendimiento individual vs global
        - Validar transferibilidad de modelos
        - Seleccionar estrategia óptima para nowcasting
        - Análisis de importancia de variables
        """)


def xgboost_unified_page():
    """Página principal del módulo XGBoost unificado."""
    
    # Panel de información
    show_info_panel()
    
    # Inicializar trainer
    unified_trainer = XGBoostUnifiedTrainer()
    
    # Cargar datos
    if not st.session_state.xgb_unified_data_loaded:
        if st.button("🚀 Cargar Dataset Completo", type="primary"):
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
        key_prefix = "individual"
    else:
        st.session_state.xgb_unified_mode = 'global'
        key_prefix = "global"
    
    st.divider()
    
    # Mostrar configuración
    config = unified_trainer.show_configuration_panel(st.session_state.xgb_unified_mode, key_prefix)
    
    # Validar configuración mínima
    if st.session_state.xgb_unified_mode == 'global':
        if not config.get('sensores_train') or not config.get('sensores_test'):
            return
    
    # Selección de variables
    selected_features = unified_trainer.show_variable_selection(key_prefix)
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable para continuar.")
        return
    
    # Mostrar resumen de configuración
    unified_trainer.show_configuration_summary(config, selected_features)
    
    # Preparar datos
    data_prep = unified_trainer.prepare_data(config, selected_features, st.session_state.xgb_unified_mode)
    
    if not data_prep:
        st.error("❌ Error en la preparación de datos.")
        return
    
    # Verificar si existe modelo (solo para individual)
    model_exists = False
    if st.session_state.xgb_unified_mode == 'individual':
        model_filename = f'data/models/xgboost_model_{config["sensor"]}_{config["outlier_method"]}_{config["preprocessing"]}.pkl'
        model_exists = os.path.exists(model_filename)
    
    # Mostrar información de datos
    st.subheader("📊 Información del Conjunto de Datos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Muestras entrenamiento", len(data_prep['train_df']))
    with col2:
        st.metric("Muestras evaluación", len(data_prep['test_df']))
    with col3:
        st.metric("Variables seleccionadas", len(selected_features))
    with col4:
        st.metric("Outliers eliminados", data_prep['outliers_removed'])
    
    # Crear clave única para la configuración
    if st.session_state.xgb_unified_mode == 'individual':
        config_key = f"individual_{config['sensor']}_{config['outlier_method']}_{config['preprocessing']}_{len(selected_features)}"
    else:
        config_key = f"global_{len(config['sensores_train'])}_{len(config['sensores_test'])}_{config['outlier_method']}_{config['preprocessing']}_{len(selected_features)}"
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if model_exists and st.session_state.xgb_unified_mode == 'individual':
            analyze_button = st.button("🔍 Analizar Modelo Existente", type="primary", key=f"analyze_{config_key}")
        else:
            analyze_button = False
            if st.session_state.xgb_unified_mode == 'individual':
                st.info("💡 No existe un modelo entrenado con esta configuración")
    
    with col2:
        train_button = st.button("🚀 Entrenar Nuevo Modelo", type="secondary", key=f"train_{config_key}")
    
    # Ejecutar análisis (solo individual)
    if analyze_button and model_exists:
        with st.spinner("Cargando y analizando modelo..."):
            model_info = unified_trainer.load_model(model_filename)
            
            if model_info:
                # Preparar datos de prueba
                X_test = data_prep['test_df'][selected_features].copy()
                y_test = data_prep['test_df']['no2_value'].copy()
                
                # Escalar datos de prueba
                for feature in selected_features:
                    if feature in model_info['scaler_dict']:
                        X_test[feature] = model_info['scaler_dict'][feature].transform(X_test[[feature]])
                
                # Evaluar modelo
                metrics = unified_trainer.evaluate_model(model_info['model'], X_test, y_test, model_info['scaler_target'])
                
                # Crear resultados
                results = {
                    'model': model_info['model'],
                    'metrics': metrics,
                    'test_df': data_prep['test_df'],
                    'scaler_dict': model_info['scaler_dict'],
                    'scaler_target': model_info['scaler_target'],
                    'selected_features': selected_features,
                    'config': config
                }
                
                unified_trainer.show_analysis_interface(results, config_key)
    
    # Ejecutar entrenamiento
    if train_button:
        results = unified_trainer.train_model(data_prep, selected_features, config)
        
        if results:
            unified_trainer.show_analysis_interface(results, config_key)
    
    # Mostrar análisis si ya existe en session_state
    elif config_key in st.session_state.xgb_unified_analysis_data:
        results = st.session_state.xgb_unified_analysis_data[config_key]
        unified_trainer.show_analysis_interface(results, config_key)


if __name__ == "__main__":
    xgboost_unified_page() 