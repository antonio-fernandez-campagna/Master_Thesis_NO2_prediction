"""
Módulo para entrenamiento de modelos XGBoost GLOBALES.

Este módulo entrena un único modelo XGBoost utilizando TODOS los sensores,
ideal para nowcasting en ubicaciones sin sensores históricos.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import warnings
import statsmodels.api as sm
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# Importar configuraciones del módulo original
from xgboost_training import (
    OUTLIER_METHODS, PREPROCESSING_OPTIONS, VARIABLE_CATEGORIES, 
    VARIABLE_METADATA, COLUMNS_FOR_OUTLIERS,
    show_model_metrics, show_residual_analysis, show_feature_importance,
    show_temporal_predictions, show_residuals_over_time
)


class XGBoostGlobalTrainer:
    """Clase para entrenamiento de modelos XGBoost globales (todos los sensores)."""
    
    def __init__(self):
        self.df_master = None
        self.model = None
        self.scaler_dict = {}
        self.scaler_target = None
        self.label_encoders = {}  # Para codificar IDs de sensores
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'xgboost_global_data_loaded' not in st.session_state:
            st.session_state.xgboost_global_data_loaded = False
        if 'xgboost_global_model_trained' not in st.session_state:
            st.session_state.xgboost_global_model_trained = False
        if 'xgboost_global_config' not in st.session_state:
            st.session_state.xgboost_global_config = {}
    
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
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables cíclicas para capturar patrones temporales."""
        df = df.copy()

        # Crear variables temporales base
        df['day_of_week'] = df['fecha'].dt.dayofweek
        df['day_of_year'] = df['fecha'].dt.dayofyear
        df['month'] = df['fecha'].dt.month
        df['year'] = df['fecha'].dt.year
        df['weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['hour'] = df['fecha'].dt.hour
        df['day'] = df['fecha'].dt.day
        
        # Crear variable estacional numérica
        df['season'] = df['month'].apply(
            lambda x: 0 if x in [12,1,2] else 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3
        )
        
        # Variables cíclicas temporales
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
        
        return df
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte unidades meteorológicas a formatos más interpretables."""
        df = df.copy()
        
        # Temperatura (K -> °C)
        if 'd2m' in df.columns:
            df['d2m'] = df['d2m'] - 273.15
        if 't2m' in df.columns:
            df['t2m'] = df['t2m'] - 273.15
        
        # Radiación (J/m² -> W/m²)
        if 'ssr' in df.columns:
            df['ssr'] = df['ssr'] / 3600
        if 'ssrd' in df.columns:
            df['ssrd'] = df['ssrd'] / 3600
        
        # Viento (m/s -> km/h) y calcular magnitud y dirección
        u10_mps = df.get('u10', pd.Series(0, index=df.index))
        v10_mps = df.get('v10', pd.Series(0, index=df.index))
        
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(u10_mps**2 + v10_mps**2) * 3.6  # km/h
            df['wind_direction'] = (270 - np.arctan2(v10_mps, u10_mps) * 180/np.pi) % 360
            df['u10'] = df['u10'] * 3.6  # km/h
            df['v10'] = df['v10'] * 3.6  # km/h
        
        # Presión (Pa -> hPa)
        if 'sp' in df.columns:
            df['sp'] = df['sp'] / 100
        
        # Precipitación (m -> mm)
        if 'tp' in df.columns:
            df['tp'] = df['tp'] * 1000
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Elimina outliers según el método especificado - SOLO en datos de entrenamiento."""
        if method == 'none':
            return df
        
        df_filtered = df.copy()
        
        if method == 'iqr':
            for col in COLUMNS_FOR_OUTLIERS:
                if col in df_filtered.columns:
                    Q1 = df_filtered[col].quantile(0.25)
                    Q3 = df_filtered[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
        
        elif method == 'zscore':
            for col in COLUMNS_FOR_OUTLIERS:
                if col in df_filtered.columns:
                    z_scores = zscore(df_filtered[col], nan_policy='omit')
                    df_filtered = df_filtered[np.abs(z_scores) < 3.0]
        
        elif method == 'quantiles':
            for col in COLUMNS_FOR_OUTLIERS:
                if col in df_filtered.columns:
                    lower = df_filtered[col].quantile(0.01)
                    upper = df_filtered[col].quantile(0.99)
                    df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
        
        return df_filtered
    
    def split_data(self, df: pd.DataFrame, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide los datos en entrenamiento y prueba basado en fecha."""
        train = df[df['fecha'] < split_date].copy()
        test = df[df['fecha'] >= split_date].copy()
        return train, test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Escala las variables predictoras."""
        scaler_dict = {}
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        for feature in features:
            if feature in X_train.columns and pd.api.types.is_numeric_dtype(X_train[feature]):
                scaler = StandardScaler()
                X_train_scaled[feature] = scaler.fit_transform(X_train[[feature]]).flatten()
                X_test_scaled[feature] = scaler.transform(X_test[[feature]]).flatten()
                scaler_dict[feature] = scaler
        
        return X_train_scaled, X_test_scaled, scaler_dict
    
    def scale_target(self, y_train: pd.Series) -> Tuple[np.ndarray, StandardScaler]:
        """Escala la variable objetivo."""
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        return y_scaled, scaler
    
    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> xgb.XGBRegressor:
        """Entrena el modelo XGBoost global."""
        
        # Filtrar solo variables numéricas
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = X_train[numeric_features]
        X_val_numeric = X_val[numeric_features]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse',
            early_stopping_rounds=50
        )
        
        # Configurar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        eval_set = [(X_val_numeric, y_val)]
        
        status_text.text("Entrenando modelo XGBoost GLOBAL...")
        
        model.fit(
            X_train_numeric, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        progress_bar.progress(100)
        status_text.success("Entrenamiento GLOBAL completado.")
        
        return model
    
    def evaluate_model(self, model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series, scaler_target: StandardScaler) -> Dict:
        """Evalúa el modelo y devuelve métricas."""
        
        # Filtrar solo variables numéricas
        numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
        X_test_numeric = X_test[numeric_features]
        
        # Predicciones escaladas
        y_pred_scaled = model.predict(X_test_numeric)
        
        # Desescalar predicciones
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'y_pred': y_pred,
            'y_pred_scaled': y_pred_scaled
        }
    
    def evaluate_by_sensor(self, model: xgb.XGBRegressor, test_df: pd.DataFrame, 
                          selected_features: List[str], scaler_target: StandardScaler) -> pd.DataFrame:
        """Evalúa el modelo por sensor individual para análisis de transferibilidad."""
        
        sensor_metrics = []
        
        for sensor_id in sorted(test_df['id_no2'].unique()):
            sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
            
            if len(sensor_data) == 0:
                continue
            
            X_sensor = sensor_data[selected_features]
            y_sensor = sensor_data['no2_value']
            
            # Filtrar solo variables numéricas
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
        
        return pd.DataFrame(sensor_metrics)
    
    def save_model(self, model: xgb.XGBRegressor, feature_names: List[str], scaler_dict: Dict, 
                   scaler_target: StandardScaler, label_encoders: Dict, outlier_method: str, 
                   preprocessing: str) -> str:
        """Guarda el modelo global y sus metadatos."""
        
        model_info = {
            'model': model,
            'feature_names': feature_names,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'label_encoders': label_encoders,
            'variable_metadata': VARIABLE_METADATA,
            'outlier_method': outlier_method,
            'preprocessing': preprocessing,
            'model_type': 'xgboost_global',
            'n_sensors_trained': len(label_encoders.get('sensor_encoder', LabelEncoder()).classes_) if 'sensor_encoder' in label_encoders else 0
        }
        
        model_dir = 'data/models'
        os.makedirs(model_dir, exist_ok=True)
        filename = f'{model_dir}/xgboost_global_model_{outlier_method}_{preprocessing}.pkl'
        
        joblib.dump(model_info, filename)
        return filename
    
    def load_model(self, filepath: str) -> Optional[Dict]:
        """Carga un modelo global guardado."""
        try:
            return joblib.load(filepath)
        except Exception as e:
            st.error(f"Error al cargar modelo global: {str(e)}")
            return None


def show_sensor_comparison(sensor_metrics: pd.DataFrame):
    """Muestra comparación de rendimiento por sensor."""
    st.subheader("📊 Rendimiento por Sensor Individual")
    
    # Métricas resumidas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "RMSE Promedio",
            f"{sensor_metrics['rmse'].mean():.2f} µg/m³",
            f"±{sensor_metrics['rmse'].std():.2f}"
        )
    
    with col2:
        st.metric(
            "R² Promedio", 
            f"{sensor_metrics['r2'].mean():.3f}",
            f"±{sensor_metrics['r2'].std():.3f}"
        )
    
    with col3:
        st.metric(
            "MAE Promedio",
            f"{sensor_metrics['mae'].mean():.2f} µg/m³",
            f"±{sensor_metrics['mae'].std():.2f}"
        )
    
    # Gráfico de barras con métricas por sensor
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE por sensor
    axes[0].bar(sensor_metrics['sensor_id'].astype(str), sensor_metrics['rmse'])
    axes[0].set_title('RMSE por Sensor')
    axes[0].set_ylabel('RMSE (µg/m³)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² por sensor
    axes[1].bar(sensor_metrics['sensor_id'].astype(str), sensor_metrics['r2'])
    axes[1].set_title('R² por Sensor')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAE por sensor
    axes[2].bar(sensor_metrics['sensor_id'].astype(str), sensor_metrics['mae'])
    axes[2].set_title('MAE por Sensor')
    axes[2].set_ylabel('MAE (µg/m³)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Tabla detallada
    with st.expander("📋 Métricas Detalladas por Sensor"):
        st.dataframe(
            sensor_metrics.style.format({
                'rmse': '{:.2f}',
                'r2': '{:.3f}',
                'mae': '{:.2f}',
                'no2_mean': '{:.2f}',
                'no2_std': '{:.2f}'
            }),
            use_container_width=True
        )


def show_global_info_panel():
    """Muestra panel de información sobre XGBoost Global."""
    with st.expander("ℹ️ Acerca del Modelo XGBoost Global", expanded=True):
        st.markdown("""
        **🌍 Modelo XGBoost Global (Multi-Sensor)**
        
        **¿Qué es diferente?**
        - **Entrena con TODOS los sensores** en un solo modelo
        - **Usa variables NATURALES** sin artificios
        - **Aprende patrones generales** aplicables a nuevas ubicaciones
        
        **🎯 Ventajas para Nowcasting:**
        - ✅ **Transferibilidad**: Funciona en ciudades sin sensores históricos
        - ✅ **Más datos**: ~736k registros vs ~57k por sensor individual
        - ✅ **Robustez**: Maneja mejor condiciones extremas o atípicas
        - ✅ **Patrones universales**: Relaciones meteorología/tráfico → NO₂
        
        **🔬 Variables utilizadas (TODAS naturalmente transferibles):**
        - **Meteorológicas**: Temperatura, viento, presión, radiación, etc.
        - **Tráfico**: Intensidad, velocidad, carga, ocupación
        - **Temporales**: Hora, día, mes, estación (cíclicas o lineales)
        
        **✅ Simplicidad = Transferibilidad:**
        - XGBoost aprende interacciones automáticamente
        - Variables numéricas conservan toda la información
        - No hay categorizaciones artificiales
        - No hay características específicas de Madrid
        
        **📊 Evaluación multi-nivel:**
        1. **Métricas globales**: Rendimiento general del modelo
        2. **Métricas por sensor**: Transferibilidad a cada ubicación
        3. **Análisis temporal**: Consistencia a lo largo del tiempo
        
        **🚀 Aplicación práctica:**
        Para usar en una ciudad nueva sin sensores:
        1. Recopilar datos meteorológicos y de tráfico
        2. Aplicar el modelo directamente (sin preprocesamiento especial)
        3. El modelo usa patrones UNIVERSALES aprendidos
        """)


def xgboost_global_training_page():
    """Función principal del módulo de entrenamiento XGBoost Global."""
    
    # Inicializar trainer
    trainer = XGBoostGlobalTrainer()
    
    # Panel de información
    show_global_info_panel()
    
    # Cargar datos
    if not st.session_state.xgboost_global_data_loaded:
        if st.button("🌍 Cargar datos para entrenamiento GLOBAL", type="primary"):
            with st.spinner("Cargando datos de TODOS los sensores..."):
                trainer.df_master = trainer.load_data()
                if not trainer.df_master.empty:
                    st.session_state.xgboost_global_data_loaded = True
                    st.success("Datos de todos los sensores cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    trainer.df_master = trainer.load_data()
    
    if trainer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Mostrar estadísticas del dataset global
    st.header("🌍 Dataset Global Multi-Sensor")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total registros", f"{len(trainer.df_master):,}")
    with col2:
        st.metric("Sensores NO₂", trainer.df_master['id_no2'].nunique())
    with col3:
        st.metric("Sensores tráfico", trainer.df_master['id_trafico'].nunique())
    with col4:
        periodo_años = (trainer.df_master['fecha'].max() - trainer.df_master['fecha'].min()).days / 365.25
        st.metric("Período", f"{periodo_años:.1f} años")
    
    # Configuración del modelo
    st.header("⚙️ Configuración del Modelo Global")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fechas disponibles
        fecha_min = trainer.df_master["fecha"].min().date()
        fecha_max = trainer.df_master["fecha"].max().date()
        
        # Fecha de división
        fecha_division = st.date_input(
            "Fecha de división (entrenamiento/evaluación)",
            value=pd.to_datetime('2024-01-01').date(),
            min_value=fecha_min,
            max_value=fecha_max,
            help="Los datos anteriores se usan para entrenamiento, posteriores para evaluación"
        )
    
    with col2:
        # Método de filtrado de outliers
        outlier_method = st.selectbox(
            "Método de filtrado de outliers",
            options=list(OUTLIER_METHODS.keys()),
            format_func=lambda x: OUTLIER_METHODS[x],
            index=0  # iqr por defecto para modelo global
        )
        
        # Preprocesamiento
        preprocessing = st.selectbox(
            "Preprocesamiento temporal",
            options=list(PREPROCESSING_OPTIONS.keys()),
            format_func=lambda x: PREPROCESSING_OPTIONS[x],
            index=0  # sin_cos por defecto para modelo global
        )
    
    # Selección de variables (automática para modelo global)
    st.subheader("🔧 Variables del Modelo Global")
    
    # Variables automáticas para modelo global
    selected_features = []
    
    # No necesitamos variables espaciales artificiales
    # Las variables naturales YA SON transferibles
    
    # Variables temporales (según preprocesamiento)
    if preprocessing == 'sin_cos':
        temporal_vars = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
            'weekend', 'season_sin', 'season_cos'
        ]
    else:
        temporal_vars = ['hour', 'month', 'day_of_week', 'day_of_year', 'weekend']
    
    selected_features.extend(temporal_vars)
    
    # Variables de tráfico
    traffic_vars = ['intensidad', 'carga', 'ocupacion', 'vmed']
    selected_features.extend(traffic_vars)
    
    # Variables meteorológicas
    meteo_vars = ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']
    selected_features.extend(meteo_vars)
    
    # Mostrar selección
    with st.expander("📋 Variables Seleccionadas Automáticamente"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Temporales:**")
            for var in temporal_vars[:7]:  # Primeras 7
                st.write(f"• {var}")
        with col2:
            st.write("**Tráfico:**")
            for var in traffic_vars:
                st.write(f"• {var}")
        with col3:
            st.write("**Meteorológicas:**")
            for var in meteo_vars[:7]:  # Primeras 7
                st.write(f"• {var}")
    
    # Crear clave única para la configuración actual
    config_key = f"global_{outlier_method}_{preprocessing}_{len(selected_features)}"
    
    # Actualizar configuración en session_state
    st.session_state.xgboost_global_config = {
        'tipo': 'Modelo Global (Todos los Sensores)',
        'fecha_division': fecha_division.strftime('%Y-%m-%d'),
        'outlier_method': OUTLIER_METHODS[outlier_method],
        'preprocessing': PREPROCESSING_OPTIONS[preprocessing],
        'num_variables': len(selected_features),
        'config_key': config_key
    }
    
    # Mostrar resumen de configuración
    with st.expander("📋 Resumen de Configuración Global"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Tipo:** Modelo Global")
            st.write(f"**Variables:** {len(selected_features)}")
        with col2:
            st.write(f"**Outliers:** {OUTLIER_METHODS[outlier_method]}")
            st.write(f"**Preprocesamiento:** {PREPROCESSING_OPTIONS[preprocessing]}")
        with col3:
            st.write(f"**Fecha división:** {fecha_division}")
            st.write(f"**Sensores:** {trainer.df_master['id_no2'].nunique()}")
    
    # Preparar datos
    with st.spinner("Preparando datos globales..."):
        # Aplicar transformaciones básicas
        df_processed = trainer.df_master.copy()
        
        st.write("📊 Datos originales:", len(df_processed))
        
        # Crear variables cíclicas si se requiere
        if preprocessing == 'sin_cos':
            df_processed = trainer.create_cyclical_features(df_processed)
        
        # Convertir unidades
        df_processed = trainer.convert_units(df_processed)
        
        # Dividir datos ANTES de eliminar outliers
        fecha_division_dt = pd.to_datetime(fecha_division)
        train_df, test_df = trainer.split_data(df_processed, fecha_division_dt)
        
        st.write("📅 Datos entrenamiento (antes outliers):", len(train_df))
        st.write("📅 Datos evaluación:", len(test_df))
        
        # Eliminar outliers SOLO del conjunto de entrenamiento
        if outlier_method != 'none':
            train_df = trainer.remove_outliers(train_df, outlier_method)
            st.write("🔍 Datos entrenamiento (después outliers):", len(train_df))
            outliers_removed = len(df_processed[df_processed['fecha'] < fecha_division_dt]) - len(train_df)
            st.write(f"❌ Outliers eliminados: {outliers_removed}")
        else:
            outliers_removed = 0
        
        if train_df.empty or test_df.empty:
            st.error("No hay suficientes datos para entrenamiento o evaluación.")
            return
    
    # Verificar si existe modelo entrenado
    model_filename = f'data/models/xgboost_global_model_{outlier_method}_{preprocessing}.pkl'
    model_exists = os.path.exists(model_filename)
    
    # Mostrar información de datos
    st.subheader("📊 Información del Dataset Global")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Muestras entrenamiento", f"{len(train_df):,}")
    with col2:
        st.metric("Muestras evaluación", f"{len(test_df):,}")
    with col3:
        st.metric("Variables totales", len(selected_features))
    with col4:
        st.metric("Outliers eliminados", f"{outliers_removed:,}")
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if model_exists:
            analyze_button = st.button("🔍 Analizar Modelo Global Existente", type="primary")
        else:
            analyze_button = False
            st.info("No existe un modelo XGBoost Global entrenado con esta configuración")
    
    with col2:
        train_button = st.button("🚀 Entrenar Nuevo Modelo Global", type="secondary")
    
    # Inicializar variables de estado para el análisis si no existen
    if 'xgboost_global_analysis_data' not in st.session_state:
        st.session_state.xgboost_global_analysis_data = {}
    
    # Ejecutar análisis o entrenamiento
    if analyze_button and model_exists:
        with st.spinner("Cargando y analizando modelo XGBoost Global..."):
            model_info = trainer.load_model(model_filename)
            
            if model_info:
                model = model_info['model']
                feature_names = model_info['feature_names']
                scaler_dict = model_info['scaler_dict']
                scaler_target = model_info['scaler_target']
                trainer.label_encoders = model_info['label_encoders']
                
                # Preparar datos de prueba
                X_test = test_df[selected_features].copy()
                y_test = test_df['no2_value'].copy()
                
                # Escalar datos de prueba
                for feature in selected_features:
                    if feature in scaler_dict:
                        X_test[feature] = scaler_dict[feature].transform(X_test[[feature]])
                
                # Evaluar modelo
                metrics = trainer.evaluate_model(model, X_test, y_test, scaler_target)
                
                # Evaluar por sensor
                sensor_metrics = trainer.evaluate_by_sensor(model, test_df, selected_features, scaler_target)
                
                # Guardar datos del análisis en session_state
                st.session_state.xgboost_global_analysis_data[config_key] = {
                    'model': model,
                    'feature_names': feature_names,
                    'scaler_dict': scaler_dict,
                    'scaler_target': scaler_target,
                    'test_df': test_df,
                    'y_test': y_test,
                    'metrics': metrics,
                    'sensor_metrics': sensor_metrics,
                    'model_info': model_info
                }
    
    # Ejecutar entrenamiento
    if train_button:
        with st.spinner("Entrenando modelo XGBoost Global..."):
            # Preparar datos para entrenamiento
            X_train = train_df[selected_features].copy()
            y_train = train_df['no2_value'].copy()
            X_test = test_df[selected_features].copy()
            y_test = test_df['no2_value'].copy()
            
            # Eliminar filas con NaN
            train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            test_df_clean = test_df[test_mask]
            
            if X_train.empty or X_test.empty:
                st.error("No hay datos válidos después de la limpieza.")
                return
            
            # Escalar datos
            X_train_scaled, X_test_scaled, scaler_dict = trainer.scale_features(X_train, X_test, selected_features)
            y_train_scaled, scaler_target = trainer.scale_target(y_train)
            y_test_scaled, _ = trainer.scale_target(y_test)
            
            # Entrenar modelo
            model = trainer.train_xgboost_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
            
            # Guardar modelo
            model_path = trainer.save_model(
                model, selected_features, scaler_dict, scaler_target,
                trainer.label_encoders, outlier_method, preprocessing
            )
            
            st.success(f"Modelo XGBoost Global entrenado y guardado en: {model_path}")
            
            # Evaluar modelo
            metrics = trainer.evaluate_model(model, X_test_scaled, y_test, scaler_target)
            sensor_metrics = trainer.evaluate_by_sensor(model, test_df_clean, selected_features, scaler_target)
            
            # Guardar datos del análisis en session_state
            st.session_state.xgboost_global_analysis_data[config_key] = {
                'model': model,
                'feature_names': selected_features,
                'scaler_dict': scaler_dict,
                'scaler_target': scaler_target,
                'test_df': test_df_clean,
                'y_test': y_test,
                'metrics': metrics,
                'sensor_metrics': sensor_metrics,
                'model_info': {
                    'model': model,
                    'feature_names': selected_features,
                    'scaler_dict': scaler_dict,
                    'scaler_target': scaler_target,
                    'label_encoders': trainer.label_encoders
                }
            }
    
    # Mostrar análisis si existen datos
    if config_key in st.session_state.xgboost_global_analysis_data:
        analysis_data = st.session_state.xgboost_global_analysis_data[config_key]
        
        st.header("📊 Análisis del Modelo XGBoost Global")
        
        # Inicializar el estado del análisis si no existe
        if 'xgboost_global_analysis_tab' not in st.session_state:
            st.session_state.xgboost_global_analysis_tab = 0
        
        # Usar radio buttons para evitar problemas con tabs y reruns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Métricas Globales", use_container_width=True, type="primary" if st.session_state.xgboost_global_analysis_tab == 0 else "secondary"):
                st.session_state.xgboost_global_analysis_tab = 0
                st.rerun()
        
        with col2:
            if st.button("🎯 Por Sensor", use_container_width=True, type="primary" if st.session_state.xgboost_global_analysis_tab == 1 else "secondary"):
                st.session_state.xgboost_global_analysis_tab = 1
                st.rerun()
        
        with col3:
            if st.button("📈 Predicciones", use_container_width=True, type="primary" if st.session_state.xgboost_global_analysis_tab == 2 else "secondary"):
                st.session_state.xgboost_global_analysis_tab = 2
                st.rerun()
        
        with col4:
            if st.button("🔍 Importancia", use_container_width=True, type="primary" if st.session_state.xgboost_global_analysis_tab == 3 else "secondary"):
                st.session_state.xgboost_global_analysis_tab = 3
                st.rerun()
        
        st.divider()
        
        # Mostrar contenido según la pestaña seleccionada
        if st.session_state.xgboost_global_analysis_tab == 0:
            st.subheader("📊 Métricas Globales")
            show_model_metrics(analysis_data['metrics'])
            st.divider()
            show_residual_analysis(analysis_data['y_test'], analysis_data['metrics']['y_pred'])
        
        elif st.session_state.xgboost_global_analysis_tab == 1:
            show_sensor_comparison(analysis_data['sensor_metrics'])
        
        elif st.session_state.xgboost_global_analysis_tab == 2:
            show_temporal_predictions(analysis_data['test_df'], analysis_data['metrics']['y_pred'], f"global_{config_key}")
            st.divider()
            show_residuals_over_time(analysis_data['test_df'], analysis_data['metrics']['y_pred'], f"global_{config_key}")
        
        elif st.session_state.xgboost_global_analysis_tab == 3:
            st.subheader("🎯 Importancia de Variables Global")
            show_feature_importance(analysis_data['model'], analysis_data['feature_names'])


if __name__ == "__main__":
    xgboost_global_training_page() 