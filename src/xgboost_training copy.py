"""
Módulo refactorizado para entrenamiento y análisis de modelos XGBoost.

Este módulo proporciona una interfaz integrada para entrenar y analizar modelos XGBoost
para predecir niveles de NO2 basándose en variables de tráfico, meteorológicas y temporales.
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
import statsmodels.api as sm
from scipy.stats import zscore

warnings.filterwarnings('ignore')


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

OUTLIER_METHODS = {
    'iqr': 'Rango Intercuartílico (IQR)',
    'zscore': 'Z-Score (Desviación Estándar)',
    'quantiles': 'Percentiles Extremos',
    'none': 'Sin filtrado'
}

PREPROCESSING_OPTIONS = {
    'sin_cos': 'Variables Cíclicas (Sin/Cos)',
    'none': 'Sin preprocesamiento'
}

VARIABLE_CATEGORIES = {
    "Variables Temporales": [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
        'weekend', 'season_sin', 'season_cos', 'hour', 'month', 'day_of_week', 'day_of_year'
    ],
    "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
    "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']
}

VARIABLE_METADATA = {
    'd2m': {'name': 'Punto de Rocío', 'unit': '°C', 'typical_range': (-10, 30)},
    't2m': {'name': 'Temperatura', 'unit': '°C', 'typical_range': (-5, 40)},
    'ssr': {'name': 'Radiación Solar Neta', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'ssrd': {'name': 'Radiación Solar Descendente', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'u10': {'name': 'Viento U 10m', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'v10': {'name': 'Viento V 10m', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'sp': {'name': 'Presión Superficial', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'tp': {'name': 'Precipitación Total', 'unit': 'mm', 'typical_range': (0, 50)},
    'intensidad': {'name': 'Intensidad de Tráfico', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'carga': {'name': 'Carga de Tráfico', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion': {'name': 'Ocupación Vial', 'unit': '%', 'typical_range': (0, 100)},
    'vmed': {'name': 'Velocidad Media', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_speed': {'name': 'Velocidad del Viento', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_direction': {'name': 'Dirección del Viento', 'unit': '°', 'typical_range': (0, 360)}
}

COLUMNS_FOR_OUTLIERS = [
    'no2_value'
]


# ==================== CLASE PRINCIPAL ====================

class XGBoostTrainer:
    """Clase principal para entrenamiento y análisis de modelos XGBoost."""
    
    def __init__(self):
        self.df_master = None
        self.model = None
        self.scaler_dict = {}
        self.scaler_target = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'xgboost_data_loaded' not in st.session_state:
            st.session_state.xgboost_data_loaded = False
        if 'xgboost_model_trained' not in st.session_state:
            st.session_state.xgboost_model_trained = False
        if 'xgboost_config' not in st.session_state:
            st.session_state.xgboost_config = {}
    
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
        
        # Crear variable estacional numérica (0-3: winter, spring, summer, autumn)
        df['season'] = df['month'].apply(
            lambda x: 0 if x in [12,1,2] else 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3
        )
        
        # Variables cíclicas temporales básicas
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Variables cíclicas adicionales más específicas
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
        """Elimina outliers según el método especificado."""
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
        """Entrena el modelo XGBoost."""
        
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
        
        status_text.text("Entrenando modelo XGBoost...")
        
        model.fit(
            X_train_numeric, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        progress_bar.progress(100)
        status_text.success("Entrenamiento completado.")
        
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
    
    def save_model(self, model: xgb.XGBRegressor, feature_names: List[str], scaler_dict: Dict, 
                   scaler_target: StandardScaler, sensor_id: str, outlier_method: str, 
                   preprocessing: str) -> str:
        """Guarda el modelo y sus metadatos."""
        
        model_info = {
            'model': model,
            'feature_names': feature_names,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'variable_metadata': VARIABLE_METADATA,
            'sensor_id': sensor_id,
            'outlier_method': outlier_method,
            'preprocessing': preprocessing,
            'model_type': 'xgboost'
        }
        
        model_dir = 'data/models'
        os.makedirs(model_dir, exist_ok=True)
        filename = f'{model_dir}/xgboost_model_{sensor_id}_{outlier_method}_{preprocessing}.pkl'
        
        joblib.dump(model_info, filename)
        return filename
    
    def load_model(self, filepath: str) -> Optional[Dict]:
        """Carga un modelo guardado."""
        try:
            return joblib.load(filepath)
        except Exception as e:
            st.error(f"Error al cargar modelo: {str(e)}")
            return None


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_model_metrics(metrics: Dict):
    """Muestra las métricas del modelo en formato de tarjetas."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="RMSE",
            value=f"{metrics['rmse']:.2f} µg/m³",
            help="Error cuadrático medio (menor es mejor)"
        )
    
    with col2:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.3f}",
            help="Coeficiente de determinación (cercano a 1 es mejor)"
        )
    
    with col3:
        st.metric(
            label="MAE",
            value=f"{metrics['mae']:.2f} µg/m³",
            help="Error absoluto medio (menor es mejor)"
        )


def show_residual_analysis(y_test: pd.Series, y_pred: np.ndarray):
    """Muestra análisis de residuos."""
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title('Distribución de Residuos')
        ax.set_xlabel('Residuo (Real - Predicción) [µg/m³]')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.qqplot(residuals, line='45', ax=ax, fit=True)
        ax.set_title('Q-Q Plot de Residuos')
        st.pyplot(fig)
        plt.close()
    
    # Estadísticas de residuos
    st.subheader("📊 Estadísticas de Residuos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Media", f"{residuals.mean():.3f}")
    with col2:
        st.metric("Desv. Estándar", f"{residuals.std():.3f}")
    with col3:
        st.metric("Sesgo", f"{residuals.skew():.3f}")
    with col4:
        st.metric("Curtosis", f"{residuals.kurtosis():.3f}")


def show_feature_importance(model: xgb.XGBRegressor, feature_names: List[str]):
    """Muestra la importancia de las variables."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Crear DataFrame para mejor manejo
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Gráfico de barras
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Importancia de Variables (XGBoost)')
        ax.set_xlabel('Importancia Relativa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Tabla de importancias
        with st.expander("📋 Tabla de Importancias"):
            st.dataframe(importance_df.sort_values('importance', ascending=False), use_container_width=True)


def show_temporal_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, key_prefix: str = "copy_default"):
    """Muestra gráficos temporales de predicciones vs valores reales."""
    df_plot = test_df[['fecha', 'no2_value']].copy()
    df_plot['Predicción'] = y_pred
    df_plot = df_plot.set_index('fecha')
    
    st.subheader("📈 Predicciones vs Valores Reales")
    
    # Controles para zoom temporal
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para visualizar:",
            value=(df_plot.index.min().date(), df_plot.index.max().date()),
            min_value=df_plot.index.min().date(),
            max_value=df_plot.index.max().date(),
            key=f"{key_prefix}_temporal_predictions_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad:",
            options=['Horaria', 'Media Diaria', 'Media Semanal'],
            index=0,
            key=f"{key_prefix}_temporal_predictions_granularity"
        )
    
    # Filtrar por fechas
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]
    
    if df_filtered.empty:
        st.warning("No hay datos en el rango seleccionado.")
        return
    
    # Aplicar granularidad
    if granularity == 'Media Diaria':
        df_agg = df_filtered.resample('D').mean()
        title = 'Predicciones vs Reales (Media Diaria)'
    elif granularity == 'Media Semanal':
        df_agg = df_filtered.resample('W-MON').mean()
        title = 'Predicciones vs Reales (Media Semanal)'
    else:
        df_agg = df_filtered
        title = 'Predicciones vs Reales (Horario)'
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_agg.index, df_agg['no2_value'], label='Valor Real', alpha=0.8, linewidth=1.5)
    ax.plot(df_agg.index, df_agg['Predicción'], label='Predicción XGBoost', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Formatear eje X
    if granularity != 'Horaria':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    fig.autofmt_xdate()
    ax.set_title(title)
    ax.set_ylabel('Concentración NO₂ (µg/m³)')
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


def show_residuals_over_time(test_df: pd.DataFrame, y_pred: np.ndarray, key_prefix: str = "copy_default"):
    """Muestra residuos a lo largo del tiempo."""
    df_plot = test_df[['fecha', 'no2_value']].copy()
    df_plot['Residuos'] = df_plot['no2_value'] - y_pred
    df_plot = df_plot.set_index('fecha')
    
    st.subheader("📉 Análisis Temporal de Errores")
    
    # Controles
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para errores:",
            value=(df_plot.index.min().date(), df_plot.index.max().date()),
            min_value=df_plot.index.min().date(),
            max_value=df_plot.index.max().date(),
            key=f"{key_prefix}_residuals_over_time_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad de errores:",
            options=['Horaria', 'Media Diaria', 'MAE Diario', 'Media Semanal'],
            index=0,
            key=f"{key_prefix}_residuals_over_time_granularity"
        )
    
    # Filtrar datos
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    df_filtered = df_plot[(df_plot.index >= start_date) & (df_plot.index < end_date)]
    
    if df_filtered.empty:
        st.warning("No hay datos de residuos en el rango seleccionado.")
        return
    
    # Aplicar granularidad
    if granularity == 'Media Diaria':
        df_agg = df_filtered.resample('D').mean()
        y_label = 'Residuo Medio (µg/m³)'
        title = 'Residuos Medios Diarios'
    elif granularity == 'MAE Diario':
        df_agg = df_filtered.resample('D')['Residuos'].apply(lambda x: x.abs().mean()).to_frame()
        df_agg.columns = ['Residuos']
        y_label = 'MAE Diario (µg/m³)'
        title = 'Error Absoluto Medio Diario'
    elif granularity == 'Media Semanal':
        df_agg = df_filtered.resample('W-MON').mean()
        y_label = 'Residuo Medio (µg/m³)'
        title = 'Residuos Medios Semanales'
    else:
        df_agg = df_filtered
        y_label = 'Residuo (µg/m³)'
        title = 'Residuos Horarios'
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_agg.index, df_agg['Residuos'], alpha=0.9, linewidth=1.5)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Error Cero')
    
    # Formatear eje X
    if granularity != 'Horaria':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    fig.autofmt_xdate()
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Fecha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


def show_info_panel():
    """Muestra panel de información sobre XGBoost."""
    with st.expander("ℹ️ Acerca del Entrenamiento XGBoost", expanded=False):
        st.markdown("""
        **Modelos XGBoost (eXtreme Gradient Boosting)**
        
        XGBoost es un algoritmo de machine learning que utiliza boosting para combinar 
        múltiples árboles de decisión débiles en un modelo robusto.
        
        **Características:**
        - **Excelente rendimiento**: Superior a muchos algoritmos tradicionales
        - **Manejo automático de valores faltantes**: Gestión inteligente de NaNs
        - **Regularización incorporada**: Previene sobreajuste automáticamente
        - **Paralelización eficiente**: Entrenamiento rápido en múltiples cores
        - **Importancia de variables**: Ranking automático de predictores
        
        **Proceso de entrenamiento:**
        1. **Preprocesamiento**: Variables cíclicas y conversión de unidades
        2. **División temporal**: Separación por fechas (entrenamiento/evaluación)
        3. **Filtrado de outliers**: ⚠️ **Solo en datos de entrenamiento**
        4. **Escalado**: Normalización para mejor convergencia
        5. **Entrenamiento**: Optimización de árboles con boosting
        6. **Early Stopping**: Prevención automática de sobreajuste
        
        **Ventajas vs GAM:**
        - ✅ **Captura interacciones**: Relaciones complejas entre variables
        - ✅ **Robusto a outliers**: Menos sensible a valores atípicos
        - ✅ **Escalabilidad**: Maneja grandes volúmenes de datos
        - ✅ **Flexibilidad**: Adapta automáticamente la complejidad
        
        **Aplicación específica:**
        Nowcasting de NO₂ basado en variables meteorológicas y de tráfico con 
        capacidad para detectar patrones complejos y no lineales.
        """)
        
        st.markdown("---")
        st.markdown("""
        **🚀 ¿Por qué XGBoost para nowcasting?**
        
        - **Tiempo real**: Predicciones rápidas para sistemas en producción
        - **Robustez**: Mantiene rendimiento con datos ruidosos o incompletos
        - **Interpretabilidad**: Feature importance para análisis de sensibilidad
        - **Adaptabilidad**: Se ajusta a patrones estacionales y tendencias
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Configuración Actual")
        
        if 'xgboost_config' in st.session_state:
            config = st.session_state.xgboost_config
            config_info = ""
            for key, value in config.items():
                if key != 'config_key':  # No mostrar la clave interna
                    config_info += f"**{key.replace('_', ' ').title()}:** {value}  \n"
            st.markdown(config_info)


# ==================== FUNCIÓN PRINCIPAL ====================

def xgboost_training_page():
    """Función principal del módulo de entrenamiento XGBoost."""
    
    # Inicializar trainer
    trainer = XGBoostTrainer()
    
    # Panel de información
    show_info_panel()
    
    # Cargar datos
    if not st.session_state.xgboost_data_loaded:
        if st.button("Cargar datos para entrenamiento XGBoost", type="primary"):
            with st.spinner("Cargando datos de entrenamiento..."):
                trainer.df_master = trainer.load_data()
                if not trainer.df_master.empty:
                    st.session_state.xgboost_data_loaded = True
                    st.success("Datos cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    trainer.df_master = trainer.load_data()
    
    if trainer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Configuración del modelo
    st.header("⚙️ Configuración del Modelo XGBoost")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de sensor
        sensores = sorted(trainer.df_master['id_no2'].unique())
        sensor_seleccionado = st.selectbox(
            "Sensor de NO₂", 
            sensores, 
            index=2 if len(sensores) > 2 else 0
        )
        
        # Filtrar por sensor
        df_sensor = trainer.df_master[trainer.df_master['id_no2'] == sensor_seleccionado]
        
        # Fechas disponibles
        fecha_min = df_sensor["fecha"].min().date()
        fecha_max = df_sensor["fecha"].max().date()
        
        # Fecha de división
        fecha_division = st.date_input(
            "Fecha de división (entrenamiento/prueba)",
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
            index=0  # none por defecto para XGBoost (maneja outliers naturalmente)
        )
        
        # Preprocesamiento
        preprocessing = st.selectbox(
            "Preprocesamiento temporal",
            options=list(PREPROCESSING_OPTIONS.keys()),
            format_func=lambda x: PREPROCESSING_OPTIONS[x],
            index=1  # none por defecto (XGBoost puede manejar variables temporales directamente)
        )
    
    # Selección de variables
    st.subheader("🔧 Selección de Variables")
    
    # Crear tabs para categorías
    var_tabs = st.tabs(list(VARIABLE_CATEGORIES.keys()))
    
    selected_features = []
    for i, (category, vars_list) in enumerate(VARIABLE_CATEGORIES.items()):
        with var_tabs[i]:
            # Filtrar variables que existen en los datos
            available_vars = [var for var in vars_list if var in trainer.df_master.columns or 'sin' in var or 'cos' in var]
            
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
                help=f"Selecciona las variables de {category.lower()} para el modelo"
            )
            selected_features.extend(selected_in_category)
    
    if not selected_features:
        st.warning("Selecciona al menos una variable para continuar.")
        return
    
    # Crear clave única para la configuración actual
    config_key = f"{sensor_seleccionado}_{outlier_method}_{preprocessing}_{len(selected_features)}"
    
    # Actualizar configuración en session_state
    st.session_state.xgboost_config = {
        'sensor': sensor_seleccionado,
        'fecha_division': fecha_division.strftime('%Y-%m-%d'),
        'outlier_method': OUTLIER_METHODS[outlier_method],
        'preprocessing': PREPROCESSING_OPTIONS[preprocessing],
        'num_variables': len(selected_features),
        'config_key': config_key
    }
    
    # Mostrar resumen de configuración
    with st.expander("📋 Resumen de Configuración"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Sensor:** {sensor_seleccionado}")
            st.write(f"**Variables:** {len(selected_features)}")
        with col2:
            st.write(f"**Outliers:** {OUTLIER_METHODS[outlier_method]}")
            st.write(f"**Preprocesamiento:** {PREPROCESSING_OPTIONS[preprocessing]}")
        with col3:
            st.write(f"**Fecha división:** {fecha_division}")
            st.write(f"**Período entrenamiento:** {fecha_min} - {fecha_division}")

    
    # Preparar datos
    with st.spinner("Preparando datos..."):
        # Aplicar transformaciones básicas
        df_processed = df_sensor.copy()

        st.write("📊 Datos originales:", len(df_sensor))
        
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
    model_filename = f'data/models/xgboost_model_{sensor_seleccionado}_{outlier_method}_{preprocessing}.pkl'
    model_exists = os.path.exists(model_filename)
    
    # Mostrar información de datos
    st.subheader("📊 Información del Conjunto de Datos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Muestras entrenamiento", len(train_df))
    with col2:
        st.metric("Muestras evaluación", len(test_df))
    with col3:
        st.metric("Variables seleccionadas", len(selected_features))
    with col4:
        st.metric("Outliers eliminados", outliers_removed)
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if model_exists:
            analyze_button = st.button("🔍 Analizar Modelo Existente", type="primary")
        else:
            analyze_button = False
            st.info("No existe un modelo XGBoost entrenado con esta configuración")
    
    with col2:
        train_button = st.button("🚀 Entrenar Nuevo Modelo XGBoost", type="secondary")
    
    # Inicializar variables de estado para el análisis si no existen
    if 'xgboost_analysis_data' not in st.session_state:
        st.session_state.xgboost_analysis_data = {}
    
    # Ejecutar análisis o entrenamiento
    if analyze_button and model_exists:
        with st.spinner("Cargando y analizando modelo XGBoost..."):
            model_info = trainer.load_model(model_filename)
            
            if model_info:
                model = model_info['model']
                feature_names = model_info['feature_names']
                scaler_dict = model_info['scaler_dict']
                scaler_target = model_info['scaler_target']
                
                # Preparar datos de prueba
                X_test = test_df[selected_features].copy()
                y_test = test_df['no2_value'].copy()
                
                # Escalar datos de prueba
                for feature in selected_features:
                    if feature in scaler_dict:
                        X_test[feature] = scaler_dict[feature].transform(X_test[[feature]])
                
                # Evaluar modelo
                metrics = trainer.evaluate_model(model, X_test, y_test, scaler_target)
                
                # Guardar datos del análisis en session_state
                st.session_state.xgboost_analysis_data[config_key] = {
                    'model': model,
                    'feature_names': feature_names,
                    'scaler_dict': scaler_dict,
                    'scaler_target': scaler_target,
                    'test_df': test_df,
                    'y_test': y_test,
                    'metrics': metrics,
                    'model_info': model_info
                }
    
    # Ejecutar entrenamiento
    if train_button:
        with st.spinner("Entrenando modelo XGBoost..."):
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
                sensor_seleccionado, outlier_method, preprocessing
            )
            
            st.success(f"Modelo XGBoost entrenado y guardado en: {model_path}")
            
            # Evaluar modelo
            metrics = trainer.evaluate_model(model, X_test_scaled, y_test, scaler_target)
            
            # Guardar datos del análisis en session_state
            st.session_state.xgboost_analysis_data[config_key] = {
                'model': model,
                'feature_names': selected_features,
                'scaler_dict': scaler_dict,
                'scaler_target': scaler_target,
                'test_df': test_df_clean,
                'y_test': y_test,
                'metrics': metrics,
                'model_info': {
                    'model': model,
                    'feature_names': selected_features,
                    'scaler_dict': scaler_dict,
                    'scaler_target': scaler_target
                }
            }
    
    # Mostrar análisis si existen datos
    if config_key in st.session_state.xgboost_analysis_data:
        analysis_data = st.session_state.xgboost_analysis_data[config_key]
        
        st.header("📊 Análisis del Modelo XGBoost")
        
        # Inicializar el estado del análisis si no existe
        if 'xgboost_analysis_tab' not in st.session_state:
            st.session_state.xgboost_analysis_tab = 0
        
        # Usar radio buttons para evitar problemas con tabs y reruns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Métricas", use_container_width=True, type="primary" if st.session_state.xgboost_analysis_tab == 0 else "secondary"):
                st.session_state.xgboost_analysis_tab = 0
                st.rerun()
        
        with col2:
            if st.button("📈 Predicciones", use_container_width=True, type="primary" if st.session_state.xgboost_analysis_tab == 1 else "secondary"):
                st.session_state.xgboost_analysis_tab = 1
                st.rerun()
        
        with col3:
            if st.button("🔍 Análisis", use_container_width=True, type="primary" if st.session_state.xgboost_analysis_tab == 2 else "secondary"):
                st.session_state.xgboost_analysis_tab = 2
                st.rerun()
        
        st.divider()
        
        # Mostrar contenido según la pestaña seleccionada
        if st.session_state.xgboost_analysis_tab == 0:
            st.subheader("📊 Métricas de Evaluación")
            show_model_metrics(analysis_data['metrics'])
            st.divider()
            show_residual_analysis(analysis_data['y_test'], analysis_data['metrics']['y_pred'])
        
        elif st.session_state.xgboost_analysis_tab == 1:
            show_temporal_predictions(analysis_data['test_df'], analysis_data['metrics']['y_pred'], key_prefix=config_key)
            st.divider()
            show_residuals_over_time(analysis_data['test_df'], analysis_data['metrics']['y_pred'], key_prefix=config_key)
        
        elif st.session_state.xgboost_analysis_tab == 2:
            st.subheader("🎯 Importancia de Variables")
            show_feature_importance(analysis_data['model'], analysis_data['feature_names']) 