"""
Módulo de Nowcasting Bayesiano para predicción de NO₂ con cuantificación de incertidumbre.

Este módulo implementa redes neuronales bayesianas usando TensorFlow Probability para
predecir niveles de NO₂ con estimaciones de incertidumbre mediante muestreo Monte Carlo.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import time
import warnings
from typing import Dict, List, Tuple, Optional
import joblib
import os

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configurar TensorFlow para usar la GPU si está disponible
tf.config.experimental.enable_memory_growth = True

# Distribuciones de TensorFlow Probability
tfd = tfp.distributions
tfpl = tfp.layers


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

# Categorías de variables para selección
VARIABLE_CATEGORIES = {
    "Variables Temporales": [
        'hour', 'day_of_week', 'month', 'day_of_year', 'weekend',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'season', 'season_sin', 'season_cos'
    ],
    "Variables de Tráfico": [
        'intensidad', 'carga', 'ocupacion', 'vmed'
    ],
    "Variables Meteorológicas": [
        'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp',
        'wind_speed', 'wind_direction'
    ]
}

# Metadatos de variables para interpretación
VARIABLE_METADATA = {
    # Variables meteorológicas
    'd2m': {'name': 'Punto de Rocío', 'unit': '°C', 'typical_range': (-10, 30)},
    't2m': {'name': 'Temperatura', 'unit': '°C', 'typical_range': (-5, 40)},
    'ssr': {'name': 'Radiación Solar Neta', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'ssrd': {'name': 'Radiación Solar Descendente', 'unit': 'W/m²', 'typical_range': (0, 1000)},
    'u10': {'name': 'Viento U 10m', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'v10': {'name': 'Viento V 10m', 'unit': 'km/h', 'typical_range': (-50, 50)},
    'sp': {'name': 'Presión Superficial', 'unit': 'hPa', 'typical_range': (980, 1030)},
    'tp': {'name': 'Precipitación Total', 'unit': 'mm', 'typical_range': (0, 50)},
    'wind_speed': {'name': 'Velocidad del Viento', 'unit': 'km/h', 'typical_range': (0, 100)},
    'wind_direction': {'name': 'Dirección del Viento', 'unit': '°', 'typical_range': (0, 360)},
    
    # Variables de tráfico
    'intensidad': {'name': 'Intensidad de Tráfico', 'unit': 'veh/h', 'typical_range': (0, 1500)},
    'carga': {'name': 'Carga de Tráfico', 'unit': '%', 'typical_range': (0, 100)},
    'ocupacion': {'name': 'Ocupación Vial', 'unit': '%', 'typical_range': (0, 100)},
    'vmed': {'name': 'Velocidad Media', 'unit': 'km/h', 'typical_range': (0, 100)},
    
    # Variables temporales
    'hour': {'name': 'Hora del Día', 'unit': 'h', 'typical_range': (0, 23)},
    'day_of_week': {'name': 'Día de la Semana', 'unit': '-', 'typical_range': (0, 6)},
    'month': {'name': 'Mes', 'unit': '-', 'typical_range': (1, 12)},
    'weekend': {'name': 'Fin de Semana', 'unit': '0/1', 'typical_range': (0, 1)}
}

# Configuraciones de modelo predefinidas
MODEL_CONFIGS = {
    'simple': {
        'name': 'Modelo Simple',
        'description': 'Red bayesiana básica con 2 capas densas',
        'layers': [64, 32],
        'dropout': 0.1
    },
    'medium': {
        'name': 'Modelo Medio',
        'description': 'Red bayesiana con 3 capas y regularización',
        'layers': [128, 64, 32],
        'dropout': 0.2
    },
    'complex': {
        'name': 'Modelo Complejo',
        'description': 'Red bayesiana profunda con 4 capas',
        'layers': [256, 128, 64, 32],
        'dropout': 0.3
    }
}

FEATURE_GROUPS = {
    "Variables Temporales": [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
        'weekend', 'season_sin', 'season_cos'
    ],
    "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
    "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction'],
    "NO₂ Histórico": ['no2_lag_1', 'no2_lag_2', 'no2_lag_3', 'no2_lag_6', 'no2_lag_12', 'no2_lag_24']
}


# ==================== CLASE PRINCIPAL ====================

class BayesianNowcaster:
    """Clase principal para nowcasting bayesiano de NO₂."""
    
    def __init__(self):
        self.df_master = None
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.selected_features = []
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'bnn_data_loaded' not in st.session_state:
            st.session_state.bnn_data_loaded = False
        if 'bnn_model_trained' not in st.session_state:
            st.session_state.bnn_model_trained = False
        if 'bnn_config' not in st.session_state:
            st.session_state.bnn_config = {}
        if 'bnn_show_results' not in st.session_state:
            st.session_state.bnn_show_results = False
    
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
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables temporales para capturar patrones cíclicos."""
        df = df.copy()
        
        # Variables temporales básicas
        df['hour'] = df['fecha'].dt.hour
        df['day_of_week'] = df['fecha'].dt.dayofweek
        df['month'] = df['fecha'].dt.month
        df['day_of_year'] = df['fecha'].dt.dayofyear
        df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Variables cíclicas temporales
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Estación del año
        df['season'] = df['month'].apply(
            lambda x: 0 if x in [12,1,2] else 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3
        )
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
        
        # Viento (m/s -> km/h)
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2) * 3.6
            df['wind_direction'] = (270 - np.arctan2(df['v10'], df['u10']) * 180/np.pi) % 360
            df['u10'] = df['u10'] * 3.6
            df['v10'] = df['v10'] * 3.6
        
        # Presión (Pa -> hPa)
        if 'sp' in df.columns:
            df['sp'] = df['sp'] / 100
        
        # Precipitación (m -> mm)
        if 'tp' in df.columns:
            df['tp'] = df['tp'] * 1000
        
        return df
    
    def prepare_nowcasting_data(self, df: pd.DataFrame, selected_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para nowcasting sin usar lags de NO₂.
        Solo usa variables meteorológicas, de tráfico y temporales.
        """
        # Eliminar filas con valores faltantes en las características o target
        df_clean = df.dropna(subset=selected_features + ['no2_value']).copy()
        
        if df_clean.empty:
            st.error("No hay datos válidos después de eliminar valores faltantes.")
            return np.array([]), np.array([])
        
        # Preparar características (X) y target (y)
        X = df_clean[selected_features].values
        y = df_clean['no2_value'].values
        
        return X, y
    
    def create_bayesian_model(self, input_shape: int, config: str = 'simple') -> tf.keras.Model:
        """Crea modelo bayesiano para nowcasting usando Monte Carlo Dropout."""
        
        model_config = MODEL_CONFIGS[config]
        layers = model_config['layers']
        dropout_rate = model_config['dropout']
        
        # Modelo secuencial más simple y estable
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,), name='input_layer')
        ])
        
        # Capas densas con dropout para aproximación bayesiana
        for i, units in enumerate(layers):
            model.add(tf.keras.layers.Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            
            # Dropout bayesiano (se mantiene activo durante inferencia)
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(
                    rate=dropout_rate, 
                    name=f'dropout_{i+1}'
                ))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(
            units=1,
            activation='linear',
            name='output_layer'
        ))
        
        return model
    
    def compile_model(self, model: tf.keras.Model, learning_rate: float = 0.001):
        """Compila el modelo con función de pérdida estándar."""
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """Entrena el modelo bayesiano."""
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones con cuantificación de incertidumbre usando Monte Carlo Dropout."""
        predictions = []
        
        # Muestreo Monte Carlo con dropout activo
        for _ in range(n_samples):
            # training=True mantiene el dropout activo durante la inferencia
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calcular estadísticas
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      n_samples: int = 100) -> Dict:
        """Evalúa el modelo con métricas de incertidumbre."""
        
        # Predicciones con incertidumbre
        y_pred_mean, y_pred_std = self.predict_with_uncertainty(X_test, n_samples)
        
        # Reshape si es necesario
        if len(y_pred_mean.shape) > 1:
            y_pred_mean = y_pred_mean.reshape(-1)
        if len(y_pred_std.shape) > 1:
            y_pred_std = y_pred_std.reshape(-1)
        if len(y_test.shape) > 1:
            y_test = y_test.reshape(-1)
        
        # Métricas tradicionales
        mse = mean_squared_error(y_test, y_pred_mean)
        mae = mean_absolute_error(y_test, y_pred_mean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_mean)
        
        # Métricas de incertidumbre
        mean_uncertainty = np.mean(y_pred_std)
        
        # Intervalos de confianza
        ci_lower = y_pred_mean - 1.96 * y_pred_std
        ci_upper = y_pred_mean + 1.96 * y_pred_std
        
        # Cobertura del intervalo de confianza (95%)
        within_ci = (y_test >= ci_lower) & (y_test <= ci_upper)
        coverage = np.mean(within_ci)
        
        # Debug: Información adicional para diagnosticar el problema
        debug_info = {
            'n_samples': len(y_test),
            'n_within_ci': np.sum(within_ci),
            'y_test_range': (np.min(y_test), np.max(y_test)),
            'pred_mean_range': (np.min(y_pred_mean), np.max(y_pred_mean)),
            'pred_std_range': (np.min(y_pred_std), np.max(y_pred_std)),
            'ci_lower_range': (np.min(ci_lower), np.max(ci_lower)),
            'ci_upper_range': (np.min(ci_upper), np.max(ci_upper)),
            'mean_ci_width': np.mean(ci_upper - ci_lower)
        }
        
        # Ancho promedio del intervalo de confianza
        interval_width = np.mean(ci_upper - ci_lower)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_uncertainty': mean_uncertainty,
            'coverage_95': coverage,
            'interval_width': interval_width,
            'predictions_mean': y_pred_mean,
            'predictions_std': y_pred_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'debug_info': debug_info
        }


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_uncertainty_predictions(df_test: pd.DataFrame, metrics: Dict, n_points: int = 500):
    """Muestra las predicciones con bandas de incertidumbre y controles de filtrado."""
    
    # Crear DataFrame completo con todas las predicciones
    df_viz = df_test.copy()
    df_viz['pred_mean'] = metrics['predictions_mean']
    df_viz['pred_std'] = metrics['predictions_std']
    df_viz['ci_lower'] = metrics['ci_lower']
    df_viz['ci_upper'] = metrics['ci_upper']
    df_viz = df_viz.set_index('fecha')
    
    st.subheader("🎯 Predicciones con Incertidumbre")
    
    # Controles para filtrado temporal
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para visualizar:",
            value=(df_viz.index.min().date(), df_viz.index.max().date()),
            min_value=df_viz.index.min().date(),
            max_value=df_viz.index.max().date(),
            key="uncertainty_predictions_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad temporal:",
            options=['Horaria', 'Media Diaria', 'Media Semanal'],
            index=0,
            key="uncertainty_predictions_granularity"
        )
    
    # Filtrar por fechas
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
    df_filtered = df_viz[(df_viz.index >= start_date) & (df_viz.index < end_date)]
    
    if df_filtered.empty:
        st.warning("No hay datos en el rango seleccionado.")
        return
    
    # Aplicar agregación temporal con propagación correcta de incertidumbre
    if granularity == 'Media Diaria':
        df_agg = aggregate_with_uncertainty(df_filtered, 'D')
        title = 'Predicciones Bayesianas vs Reales (Media Diaria)'
        date_format = '%Y-%m-%d'
    elif granularity == 'Media Semanal':
        df_agg = aggregate_with_uncertainty(df_filtered, 'W-MON')
        title = 'Predicciones Bayesianas vs Reales (Media Semanal)'
        date_format = '%Y-%m-%d'
    else:
        # Para datos horarios, submuestrear si hay demasiados puntos
        if len(df_filtered) > n_points:
            indices = np.linspace(0, len(df_filtered) - 1, n_points, dtype=int)
            df_agg = df_filtered.iloc[indices]
        else:
            df_agg = df_filtered
        title = 'Predicciones Bayesianas vs Reales (Horario)'
        date_format = '%Y-%m-%d %H:%M'
    
    # Crear gráfico con Plotly para mejor interactividad
    fig = go.Figure()
    
    # Banda de incertidumbre (95% CI)
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg['ci_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg['ci_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        name='IC 95% (Incertidumbre)',
        hovertemplate='IC Inferior: %{y:.2f} µg/m³<extra></extra>'
    ))
    
    # Valores reales
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg['no2_value'],
        mode='lines',
        name='Valor Real',
        line=dict(color='blue', width=2),
        hovertemplate='Fecha: %{x}<br>Real: %{y:.2f} µg/m³<extra></extra>'
    ))
    
    # Predicción media
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg['pred_mean'],
        mode='lines',
        name='Predicción Media',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Fecha: %{x}<br>Predicción: %{y:.2f} µg/m³<br>Incertidumbre: ±%{customdata:.2f} µg/m³<extra></extra>',
        customdata=df_agg['pred_std']
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Fecha',
        yaxis_title='Concentración NO₂ (µg/m³)',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar estadísticas de la agregación
    if granularity != 'Horaria':
        show_aggregation_stats(df_agg, granularity)


def aggregate_with_uncertainty(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Agrega datos temporalmente con propagación matemáticamente correcta de incertidumbre.
    
    Para la agregación de incertidumbres independientes:
    - Media: σ_media = σ / √n (error estándar de la media)
    - Varianza total: σ²_total = Σσᵢ² / n² (para medias)
    """
    
    def aggregate_group(group):
        n = len(group)
        if n == 0:
            return pd.Series({
                'no2_value': np.nan,
                'pred_mean': np.nan,
                'pred_std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_samples': 0
            })
        
        # Medias simples
        no2_mean = group['no2_value'].mean()
        pred_mean = group['pred_mean'].mean()
        
        # Propagación de incertidumbre para la media
        # σ_media = √(Σσᵢ²) / n (error estándar de la media)
        pred_std_aggregated = np.sqrt(np.sum(group['pred_std']**2)) / n
        
        # Nuevos intervalos de confianza
        ci_lower = pred_mean - 1.96 * pred_std_aggregated
        ci_upper = pred_mean + 1.96 * pred_std_aggregated
        
        return pd.Series({
            'no2_value': no2_mean,
            'pred_mean': pred_mean,
            'pred_std': pred_std_aggregated,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': n
        })
    
    return df.groupby(pd.Grouper(freq=freq)).apply(aggregate_group).dropna()


def show_aggregation_stats(df_agg: pd.DataFrame, granularity: str):
    """Muestra estadísticas sobre la agregación temporal."""
    
    with st.expander(f"📊 Estadísticas de Agregación {granularity}"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Períodos agregados",
                len(df_agg),
                help=f"Número de {granularity.lower()}s en el rango seleccionado"
            )
        
        with col2:
            avg_uncertainty = df_agg['pred_std'].mean()
            st.metric(
                "Incertidumbre promedio",
                f"{avg_uncertainty:.2f} µg/m³",
                help="Incertidumbre media después de la agregación temporal"
            )
        
        with col3:
            coverage = np.mean(
                (df_agg['no2_value'] >= df_agg['ci_lower']) & 
                (df_agg['no2_value'] <= df_agg['ci_upper'])
            )
            st.metric(
                "Cobertura IC 95%",
                f"{coverage:.1%}",
                help="Porcentaje de valores reales dentro del intervalo de confianza"
            )
        
        with col4:
            avg_width = (df_agg['ci_upper'] - df_agg['ci_lower']).mean()
            st.metric(
                "Ancho promedio IC",
                f"{avg_width:.2f} µg/m³",
                help="Ancho promedio del intervalo de confianza"
            )
        
        # Información matemática sobre la agregación
        st.markdown("---")
        st.markdown("""
        **🔬 Propagación de Incertidumbre:**
        
        - **Media temporal**: Simple promedio aritmético
        - **Incertidumbre agregada**: σ_agregada = √(Σσᵢ²) / n 
        - **Interpretación**: La incertidumbre se reduce con más muestras (ley de grandes números)
        - **IC 95%**: media ± 1.96 × σ_agregada
        
        Este enfoque es matemáticamente correcto para incertidumbres independientes.
        """)


def show_uncertainty_histogram(metrics: Dict):
    """Muestra histograma de incertidumbre."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=metrics['predictions_std'],
                nbinsx=30,
                name='Incertidumbre',
                marker_color='skyblue',
                opacity=0.7
            )
        ])
        
        fig_hist.update_layout(
            title='Distribución de la Incertidumbre',
            xaxis_title='Desviación Estándar (µg/m³)',
            yaxis_title='Frecuencia',
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Gráfico de dispersión: Error vs Incertidumbre
        df_scatter = pd.DataFrame({
            'uncertainty': metrics['predictions_std'],
            'error': np.abs(metrics['predictions_mean'] - metrics.get('y_true', metrics['predictions_mean']))
        })
        
        fig_scatter = px.scatter(
            df_scatter,
            x='uncertainty',
            y='error',
            title='Error vs Incertidumbre',
            labels={
                'uncertainty': 'Incertidumbre (µg/m³)',
                'error': 'Error Absoluto (µg/m³)'
            },
            opacity=0.6,
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)


def show_model_metrics(metrics: Dict):
    """Muestra métricas del modelo en formato dashboard."""
    
    st.subheader("📊 Métricas del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="RMSE",
            value=f"{metrics['rmse']:.2f}",
            help="Error cuadrático medio"
        )
        
    with col2:
        st.metric(
            label="MAE",
            value=f"{metrics['mae']:.2f}",
            help="Error absoluto medio"
        )
        
    with col3:
        st.metric(
            label="R²",
            value=f"{metrics['r2']:.3f}",
            help="Coeficiente de determinación"
        )
        
    with col4:
        st.metric(
            label="Incertidumbre Media",
            value=f"{metrics['mean_uncertainty']:.2f}",
            help="Incertidumbre promedio de las predicciones"
        )
    
    # Segunda fila de métricas específicas de incertidumbre
    col5, col6 = st.columns(2)
    
    with col5:
        coverage_color = "normal" if 0.90 <= metrics['coverage_95'] <= 0.98 else "inverse"
        st.metric(
            label="Cobertura IC 95%",
            value=f"{metrics['coverage_95']:.1%}",
            help="Porcentaje de valores reales dentro del intervalo de confianza del 95%",
            delta=f"{metrics['coverage_95'] - 0.95:.1%}" if coverage_color == "normal" else None
        )
        
    with col6:
        st.metric(
            label="Ancho IC Promedio",
            value=f"{metrics['interval_width']:.2f}",
            help="Ancho promedio del intervalo de confianza del 95%"
        )


def show_training_history(history):
    """Muestra la historia del entrenamiento."""
    
    if history is None:
        return
        
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Pérdida durante el Entrenamiento', 'MAE durante el Entrenamiento')
    )
    
    # Pérdida
    fig.add_trace(
        go.Scatter(
            y=history.history['loss'],
            name='Entrenamiento',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    if 'val_loss' in history.history:
        fig.add_trace(
            go.Scatter(
                y=history.history['val_loss'],
                name='Validación',
                line=dict(color='red')
            ),
            row=1, col=1
        )
    
    # MAE
    fig.add_trace(
        go.Scatter(
            y=history.history['mae'],
            name='Entrenamiento',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    if 'val_mae' in history.history:
        fig.add_trace(
            go.Scatter(
                y=history.history['val_mae'],
                name='Validación',
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Época", row=1, col=1)
    fig.update_xaxes(title_text="Época", row=1, col=2)
    fig.update_yaxes(title_text="Pérdida", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    
    fig.update_layout(height=400, title_text="Historia del Entrenamiento")
    
    st.plotly_chart(fig, use_container_width=True)


# ==================== FUNCIÓN PRINCIPAL ====================

def bayesian_nowcasting_page():
    """Función principal para la página de nowcasting bayesiano."""
    
    # Inicializar el nowcaster
    nowcaster = BayesianNowcaster()
    
    # Panel de información
    show_info_panel()
    
    # Cargar datos
    if not st.session_state.bnn_data_loaded:
        if st.button("🔄 Cargar datos para Nowcasting Bayesiano", type="primary"):
            with st.spinner("Cargando datos..."):
                nowcaster.df_master = nowcaster.load_data()
                if not nowcaster.df_master.empty:
                    # Crear características temporales y de retraso
                    nowcaster.df_master = nowcaster.create_temporal_features(nowcaster.df_master)
                    nowcaster.df_master = nowcaster.convert_units(nowcaster.df_master)
                    st.session_state.bnn_data_loaded = True
                    st.success("✅ Datos cargados correctamente")
                    st.rerun()
        return
    
    # Recuperar datos
    nowcaster.df_master = nowcaster.load_data()
    if not nowcaster.df_master.empty:
        nowcaster.df_master = nowcaster.create_temporal_features(nowcaster.df_master)
        nowcaster.df_master = nowcaster.convert_units(nowcaster.df_master)
    
    if nowcaster.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Configuración del nowcasting
    st.subheader("🎯 Configuración del Nowcasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de sensor
        sensores = sorted(nowcaster.df_master['id_no2'].unique())
        sensor_seleccionado = st.selectbox(
            "Sensor de NO₂", 
            sensores, 
            index=2 if len(sensores) > 2 else 0,
            key="sensor_selection"
        )
        
        # Filtrar por sensor
        df_sensor = nowcaster.df_master[nowcaster.df_master['id_no2'] == sensor_seleccionado]
        
        # Fechas disponibles
        fecha_min = df_sensor["fecha"].min().date()
        fecha_max = df_sensor["fecha"].max().date()
        
        # Fecha de división
        fecha_division = st.date_input(
            "Fecha de división (entrenamiento/evaluación)",
            value=pd.to_datetime('2024-01-01').date(),
            min_value=fecha_min,
            max_value=fecha_max,
            help="Los datos anteriores se usan para entrenamiento, posteriores para evaluación",
            key="split_date"
        )
    
    with col2:
        # Configuración del modelo
        model_type = st.selectbox(
            "Arquitectura del modelo",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['name'],
            help="Selecciona la complejidad del modelo bayesiano",
            key="model_architecture"
        )
        
        # Mostrar descripción del modelo
        st.info(f"📋 **{MODEL_CONFIGS[model_type]['name']}**: {MODEL_CONFIGS[model_type]['description']}")
    
    # Selección de variables
    st.subheader("🔧 Selección de Variables")
    
    # Crear tabs para categorías usando VARIABLE_CATEGORIES
    var_tabs = st.tabs(list(VARIABLE_CATEGORIES.keys()))
    
    selected_features = []
    for i, (category, vars_list) in enumerate(VARIABLE_CATEGORIES.items()):
        with var_tabs[i]:
            # Filtrar variables que existen en los datos
            available_vars = [var for var in vars_list if var in nowcaster.df_master.columns or 'sin' in var or 'cos' in var]
            
            # Configurar defaults - todas las variables disponibles
            default_vars = available_vars
            
            selected_in_category = st.multiselect(
                f"Variables de {category}",
                available_vars,
                default=default_vars,
                help=f"Selecciona las variables de {category.lower()} para el modelo",
                key=f"vars_{category.replace(' ', '_').lower()}"
            )
            selected_features.extend(selected_in_category)
    
    if not selected_features:
        st.warning("Selecciona al menos una variable para continuar.")
        return
    
    nowcaster.selected_features = selected_features
    
    # Mostrar resumen de configuración
    with st.expander("📋 Resumen de Configuración"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Sensor:** {sensor_seleccionado}")
            st.write(f"**Variables:** {len(selected_features)}")
        with col2:
            st.write(f"**Arquitectura:** {MODEL_CONFIGS[model_type]['name']}")
        with col3:
            st.write(f"**Fecha división:** {fecha_division}")
    
    # Preparar datos
    with st.spinner("Preparando datos para entrenamiento..."):
        df_processed = df_sensor.copy()
        
        # Limpiar datos
        df_clean = df_processed.dropna(subset=selected_features + ['no2_value'])
        
        if len(df_clean) < 1000:
            st.error("❌ Datos insuficientes después de limpiar NaN")
            return
        
        # Dividir datos temporalmente
        fecha_division_dt = pd.to_datetime(fecha_division)
        train_data = df_clean[df_clean['fecha'] <= fecha_division_dt]
        test_data = df_clean[df_clean['fecha'] > fecha_division_dt]
        
        if len(train_data) < 500 or len(test_data) < 100:
            st.error("❌ No hay suficientes datos para entrenamiento o evaluación.")
            return
    
    # Mostrar información de datos
    st.subheader("📊 Información del Conjunto de Datos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Registros totales", len(df_clean))
    with col2:
        st.metric("Entrenamiento", len(train_data))
    with col3:
        st.metric("Evaluación", len(test_data))
    with col4:
        st.metric("Variables", len(selected_features))
    
    # Verificar si existe modelo entrenado
    config_key = f"{sensor_seleccionado}_{model_type}_{len(selected_features)}"
    
    # Botones de acción
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.bnn_model_trained and st.session_state.get('bnn_config_key') == config_key:
            analyze_button = st.button("Ver Resultados del Modelo", type="primary")
        else:
            analyze_button = False
            if st.session_state.bnn_model_trained:
                st.info("Configuración cambiada - Entrena un nuevo modelo")
            else:
                st.info("No existe un modelo entrenado con esta configuración")
    
    with col2:
        train_button = st.button("🚀 Entrenar Modelo Bayesiano", type="secondary")
    
    # Procesar acciones
    if train_button:
        if len(selected_features) == 0:
            st.error("❌ Selecciona al menos un grupo de variables")
        else:
            st.session_state.bnn_config_key = config_key
            st.session_state.bnn_show_results = False  # Reset visualización
            train_bayesian_model(
                nowcaster, train_data, test_data, model_type, sensor_seleccionado
            )
    
    elif analyze_button and st.session_state.bnn_model_trained:
        st.session_state.bnn_show_results = True
    
    # Mostrar resultados si está activo el flag
    if st.session_state.get('bnn_show_results', False) and st.session_state.bnn_model_trained:
        show_model_results()
    
    # Si no se ha entrenado un modelo, mostrar resumen de datos
    elif not st.session_state.bnn_model_trained:
        show_data_summary(df_clean)


def show_info_panel():
    """Muestra panel de información sobre el nowcasting bayesiano."""
    
    # st.markdown("""
    # ### 🧠 Nowcasting Bayesiano de NO₂
    
    # Implementación de **redes neuronales bayesianas** para nowcasting de NO₂ con 
    # **cuantificación de incertidumbre** mediante muestreo Monte Carlo.
    # """)
    
    # Mostrar información sobre el método
    with st.expander("ℹ️ Acerca del Nowcasting Bayesiano", expanded=False):
        st.markdown("""
        **Nowcasting Bayesiano de NO₂**
        
        Implementación de **redes neuronales bayesianas** para predicción inmediata de NO₂ 
        basada únicamente en **variables meteorológicas y de tráfico actuales**.
        
        **Ventajas clave:**
        - **🌍 Aplicable a cualquier ciudad**: No requiere datos históricos de NO₂
        - **🎯 Predicción en tiempo real**: Basado en condiciones actuales
        - **📊 Cuantificación de incertidumbre**: Intervalos de confianza mediante Monte Carlo
        - **🔬 Modelado bayesiano**: Captura incertidumbre epistémica y aleatoria
        
        **Metodología:**
        1. **Entrada**: Variables meteorológicas + tráfico + temporales
        2. **Procesamiento**: Red neuronal bayesiana con capas variacionales
        3. **Salida**: Media de predicción + intervalo de incertidumbre
        4. **Muestreo**: Monte Carlo para estimar distribución posterior
        
        **Variables disponibles:**
        - **Meteorológicas**: Temperatura, humedad, viento, presión, radiación solar
        - **Tráfico**: Intensidad, velocidad, ocupación, carga vial  
        - **Temporales**: Hora, día, mes, patrones cíclicos estacionales
        
        **⚠️ Nota importante:**
        Este modelo **NO utiliza valores históricos de NO₂**, lo que permite su 
        aplicación directa a ciudades sin sensores de calidad del aire existentes.
        """)
        
        st.markdown("---")
        st.markdown("""
        **🎯 Casos de uso ideales:**
        - Implementación en ciudades sin red de monitoreo
        - Validación de sensores existentes
        - Predicción en tiempo real para alertas
        - Estudios de impacto ambiental
        """)


def train_bayesian_model(nowcaster, train_data, test_data, model_type, sensor_id):
    """Entrena el modelo bayesiano."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Preparando datos...")
        progress_bar.progress(20)
        
        # Preparar datos (ahora son arrays 2D: [samples, features])
        X_train, y_train = nowcaster.prepare_nowcasting_data(train_data, nowcaster.selected_features)
        X_test, y_test = nowcaster.prepare_nowcasting_data(test_data, nowcaster.selected_features)
        
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("❌ No se pudieron preparar suficientes datos")
            return
        
        st.write(f"📊 Datos preparados - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Validación cruzada temporal
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        status_text.text("🔄 Normalizando datos...")
        progress_bar.progress(30)
        
        # Normalizar datos (ya son 2D, no necesitan reshape)
        nowcaster.scaler_X = StandardScaler()
        X_train_scaled = nowcaster.scaler_X.fit_transform(X_train)
        X_val_scaled = nowcaster.scaler_X.transform(X_val)
        X_test_scaled = nowcaster.scaler_X.transform(X_test)
        
        nowcaster.scaler_y = StandardScaler()
        y_train_scaled = nowcaster.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = nowcaster.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        status_text.text("🔄 Creando modelo...")
        progress_bar.progress(40)
        
        # Crear y compilar modelo
        nowcaster.model = nowcaster.create_bayesian_model(X_train_scaled.shape[1], model_type)
        nowcaster.model = nowcaster.compile_model(nowcaster.model)
        
        st.write(f"🧠 Modelo creado - Input shape: {X_train_scaled.shape[1]}")
        
        status_text.text("🚀 Entrenando modelo...")
        progress_bar.progress(50)
        
        # Entrenar modelo
        history = nowcaster.train_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=100, batch_size=32
        )
        
        status_text.text("📊 Evaluando modelo...")
        progress_bar.progress(90)
        
        # Evaluar modelo - IMPORTANTE: usar datos desnormalizados para métricas finales
        y_pred_mean_scaled, y_pred_std_scaled = nowcaster.predict_with_uncertainty(
            X_test_scaled, n_samples=100
        )
        
        # Desnormalizar predicciones
        y_pred_mean = nowcaster.scaler_y.inverse_transform(
            y_pred_mean_scaled.reshape(-1, 1)
        ).flatten()
        
        # Para la desviación estándar, escalar por la std del scaler
        y_pred_std = y_pred_std_scaled.flatten() * nowcaster.scaler_y.scale_[0]
        
        # Calcular intervalos de confianza desnormalizados
        ci_lower = y_pred_mean - 1.96 * y_pred_std
        ci_upper = y_pred_mean + 1.96 * y_pred_std
        
        # Calcular cobertura con datos desnormalizados
        within_ci = (y_test >= ci_lower) & (y_test <= ci_upper)
        coverage_95 = np.mean(within_ci)
        
        # Métricas de incertidumbre
        mean_uncertainty = np.mean(y_pred_std)
        interval_width = np.mean(ci_upper - ci_lower)
        
        # Debug info
        debug_info = {
            'n_samples': len(y_test),
            'n_within_ci': np.sum(within_ci),
            'y_test_range': (np.min(y_test), np.max(y_test)),
            'pred_mean_range': (np.min(y_pred_mean), np.max(y_pred_mean)),
            'pred_std_range': (np.min(y_pred_std), np.max(y_pred_std)),
            'ci_lower_range': (np.min(ci_lower), np.max(ci_lower)),
            'ci_upper_range': (np.min(ci_upper), np.max(ci_upper)),
            'mean_ci_width': interval_width
        }
        
        # Calcular todas las métricas con datos desnormalizados
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred_mean),
            'mae': mean_absolute_error(y_test, y_pred_mean),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_mean)),
            'r2': r2_score(y_test, y_pred_mean),
            'mean_uncertainty': mean_uncertainty,
            'coverage_95': coverage_95,
            'interval_width': interval_width,
            'predictions_mean': y_pred_mean,
            'predictions_std': y_pred_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'y_true': y_test,
            'debug_info': debug_info
        }
        
        # Guardar resultados en session state
        st.session_state.bnn_model_trained = True
        st.session_state.bnn_metrics = metrics
        st.session_state.bnn_history = history
        st.session_state.bnn_test_data = test_data.iloc[-len(y_test):].copy()
        st.session_state.bnn_config = {
            'model_type': model_type,
            'features': nowcaster.selected_features,
            'sensor_id': sensor_id
        }
        st.session_state.bnn_show_results = True  # Activar visualización automática
        
        progress_bar.progress(100)
        status_text.text("✅ Modelo entrenado correctamente!")
        
        st.success("🎉 ¡Modelo bayesiano entrenado exitosamente!")
        
        # Mostrar resultados automáticamente
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        import traceback
        st.error(f"❌ Error durante el entrenamiento: {str(e)}")
        st.error(f"📋 Detalles del error: {traceback.format_exc()}")
        progress_bar.empty()
        status_text.empty()


def show_model_results():
    """Muestra los resultados del modelo entrenado."""
    
    if 'bnn_metrics' not in st.session_state:
        st.error("❌ No hay resultados de modelo disponibles")
        return
    
    metrics = st.session_state.bnn_metrics
    history = st.session_state.bnn_history
    test_data = st.session_state.bnn_test_data
    config = st.session_state.bnn_config
    
    # Botón para cerrar resultados
    col1, col2, col3 = st.columns([1, 1, 8])
    # with col1:
    #     if st.button("❌ Cerrar", help="Cerrar visualización de resultados"):
    #         st.session_state.bnn_show_results = False
    #         st.rerun()
    
    st.header("🎯 Resultados del Modelo Bayesiano")
    
    # Mostrar configuración del modelo
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Sensor:** {config['sensor_id']}")
        st.info(f"**Arquitectura:** {MODEL_CONFIGS[config['model_type']]['name']}")
    with col2:
        st.info(f"**Características:** {len(config['features'])}")
    
    # Mostrar métricas
    show_model_metrics(metrics)
    
    # Mostrar historia del entrenamiento
    st.subheader("📈 Historia del Entrenamiento")
    show_training_history(history)
    
    # Mostrar predicciones con incertidumbre
    show_uncertainty_predictions(test_data, metrics)
    
    # Análisis de incertidumbre
    st.subheader("🔍 Análisis de Incertidumbre")
    show_uncertainty_histogram(metrics)
    
    # Información adicional
    with st.expander("📊 Información Adicional"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Métricas de Calibración:**")
            st.write(f"- Cobertura IC 95%: {metrics['coverage_95']:.1%}")
            st.write(f"- Ancho promedio IC: {metrics['interval_width']:.2f} µg/m³")
            st.write(f"- Incertidumbre media: {metrics['mean_uncertainty']:.2f} µg/m³")
        
        with col2:
            st.markdown("**Configuración del Modelo:**")
            st.write(f"- Arquitectura: {MODEL_CONFIGS[config['model_type']]['description']}")
        
        # Debug de cobertura
        if 'debug_info' in metrics:
            st.markdown("---")
            st.markdown("**🔍 Debug - Información de Cobertura:**")
            debug = metrics['debug_info']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Muestras totales:** {debug['n_samples']}")
                st.write(f"**Dentro del IC:** {debug['n_within_ci']}")
                st.write(f"**Cobertura:** {debug['n_within_ci']/debug['n_samples']:.1%}")
            
            with col2:
                st.write(f"**Rango valores reales:** {debug['y_test_range'][0]:.1f} - {debug['y_test_range'][1]:.1f}")
                st.write(f"**Rango predicciones:** {debug['pred_mean_range'][0]:.1f} - {debug['pred_mean_range'][1]:.1f}")
                st.write(f"**Ancho medio IC:** {debug['mean_ci_width']:.2f}")
            
            with col3:
                st.write(f"**Rango incertidumbre:** {debug['pred_std_range'][0]:.3f} - {debug['pred_std_range'][1]:.3f}")
                st.write(f"**IC inferior:** {debug['ci_lower_range'][0]:.1f} - {debug['ci_lower_range'][1]:.1f}")
                st.write(f"**IC superior:** {debug['ci_upper_range'][0]:.1f} - {debug['ci_upper_range'][1]:.1f}")
            
            # Mostrar algunos ejemplos específicos
            st.markdown("**📋 Ejemplos de predicciones:**")
            examples_df = pd.DataFrame({
                'Real': metrics['y_true'][:10],
                'Predicción': metrics['predictions_mean'][:10],
                'Incertidumbre': metrics['predictions_std'][:10],
                'IC_inferior': metrics['ci_lower'][:10],
                'IC_superior': metrics['ci_upper'][:10],
                'Dentro_IC': ((metrics['y_true'][:10] >= metrics['ci_lower'][:10]) & 
                             (metrics['y_true'][:10] <= metrics['ci_upper'][:10]))
            })
            st.dataframe(examples_df.round(2), use_container_width=True)


def show_data_summary(df: pd.DataFrame):
    """Muestra un resumen de los datos cargados."""
    
    # Gráfico de series temporales
    st.subheader("📈 Serie Temporal de NO₂")
    
    # Agregación diaria
    daily_data = df.groupby(df['fecha'].dt.date)['no2_value'].agg(['mean', 'std']).reset_index()
    daily_data['fecha'] = pd.to_datetime(daily_data['fecha'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['fecha'],
        y=daily_data['mean'],
        mode='lines',
        name='Promedio Diario',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Evolución Temporal del NO₂ (Promedio Diario)',
        xaxis_title='Fecha',
        yaxis_title='Concentración NO₂ (µg/m³)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True) 