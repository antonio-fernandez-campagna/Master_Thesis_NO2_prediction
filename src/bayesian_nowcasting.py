"""
Módulo de Nowcasting Bayesiano para predicción de NO₂ con cuantificación de incertidumbre.

Este módulo implementa redes neuronales bayesianas usando TensorFlow Probability para
predecir niveles de NO₂ con estimaciones de incertidumbre mediante muestreo Monte Carlo.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        'name': 'Simple (2 capas)',
        'layers': [64, 32],
        'dropout': 0.2,
        'description': 'Modelo básico con 2 capas densas'
    },
    'medium': {
        'name': 'Medio (3 capas)',
        'layers': [128, 64, 32],
        'dropout': 0.25,
        'description': 'Modelo intermedio con 3 capas densas'
    },
    'deep': {
        'name': 'Profundo (4 capas)',
        'layers': [256, 128, 64, 32],
        'dropout': 0.3,
        'description': 'Modelo profundo con 4 capas densas'
    },
    'wide': {
        'name': 'Ancho (2 capas grandes)',
        'layers': [256, 128],
        'dropout': 0.2,
        'description': 'Modelo con capas más anchas para capturar patrones complejos'
    },
    'robust': {
        'name': 'Robusto (5 capas con regularización)',
        'layers': [256, 128, 64, 32, 16],
        'dropout': 0.35,
        'description': 'Modelo robusto con más regularización para datos ruidosos'
    }
}

FEATURE_GROUPS = {
    "Variables Temporales": [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
        'weekend', 'season_sin', 'season_cos'
    ],
    "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
    "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']
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
        
        # Modelo con mejor arquitectura para aprendizaje
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,), name='input_layer'),
            
            # Normalización batch para mejor convergencia
            tf.keras.layers.BatchNormalization(name='input_bn')
        ])
        
        # Capas densas con arquitectura mejorada
        for i, units in enumerate(layers):
            # Capa densa
            model.add(tf.keras.layers.Dense(
                units=units,
                kernel_initializer='he_normal',  # Mejor inicialización para ReLU
                name=f'dense_{i+1}'
            ))
            
            # Normalización batch antes de activación
            model.add(tf.keras.layers.BatchNormalization(name=f'bn_{i+1}'))
            
            # Activación
            model.add(tf.keras.layers.Activation('relu', name=f'relu_{i+1}'))
            
            # Dropout bayesiano (reducido para mejor aprendizaje inicial)
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(
                    rate=max(0.1, dropout_rate * 0.5),  # Reducir dropout inicialmente
                    name=f'dropout_{i+1}'
                ))
        
        # Capa de salida con inicialización más conservadora
        model.add(tf.keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='glorot_normal',
            name='output_layer'
        ))
        
        return model
    
    def compile_model(self, model: tf.keras.Model, learning_rate: float = 0.01):  # Learning rate más alto
        """Compila el modelo con función de pérdida estándar."""
        
        # Optimizador con mejor configuración
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 150, batch_size: int = 64, 
                   learning_rate: float = 0.01, use_early_stopping: bool = True,
                   early_stopping_patience: int = 25, reduce_lr_patience: int = 12) -> tf.keras.callbacks.History:
        """Entrena el modelo bayesiano con parámetros configurables."""
        
        # Recompilar modelo si el learning rate ha cambiado
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        if abs(current_lr - learning_rate) > 1e-6:
            self.model = self.compile_model(self.model, learning_rate=learning_rate)
        
        # Callbacks configurables
        callbacks = []
        
        # Early stopping (opcional)
        if use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ))
        
        # Reduce learning rate on plateau (siempre activo)
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ))
        
        # Learning rate scheduler (opcional, solo si no hay early stopping)
        if not use_early_stopping:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: learning_rate * (0.95 ** epoch) if epoch < 50 else learning_rate * (0.95 ** 50) * (0.98 ** (epoch - 50)),
                verbose=0
            ))
        
        # Entrenar el modelo con verbose=1 para ver progreso
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return history
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones con cuantificación de incertidumbre usando Monte Carlo Dropout."""
        
        if self.model is None:
            raise ValueError("El modelo no ha sido creado o entrenado")
        
        predictions = []
        
        try:
            # Convertir a tensor de TensorFlow si no lo es
            if not isinstance(X, tf.Tensor):
                X = tf.convert_to_tensor(X, dtype=tf.float32)
            
            # Muestreo Monte Carlo con dropout activo
            for i in range(n_samples):
                try:
                    # Usar un contexto limpio para cada predicción
                    with tf.name_scope(f'monte_carlo_sample_{i}'):
                        # training=True mantiene el dropout activo durante la inferencia
                        pred = self.model(X, training=True)
                        predictions.append(pred.numpy())
                except Exception as e:
                    # Si falla una predicción individual, intentar sin name_scope
                    try:
                        pred = self.model(X, training=True)
                        predictions.append(pred.numpy())
                    except Exception as e2:
                        st.warning(f"Error en muestra {i+1}/{n_samples}: {str(e2)}")
                        # Usar la última predicción válida si existe
                        if predictions:
                            predictions.append(predictions[-1])
                        else:
                            # Predicción estándar como fallback
                            pred = self.model(X, training=False)
                            predictions.append(pred.numpy())
            
            if not predictions:
                raise ValueError("No se pudieron generar predicciones")
            
            predictions = np.array(predictions)
            
            # Calcular estadísticas
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            return mean_pred, std_pred
            
        except Exception as e:
            st.error(f"Error en predict_with_uncertainty: {str(e)}")
            st.error("Intentando predicción sin Monte Carlo como fallback...")
            
            # Fallback: predicción estándar sin dropout
            try:
                pred = self.model(X, training=False)
                pred_np = pred.numpy()
                # Simular incertidumbre mínima
                fake_std = np.ones_like(pred_np) * 0.1
                return pred_np, fake_std
            except Exception as e2:
                st.error(f"Error en fallback: {str(e2)}")
                raise e2
    
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
    
    st.subheader(" Predicciones con Incertidumbre")
    
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
            "Granularidad:",
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
    elif granularity == 'Media Semanal':
        df_agg = aggregate_with_uncertainty(df_filtered, 'W-MON')
        title = 'Predicciones Bayesianas vs Reales (Media Semanal)'
    else:
        # Para datos horarios, submuestrear si hay demasiados puntos
        if len(df_filtered) > n_points:
            indices = np.linspace(0, len(df_filtered) - 1, n_points, dtype=int)
            df_agg = df_filtered.iloc[indices]
        else:
            df_agg = df_filtered
        title = 'Predicciones Bayesianas vs Reales (Horario)'
    
    # Crear gráfico con matplotlib
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Colores elegantes
    color_real = '#2E86AB'      # Azul elegante
    color_pred = '#A23B72'      # Magenta oscuro
    color_ci = '#F18F01'        # Naranja cálido
    
    # 1. Banda de incertidumbre (95% CI)
    ax.fill_between(
        df_agg.index, 
        df_agg['ci_lower'], 
        df_agg['ci_upper'],
        color=color_ci, 
        alpha=0.2, 
        label='Intervalo Confianza 95%',
        zorder=1
    )
    
    # 2. Líneas de los límites del IC (más sutil)
    ax.plot(df_agg.index, df_agg['ci_lower'], color=color_ci, alpha=0.4, linewidth=0.8, zorder=2)
    ax.plot(df_agg.index, df_agg['ci_upper'], color=color_ci, alpha=0.4, linewidth=0.8, zorder=2)
    
    # 3. Valores reales
    ax.plot(
        df_agg.index, 
        df_agg['no2_value'], 
        label='Valor Real', 
        color=color_real,
        alpha=0.8, 
        linewidth=2.5,
        zorder=3
    )
    
    # 4. Predicción media
    ax.plot(
        df_agg.index, 
        df_agg['pred_mean'], 
        label='Predicción Bayesiana', 
        color=color_pred,
        linestyle='--', 
        alpha=0.9, 
        linewidth=2.5,
        zorder=4
    )
    
    # Formatear eje X según granularidad
    if granularity == 'Horaria':
        if len(df_agg) > 48:  # Más de 2 días
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotar fechas para mejor legibilidad
    fig.autofmt_xdate()
    
    # Estilo del gráfico
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Concentración NO₂ (µg/m³)', fontsize=12)
    ax.set_xlabel('Fecha', fontsize=12)
    
    # Leyenda elegante
    legend = ax.legend(
        loc='upper left', 
        frameon=True, 
        fancybox=True, 
        shadow=True,
        fontsize=11
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grilla sutil
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    # Mejorar los ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar el gráfico
    st.pyplot(fig)
    plt.close()
    
    # Panel de estadísticas
    st.markdown("---")
    
    # Calcular métricas
    mae = np.mean(np.abs(df_agg['no2_value'] - df_agg['pred_mean']))
    rmse = np.sqrt(np.mean((df_agg['no2_value'] - df_agg['pred_mean'])**2))
    coverage = np.mean((df_agg['no2_value'] >= df_agg['ci_lower']) & 
                      (df_agg['no2_value'] <= df_agg['ci_upper']))
    correlation = np.corrcoef(df_agg['no2_value'], df_agg['pred_mean'])[0, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Puntos de Datos",
            f"{len(df_agg):,}",
            help="Número de puntos en el período seleccionado"
        )
    
    with col2:
        st.metric(
            "RMSE",
            f"{rmse:.2f} µg/m³",
            help="Error cuadrático medio"
        )
    
    with col3:
        st.metric(
            "Cobertura IC 95%",
            f"{coverage:.1%}",
            help="% de valores reales dentro del intervalo de confianza"
        )
    
    with col4:
        st.metric(
            "Correlación R",
            f"{correlation:.3f}",
            help="Correlación entre valores reales y predicciones"
        )
    
    # Mostrar estadísticas de agregación si aplica
    if granularity != 'Horaria':
        show_aggregation_stats(df_agg, granularity)
    
    # Panel adicional con métricas detalladas
    with st.expander(" Métricas Detalladas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estadísticas de Error:**")
            bias = np.mean(df_agg['pred_mean'] - df_agg['no2_value'])
            mape = np.mean(np.abs((df_agg['no2_value'] - df_agg['pred_mean']) / df_agg['no2_value'])) * 100
            
            st.write(f"• MAE: {mae:.2f} µg/m³")
            st.write(f"• RMSE: {rmse:.2f} µg/m³")
            st.write(f"• Sesgo: {bias:.2f} µg/m³")
            st.write(f"• MAPE: {mape:.1f}%")
        
        with col2:
            st.markdown("**Estadísticas de Incertidumbre:**")
            avg_uncertainty = df_agg['pred_std'].mean()
            ci_width = np.mean(df_agg['ci_upper'] - df_agg['ci_lower'])
            
            st.write(f"• Incertidumbre promedio: ±{avg_uncertainty:.2f} µg/m³")
            st.write(f"• Ancho IC promedio: {ci_width:.2f} µg/m³")
            st.write(f"• Cobertura IC 95%: {coverage:.1%}")
            st.write(f"• Correlación: {correlation:.3f}")


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
    
    with st.expander(f" Estadísticas de Agregación {granularity}"):
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
    """Muestra la historia del entrenamiento con análisis detallado."""
    
    if history is None:
        st.warning("No hay historia de entrenamiento disponible")
        return
    
    # Crear gráficos de entrenamiento
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Gráfico 1: Pérdida
    ax1.plot(epochs, history.history['loss'], 'b-', label='Entrenamiento', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validación', linewidth=2)
    ax1.set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Análisis de convergencia
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    min_val_loss = min(history.history['val_loss'])
    min_val_epoch = history.history['val_loss'].index(min_val_loss) + 1
    
    ax1.axhline(y=min_val_loss, color='orange', linestyle='--', alpha=0.7, 
                label=f'Mejor val_loss: {min_val_loss:.4f} (época {min_val_epoch})')
    ax1.legend()
    
    # Gráfico 2: MAE
    ax2.plot(epochs, history.history['mae'], 'b-', label='Entrenamiento MAE', linewidth=2)
    ax2.plot(epochs, history.history['val_mae'], 'r-', label='Validación MAE', linewidth=2)
    ax2.set_title('Error Absoluto Medio (MAE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Ratio Val/Train Loss
    ratio_loss = [val/train for val, train in zip(history.history['val_loss'], history.history['loss'])]
    ax3.plot(epochs, ratio_loss, 'g-', linewidth=2)
    ax3.set_title('Ratio Validación/Entrenamiento Loss', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Val Loss / Train Loss')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ratio = 1.0')
    ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Ratio = 1.2 (límite saludable)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Smoothed Loss (media móvil)
    window = max(1, len(epochs) // 10)  # Ventana del 10% de las épocas
    if len(epochs) > window:
        train_smooth = pd.Series(history.history['loss']).rolling(window=window, center=True).mean()
        val_smooth = pd.Series(history.history['val_loss']).rolling(window=window, center=True).mean()
        
        ax4.plot(epochs, train_smooth, 'b-', label='Entrenamiento (suavizado)', linewidth=2)
        ax4.plot(epochs, val_smooth, 'r-', label='Validación (suavizado)', linewidth=2)
        ax4.set_title(f'Pérdida Suavizada (ventana={window})', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('MSE Loss (suavizado)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Pocas épocas para\nsuavizado', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Pérdida Suavizada', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Análisis de diagnóstico
    st.subheader("🔍 Análisis de Diagnóstico del Entrenamiento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Épocas Entrenadas", len(epochs))
        st.metric("Pérdida Final (Train)", f"{final_train_loss:.4f}")
        st.metric("Pérdida Final (Val)", f"{final_val_loss:.4f}")
    
    with col2:
        st.metric("Mejor Val Loss", f"{min_val_loss:.4f}")
        st.metric("Época del Mejor Modelo", min_val_epoch)
        overfitting_ratio = final_val_loss / final_train_loss
        st.metric("Ratio Final Val/Train", f"{overfitting_ratio:.2f}")
    
    with col3:
        # Análisis de tendencias
        recent_epochs = min(10, len(epochs) // 4)  # Últimas 25% de épocas o 10, lo que sea menor
        if recent_epochs > 1:
            recent_train_trend = np.polyfit(range(recent_epochs), 
                                          history.history['loss'][-recent_epochs:], 1)[0]
            recent_val_trend = np.polyfit(range(recent_epochs), 
                                        history.history['val_loss'][-recent_epochs:], 1)[0]
            
            st.metric("Tendencia Train (últimas épocas)", 
                     f"{'↓' if recent_train_trend < 0 else '↑'} {abs(recent_train_trend):.6f}")
            st.metric("Tendencia Val (últimas épocas)", 
                     f"{'↓' if recent_val_trend < 0 else '↑'} {abs(recent_val_trend):.6f}")
        
        # Estabilidad del entrenamiento
        train_stability = np.std(history.history['loss'][-recent_epochs:]) if recent_epochs > 1 else 0
        st.metric("Estabilidad Train", f"{train_stability:.4f}")
    
    # Diagnóstico de problemas
    st.subheader("⚠️ Diagnóstico de Problemas")
    
    diagnostics = []
    
    # 1. Overfitting
    if overfitting_ratio > 1.5:
        diagnostics.append("🔴 **Overfitting severo**: Ratio val/train > 1.5. Considera reducir complejidad del modelo o aumentar regularización.")
    elif overfitting_ratio > 1.2:
        diagnostics.append("🟡 **Overfitting moderado**: Ratio val/train > 1.2. Monitorear de cerca.")
    else:
        diagnostics.append("🟢 **Sin overfitting significativo**: Ratio val/train saludable.")
    
    # 2. Convergencia
    if len(epochs) >= 100:  # Solo si entrenó suficientes épocas
        if recent_train_trend > -1e-5:  # Pérdida de entrenamiento no está bajando
            diagnostics.append("🔴 **Problema de convergencia**: La pérdida de entrenamiento no está disminuyendo.")
        
        if recent_val_trend > 1e-5:  # Pérdida de validación está subiendo
            diagnostics.append("🟡 **Pérdida de validación creciente**: Posible overfitting o fin de mejora.")
    
    # 3. Estabilidad
    if train_stability > final_train_loss * 0.1:  # Variabilidad > 10% de la pérdida final
        diagnostics.append("🟡 **Entrenamiento inestable**: Alta variabilidad en las últimas épocas.")
    
    # 4. Early stopping
    epochs_since_best = len(epochs) - min_val_epoch
    if epochs_since_best > 20:
        diagnostics.append(f"🟡 **Early stopping**: {epochs_since_best} épocas desde el mejor modelo. Considera reducir paciencia.")
    
    # 5. Learning rate
    if final_val_loss > min_val_loss * 1.1:  # Pérdida final > 10% del mínimo
        diagnostics.append("🟡 **Posible learning rate alto**: La pérdida final es significativamente mayor que el mínimo alcanzado.")
    
    # Mostrar diagnósticos
    for diagnostic in diagnostics:
        st.markdown(diagnostic)
    
    if not diagnostics:
        st.success("🟢 **Entrenamiento exitoso**: No se detectaron problemas significativos.")
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones")
    
    recommendations = []
    
    if overfitting_ratio > 1.3:
        recommendations.append("- Aumentar dropout rate o añadir más regularización")
        recommendations.append("- Reducir complejidad del modelo (menos capas/neuronas)")
        recommendations.append("- Aumentar el tamaño del conjunto de entrenamiento")
    
    if recent_train_trend > -1e-5 and len(epochs) >= 50:
        recommendations.append("- Aumentar learning rate si está muy bajo")
        recommendations.append("- Verificar que los datos estén correctamente normalizados")
        recommendations.append("- Considerar una arquitectura diferente")
    
    if train_stability > final_train_loss * 0.1:
        recommendations.append("- Reducir learning rate para mayor estabilidad")
        recommendations.append("- Aumentar batch size")
        recommendations.append("- Añadir batch normalization")
    
    if not recommendations:
        recommendations.append("- El entrenamiento parece estar funcionando correctamente")
        recommendations.append("- Continuar monitoreando el rendimiento en datos de prueba")
    
    for rec in recommendations:
        st.markdown(rec)


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
    st.subheader(" Configuración del Nowcasting")
    
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
    
    # ==================== CONFIGURACIÓN DE ENTRENAMIENTO ====================
    st.subheader("⚙️ Configuración de Entrenamiento")
    
    # Crear dos columnas para los controles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("** Parámetros Principales**")
        
        # Learning Rate
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Tasa de aprendizaje del optimizador. Valores más altos = convergencia más rápida pero menos estable"
        )
        
        # Épocas
        epochs = st.number_input(
            "Número máximo de épocas",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            help="Número máximo de épocas de entrenamiento"
        )
        
        # Batch Size
        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64, 128, 256],
            index=2,  # 64 por defecto
            help="Tamaño del lote para entrenamiento. Valores más grandes = entrenamiento más estable"
        )
    
    with col2:
        st.markdown("**🛑 Control de Parada**")
        
        # Early Stopping
        use_early_stopping = st.checkbox(
            "Activar Early Stopping",
            value=True,
            help="Detener entrenamiento automáticamente cuando no mejore la validación"
        )
        
        # Paciencia Early Stopping
        if use_early_stopping:
            early_stopping_patience = st.number_input(
                "Paciencia Early Stopping",
                min_value=5,
                max_value=50,
                value=25,
                step=5,
                help="Número de épocas sin mejora antes de detener el entrenamiento"
            )
        else:
            early_stopping_patience = 25
        
        # Paciencia Reduce LR
        reduce_lr_patience = st.number_input(
            "Paciencia Reduce LR",
            min_value=3,
            max_value=30,
            value=12,
            step=2,
            help="Épocas sin mejora antes de reducir learning rate"
        )
    
    # Crear configuración de entrenamiento
    training_config = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'use_early_stopping': use_early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'reduce_lr_patience': reduce_lr_patience
    }
    
    # # Mostrar resumen de configuración de entrenamiento
    # with st.expander("📋 Resumen de Configuración de Entrenamiento"):
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.write("**Optimización:**")
    #         st.write(f"- Learning Rate: {learning_rate}")
    #         st.write(f"- Batch Size: {batch_size}")
    #         st.write(f"- Épocas máx: {epochs}")
        
    #     with col2:
    #         st.write("**Control de Parada:**")
    #         st.write(f"- Early Stopping: {'✅' if use_early_stopping else '❌'}")
    #         if use_early_stopping:
    #             st.write(f"- Paciencia ES: {early_stopping_patience}")
    #         st.write(f"- Paciencia LR: {reduce_lr_patience}")
        
    #     with col3:
    #         st.write("**Estimación:**")
    #         if use_early_stopping:
    #             est_time = "5-30 min"
    #             est_epochs = f"20-{min(epochs, early_stopping_patience + 20)}"
    #         else:
    #             est_time = f"{epochs * 0.1:.0f}-{epochs * 0.3:.0f} min"
    #             est_epochs = str(epochs)
    #         st.write(f"- Tiempo estimado: {est_time}")
    #         st.write(f"- Épocas esperadas: {est_epochs}")
    
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
    st.subheader(" Información del Conjunto de Datos")
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
        train_button = st.button("Entrenar Modelo Bayesiano", type="secondary")
    
    # Procesar acciones
    if train_button:
        if len(selected_features) == 0:
            st.error("❌ Selecciona al menos un grupo de variables")
        else:
            st.session_state.bnn_config_key = config_key
            st.session_state.bnn_show_results = False  # Reset visualización
            train_bayesian_model(
                nowcaster, train_data, test_data, model_type, sensor_seleccionado,
                training_config
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
        - ** Aplicable a cualquier ciudad**: No requiere datos históricos de NO₂
        - ** Predicción en tiempo real**: Basado en condiciones actuales
        - ** Cuantificación de incertidumbre**: Intervalos de confianza mediante Monte Carlo
        - ** Modelado bayesiano**: Captura incertidumbre epistémica y aleatoria
        
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
        ** Casos de uso ideales:**
        - Implementación en ciudades sin red de monitoreo
        - Validación de sensores existentes
        - Predicción en tiempo real para alertas
        - Estudios de impacto ambiental
        """)


def train_bayesian_model(nowcaster, train_data, test_data, model_type, sensor_id, 
                        training_config):
    """Entrena el modelo bayesiano con configuración personalizable."""
    
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
        
        st.write(f"**Datos preparados:**")
        st.write(f"- Entrenamiento: {X_train.shape}")
        st.write(f"- Evaluación: {X_test.shape}")
        st.write(f"- Características: {len(nowcaster.selected_features)}")
        
        # División de validación temporal (80% train, 20% val)
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        st.write(f"**División de datos:**")
        st.write(f"- Entrenamiento final: {X_train.shape[0]} muestras")
        st.write(f"- Validación: {X_val.shape[0]} muestras")
        st.write(f"- Prueba: {X_test.shape[0]} muestras")
        
        status_text.text("🔄 Normalizando datos...")
        progress_bar.progress(30)
        
        # Normalizar datos con verificación
        nowcaster.scaler_X = StandardScaler()
        X_train_scaled = nowcaster.scaler_X.fit_transform(X_train)
        X_val_scaled = nowcaster.scaler_X.transform(X_val)
        X_test_scaled = nowcaster.scaler_X.transform(X_test)
        
        nowcaster.scaler_y = StandardScaler()
        y_train_scaled = nowcaster.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = nowcaster.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Verificar normalización
        st.write(f"**Verificación de normalización:**")
        st.write(f"- X_train_scaled media: {X_train_scaled.mean():.3f} ± {X_train_scaled.std():.3f}")
        st.write(f"- y_train_scaled media: {y_train_scaled.mean():.3f} ± {y_train_scaled.std():.3f}")
        st.write(f"- X_train_scaled rango: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        st.write(f"- y_train_scaled rango: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        status_text.text("🔄 Creando modelo...")
        progress_bar.progress(40)
        
        # Crear y compilar modelo
        nowcaster.model = nowcaster.create_bayesian_model(X_train_scaled.shape[1], model_type)
        nowcaster.model = nowcaster.compile_model(nowcaster.model, learning_rate=training_config['learning_rate'])
        
        st.write(f"**Modelo creado:**")
        st.write(f"- Input shape: {X_train_scaled.shape[1]}")
        st.write(f"- Arquitectura: {MODEL_CONFIGS[model_type]['name']}")
        st.write(f"- Parámetros totales: {nowcaster.model.count_params():,}")
        
        # Mostrar resumen del modelo
        with st.expander("🔍 Ver arquitectura del modelo"):
            model_summary = []
            nowcaster.model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
        
        status_text.text("Entrenando modelo...")
        progress_bar.progress(50)
        
        # Crear contenedor para mostrar progreso de entrenamiento
        training_container = st.empty()
        
        # Mostrar configuración de entrenamiento
        with training_container.container():
            st.write("**Configuración de entrenamiento:**")
            st.write(f"- Learning rate: {training_config['learning_rate']}")
            st.write(f"- Batch size: {training_config['batch_size']}")
            st.write(f"- Épocas máximas: {training_config['epochs']}")
            st.write(f"- Early stopping: {'✅ Activado' if training_config['use_early_stopping'] else '❌ Desactivado'}")
            if training_config['use_early_stopping']:
                st.write(f"- Paciencia early stopping: {training_config['early_stopping_patience']} épocas")
            st.write(f"- Paciencia reduce LR: {training_config['reduce_lr_patience']} épocas")
        
        # Entrenar modelo con configuración personalizada
        history = nowcaster.train_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            use_early_stopping=training_config['use_early_stopping'],
            early_stopping_patience=training_config['early_stopping_patience'],
            reduce_lr_patience=training_config['reduce_lr_patience']
        )
        
        # Mostrar información del entrenamiento completado
        with training_container.container():
            st.write("✅ **Entrenamiento completado!**")
            final_epoch = len(history.history['loss'])
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            st.write(f"- Épocas entrenadas: {final_epoch}")
            st.write(f"- Pérdida final entrenamiento: {final_train_loss:.4f}")
            st.write(f"- Pérdida final validación: {final_val_loss:.4f}")
            st.write(f"- Ratio val/train loss: {final_val_loss/final_train_loss:.2f}")
            
            # Información adicional sobre early stopping
            if training_config['use_early_stopping'] and final_epoch < training_config['epochs']:
                st.write(f"- ⏹️ Early stopping activado en época {final_epoch}")
            elif final_epoch == training_config['epochs']:
                st.write(f"- 🏁 Entrenamiento completado (máximo de épocas alcanzado)")
        
        status_text.text(" Evaluando modelo...")
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
        
        # Verificar desnormalización
        st.write(f"📈 **Verificación de predicciones:**")
        st.write(f"- Predicciones rango: [{y_pred_mean.min():.2f}, {y_pred_mean.max():.2f}] µg/m³")
        st.write(f"- Incertidumbre media: {y_pred_std.mean():.2f} µg/m³")
        st.write(f"- Valores reales rango: [{y_test.min():.2f}, {y_test.max():.2f}] µg/m³")
        
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
            'mean_ci_width': interval_width,
            'training_epochs': final_epoch,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'training_config': training_config
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
        
        # Mostrar métricas preliminares
        st.write(f" **Métricas preliminares:**")
        st.write(f"- RMSE: {metrics['rmse']:.2f} µg/m³")
        st.write(f"- R²: {metrics['r2']:.3f}")
        st.write(f"- Cobertura IC 95%: {coverage_95:.1%}")
        st.write(f"- Ancho promedio IC: {interval_width:.2f} µg/m³")
        
        # Guardar resultados en session state
        st.session_state.bnn_model_trained = True
        st.session_state.bnn_metrics = metrics
        st.session_state.bnn_history = history
        st.session_state.bnn_test_data = test_data.iloc[-len(y_test):].copy()
        st.session_state.bnn_config = {
            'model_type': model_type,
            'features': nowcaster.selected_features,
            'sensor_id': sensor_id,
            'training_config': training_config
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
    
    st.header(" Resultados del Modelo Bayesiano")
    
    # Mostrar configuración del modelo
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Sensor:** {config['sensor_id']}")
        st.info(f"**Arquitectura:** {MODEL_CONFIGS[config['model_type']]['name']}")
    with col2:
        st.info(f"**Características:** {len(config['features'])}")
        # Mostrar configuración de entrenamiento si está disponible
        if 'training_config' in config:
            tc = config['training_config']
            st.info(f"**Learning Rate:** {tc['learning_rate']}")
    with col3:
        if 'training_config' in config:
            tc = config['training_config']
            st.info(f"**Épocas máx:** {tc['epochs']}")
            st.info(f"**Batch Size:** {tc['batch_size']}")
    
    # Mostrar configuración detallada de entrenamiento
    if 'training_config' in config:
        with st.expander("🔧 Configuración de Entrenamiento Utilizada"):
            tc = config['training_config']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Parámetros de Optimización:**")
                st.write(f"- Learning Rate: {tc['learning_rate']}")
                st.write(f"- Batch Size: {tc['batch_size']}")
                st.write(f"- Épocas máximas: {tc['epochs']}")
            
            with col2:
                st.write("**Control de Parada:**")
                st.write(f"- Early Stopping: {'✅ Activado' if tc['use_early_stopping'] else '❌ Desactivado'}")
                if tc['use_early_stopping']:
                    st.write(f"- Paciencia ES: {tc['early_stopping_patience']}")
                st.write(f"- Paciencia Reduce LR: {tc['reduce_lr_patience']}")
            
            with col3:
                st.write("**Resultados del Entrenamiento:**")
                if 'debug_info' in metrics and 'training_epochs' in metrics['debug_info']:
                    debug = metrics['debug_info']
                    st.write(f"- Épocas entrenadas: {debug['training_epochs']}")
                    st.write(f"- Pérdida final train: {debug['final_train_loss']:.4f}")
                    st.write(f"- Pérdida final val: {debug['final_val_loss']:.4f}")
                    
                    # Determinar si se activó early stopping
                    if tc['use_early_stopping'] and debug['training_epochs'] < tc['epochs']:
                        st.write("- ⏹️ Early stopping activado")
                    elif debug['training_epochs'] == tc['epochs']:
                        st.write("- 🏁 Épocas máximas alcanzadas")
    
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
    with st.expander(" Información Adicional"):
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
    st.subheader("Serie Temporal de NO₂")
    
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


class BayesianUnifiedNowcaster:
    """Clase unificada para nowcasting bayesiano individual y global."""
    
    def __init__(self):
        self.df_master = None
        self.individual_nowcaster = BayesianNowcaster()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'bnn_unified_data_loaded' not in st.session_state:
            st.session_state.bnn_unified_data_loaded = False
        if 'bnn_unified_mode' not in st.session_state:
            st.session_state.bnn_unified_mode = 'individual'
    
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
        st.header(" Overview del Dataset")
        
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


def show_individual_bayesian_training():
    """Interfaz optimizada para entrenamiento de modelos bayesianos individuales."""
    
    # Crear instancia del nowcaster
    nowcaster = BayesianNowcaster()
    
    # Cargar datos
    if not st.session_state.bnn_data_loaded:
        #if st.button("🔄 Cargar datos para Nowcasting Bayesiano Individual", type="primary", key="individual_load_data"):
        with st.spinner("Cargando datos..."):
            nowcaster.df_master = nowcaster.load_data()
            if not nowcaster.df_master.empty:
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
    
    # ==================== CONFIGURACIÓN BÁSICA ====================
    st.markdown("###  Configuración del Nowcasting Individual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de sensor
        sensores = sorted(nowcaster.df_master['id_no2'].unique())
        sensor_seleccionado = st.selectbox(
            "Sensor de NO₂", 
            sensores, 
            index=2 if len(sensores) > 2 else 0,
            key="individual_sensor_selection"
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
            key="individual_split_date"
        )
    
    with col2:
        # Configuración del modelo
        model_type = st.selectbox(
            "Arquitectura del modelo",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['name'],
            help="Selecciona la complejidad del modelo bayesiano",
            key="individual_model_architecture"
        )
        
        # Mostrar descripción del modelo
        st.info(f"📋 **{MODEL_CONFIGS[model_type]['name']}**: {MODEL_CONFIGS[model_type]['description']}")
    
    # ==================== SELECCIÓN DE VARIABLES (REUTILIZABLE) ====================
    selected_features = show_variable_selection(nowcaster.df_master, key_prefix="individual_")
    
    if not selected_features:
        st.warning("Selecciona al menos una variable para continuar.")
        return
    
    nowcaster.selected_features = selected_features
    
    # ==================== CONFIGURACIÓN DE ENTRENAMIENTO (REUTILIZABLE) ====================
    training_config = show_training_configuration(key_prefix="individual_")
    
    # ==================== RESUMEN DE CONFIGURACIÓN (REUTILIZABLE) ====================
    #ion_summary(sensor_seleccionado, selected_features, model_type, fecha_division)
    
    # ==================== PREPARAR DATOS (REUTILIZABLE) ====================
    result = prepare_training_data(df_sensor, selected_features, fecha_division)
    if result[0] is None:  # Error en preparación
        return
    
    df_clean, train_data, test_data = result
    
    # ==================== INFORMACIÓN DEL CONJUNTO DE DATOS (REUTILIZABLE) ====================
    show_dataset_info(df_clean, train_data, test_data, selected_features)
    
    # ==================== BOTONES DE ACCIÓN ====================
    config_key = f"individual_{sensor_seleccionado}_{model_type}_{len(selected_features)}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('individual_bnn_model_trained', False) and st.session_state.get('individual_bnn_config_key') == config_key:
            analyze_button = st.button("Ver Resultados del Modelo", type="primary", key="individual_analyze_btn")
        else:
            analyze_button = False
            if st.session_state.get('individual_bnn_model_trained', False):
                st.info("Configuración cambiada - Entrena un nuevo modelo")
            else:
                st.info("No existe un modelo entrenado con esta configuración")
    
    with col2:
        train_button = st.button("Entrenar Modelo Bayesiano Individual", type="secondary", key="individual_train_btn")
    
    # ==================== PROCESAR ACCIONES ====================
    if train_button:
        if len(selected_features) == 0:
            st.error("❌ Selecciona al menos un grupo de variables")
        else:
            st.session_state.individual_bnn_config_key = config_key
            st.session_state.individual_bnn_show_results = False
            train_individual_bayesian_model(
                nowcaster, train_data, test_data, model_type, sensor_seleccionado,
                training_config
            )
    
    elif analyze_button and st.session_state.get('individual_bnn_model_trained', False):
        st.session_state.individual_bnn_show_results = True
    
    # ==================== MOSTRAR RESULTADOS (REUTILIZABLE) ====================
    if st.session_state.get('individual_bnn_show_results', False) and st.session_state.get('individual_bnn_model_trained', False):
        show_individual_model_results()
    
    # Si no se ha entrenado un modelo, mostrar resumen de datos
    elif not st.session_state.get('individual_bnn_model_trained', False):
        show_data_summary(df_clean)


def train_individual_bayesian_model(nowcaster, train_data, test_data, model_type, sensor_id, training_config):
    """Entrena el modelo bayesiano individual con configuración personalizable."""
    
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
        
        st.write(f" **Datos preparados:**")
        st.write(f"- Entrenamiento: {X_train.shape}")
        st.write(f"- Evaluación: {X_test.shape}")
        st.write(f"- Características: {len(nowcaster.selected_features)}")
        
        # División de validación temporal (80% train, 20% val)
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        st.write(f" **División de datos:**")
        st.write(f"- Entrenamiento final: {X_train.shape[0]} muestras")
        st.write(f"- Validación: {X_val.shape[0]} muestras")
        st.write(f"- Prueba: {X_test.shape[0]} muestras")
        
        status_text.text("🔄 Normalizando datos...")
        progress_bar.progress(30)
        
        # Normalizar datos con verificación
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
        nowcaster.model = nowcaster.compile_model(nowcaster.model, learning_rate=training_config['learning_rate'])
        
        st.write(f"🧠 **Modelo creado:**")
        st.write(f"- Input shape: {X_train_scaled.shape[1]}")
        st.write(f"- Arquitectura: {MODEL_CONFIGS[model_type]['name']}")
        st.write(f"- Parámetros totales: {nowcaster.model.count_params():,}")
        
        status_text.text("Entrenando modelo...")
        progress_bar.progress(50)
        
        # Entrenar modelo con configuración personalizada
        history = nowcaster.train_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            use_early_stopping=training_config['use_early_stopping'],
            early_stopping_patience=training_config['early_stopping_patience'],
            reduce_lr_patience=training_config['reduce_lr_patience']
        )
        
        status_text.text(" Evaluando modelo...")
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
        
        # Calcular métricas con datos desnormalizados
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        within_ci = (y_test >= ci_lower) & (y_test <= ci_upper)
        coverage_95 = np.mean(within_ci)
        mean_uncertainty = np.mean(y_pred_std)
        interval_width = np.mean(ci_upper - ci_lower)
        
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
            'y_true': y_test
        }
        
        # Guardar resultados en session state
        st.session_state.individual_bnn_model_trained = True
        st.session_state.individual_bnn_metrics = metrics
        st.session_state.individual_bnn_history = history
        st.session_state.individual_bnn_test_data = test_data.iloc[-len(y_test):].copy()
        st.session_state.individual_bnn_config = {
            'model_type': model_type,
            'features': nowcaster.selected_features,
            'sensor_id': sensor_id,
            'training_config': training_config
        }
        st.session_state.individual_bnn_show_results = True
        
        progress_bar.progress(100)
        status_text.text("✅ Modelo entrenado correctamente!")
        
        st.success("🎉 ¡Modelo bayesiano individual entrenado exitosamente!")
        
        # Mostrar resultados automáticamente
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        import traceback
        st.error(f"❌ Error durante el entrenamiento: {str(e)}")
        st.error(f"📋 Detalles del error: {traceback.format_exc()}")
        progress_bar.empty()
        status_text.empty()


def show_individual_model_results():
    """Muestra los resultados del modelo bayesiano individual usando funciones reutilizables."""
    
    if 'individual_bnn_metrics' not in st.session_state:
        st.error("❌ No hay resultados de modelo individual disponibles")
        return
    
    metrics = st.session_state.individual_bnn_metrics
    history = st.session_state.individual_bnn_history
    test_data = st.session_state.individual_bnn_test_data
    config = st.session_state.individual_bnn_config
    
    # Usar función reutilizable para mostrar resultados completos
    show_complete_model_results(
        metrics, history, test_data, config, 
        title="Resultados del Modelo Bayesiano Individual"
    )


def show_global_bayesian_training(unified_nowcaster):
    """Interfaz optimizada para entrenamiento de modelos bayesianos globales."""
    st.subheader("Nowcasting Bayesiano Global Multi-Sensor")
    
    # ==================== CONFIGURACIÓN DE SENSORES ====================
    st.markdown("### 📡 Configuración de Sensores")
    
    sensores_disponibles = sorted(unified_nowcaster.df_master['id_no2'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sensores para Entrenamiento**")
        sensores_train = st.multiselect(
            "Selecciona sensores para entrenar:",
            sensores_disponibles,
            default=sensores_disponibles[:-2],  # Todos menos los últimos 2
            key="global_bnn_train_sensors"
        )
    
    with col2:
        st.markdown("**Sensores para Evaluación**")
        sensores_test = st.multiselect(
            "Selecciona sensores para evaluar:",
            sensores_disponibles,
            default=sensores_disponibles[-2:],  # Los últimos 2
            key="global_bnn_test_sensors"
        )
    
    # Validaciones
    if not sensores_train:
        st.warning("⚠️ Selecciona al menos un sensor para entrenamiento")
        return
    
    if not sensores_test:
        st.warning("⚠️ Selecciona al menos un sensor para evaluación")
        return
    
    # Estadísticas de la configuración
    df_train = unified_nowcaster.df_master[unified_nowcaster.df_master['id_no2'].isin(sensores_train)]
    df_test = unified_nowcaster.df_master[unified_nowcaster.df_master['id_no2'].isin(sensores_test)]
    
    st.markdown("###  Estadísticas de la Configuración")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sensores entrenamiento", len(sensores_train))
    with col2:
        st.metric("Registros entrenamiento", f"{len(df_train):,}")
    with col3:
        st.metric("Sensores evaluación", len(sensores_test))
    with col4:
        st.metric("Registros evaluación", f"{len(df_test):,}")
    
    # ==================== CONFIGURACIÓN BÁSICA ====================
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Arquitectura del modelo",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['name'],
            help="Selecciona la complejidad del modelo bayesiano",
            key="global_bnn_model_architecture"
        )
    
    with col2:
        fecha_division = st.date_input(
            "Fecha de división temporal",
            value=pd.to_datetime('2024-01-01').date(),
            key="global_bnn_split_date"
        )
    
    # ==================== SELECCIÓN DE VARIABLES (REUTILIZABLE) ====================
    selected_features = show_variable_selection(unified_nowcaster.df_master, key_prefix="global_")
    
    if not selected_features:
        st.warning("⚠️ No hay variables disponibles. Verifica los datos.")
        return
    
    # Validar que tenemos features suficientes
    if len(selected_features) < 5:
        st.error(f"❌ Muy pocas variables disponibles: {len(selected_features)}")
        return
    
    # ==================== CONFIGURACIÓN DE ENTRENAMIENTO (REUTILIZABLE) ====================
    training_config = show_training_configuration(key_prefix="global_")
    
    # ==================== RESUMEN DE CONFIGURACIÓN ====================
    additional_info = {
        "Sensores entrenamiento": f"{len(sensores_train)} sensores",
        "Sensores evaluación": f"{len(sensores_test)} sensores",
        "Modo": "Modelo Global Multi-Sensor"
    }
    
    
    # ==================== BOTONES DE ACCIÓN ====================
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Entrenar Modelo Bayesiano Global", type="primary", key="global_train_btn"):
            train_global_bayesian_model(
                unified_nowcaster, sensores_train, sensores_test, 
                selected_features, model_type, fecha_division, training_config
            )
    
    # ==================== MOSTRAR RESULTADOS EXISTENTES ====================
    if 'global_bnn_model_results' in st.session_state:
        results = st.session_state.global_bnn_model_results
        
        st.divider()
        st.subheader(" Resultados del Modelo Bayesiano Global")
        
        # Mostrar métricas globales
        st.markdown("### 📈 Métricas Globales")
        show_model_metrics(results['metrics'])
        
        # Mostrar análisis por sensor
        show_global_bayesian_sensor_analysis(
            results['test_df'], 
            results['model'], 
            results['selected_features'], 
            results['scaler_X'],
            results['scaler_y'],
            results['sensores_test']
        )


def train_global_bayesian_model(unified_nowcaster, sensores_train, sensores_test, 
                               selected_features, model_type, fecha_division, training_config):
    """Entrena un modelo bayesiano global con la configuración especificada."""
    
    with st.spinner("Entrenando modelo bayesiano global..."):
        # Limpiar contexto de TensorFlow para evitar conflictos
        try:
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        except:
            pass  # Ignorar errores de limpieza
        
        # Preparar datos
        df_train = unified_nowcaster.df_master[unified_nowcaster.df_master['id_no2'].isin(sensores_train)]
        df_test = unified_nowcaster.df_master[unified_nowcaster.df_master['id_no2'].isin(sensores_test)]
        
        # Aplicar preprocesamiento ANTES de seleccionar features
        df_train = unified_nowcaster.individual_nowcaster.create_temporal_features(df_train)
        df_train = unified_nowcaster.individual_nowcaster.convert_units(df_train)
        
        df_test = unified_nowcaster.individual_nowcaster.create_temporal_features(df_test)
        df_test = unified_nowcaster.individual_nowcaster.convert_units(df_test)
        
        # VALIDAR QUE TODAS LAS FEATURES EXISTEN
        missing_features_train = [f for f in selected_features if f not in df_train.columns]
        missing_features_test = [f for f in selected_features if f not in df_test.columns]
        
        if missing_features_train or missing_features_test:
            st.error(f"❌ Features faltantes en train: {missing_features_train}")
            st.error(f"❌ Features faltantes en test: {missing_features_test}")
            return
        
        # División temporal adicional
        fecha_division_dt = pd.to_datetime(fecha_division)
        df_train = df_train[df_train['fecha'] <= fecha_division_dt]
        df_test = df_test[df_test['fecha'] > fecha_division_dt]
        
        # Preparar datos para nowcasting SIN DATA LEAKAGE
        X_train, y_train = unified_nowcaster.individual_nowcaster.prepare_nowcasting_data(df_train, selected_features)
        X_test, y_test = unified_nowcaster.individual_nowcaster.prepare_nowcasting_data(df_test, selected_features)
        
        # Validar que tenemos datos suficientes
        if len(X_train) < 500 or len(X_test) < 100:
            st.error("❌ No hay datos suficientes después de la limpieza")
            return
        
        st.write(f" **Datos preparados:**")
        st.write(f"- Entrenamiento: {X_train.shape}")
        st.write(f"- Evaluación: {X_test.shape}")
        st.write(f"- Características: {len(selected_features)}")
        
        # División de validación temporal (80% train, 20% val)
        val_split = int(0.8 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        # ESCALADO SIN DATA LEAKAGE - Solo usar X_train para fit
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)  # FIT solo con entrenamiento
        X_val_scaled = scaler_X.transform(X_val)          # TRANSFORM validación
        X_test_scaled = scaler_X.transform(X_test)        # TRANSFORM test
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()  # FIT solo con entrenamiento
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()          # TRANSFORM validación
        
        st.write(f" **División final:**")
        st.write(f"- Entrenamiento: {X_train_scaled.shape[0]} muestras")
        st.write(f"- Validación: {X_val_scaled.shape[0]} muestras")
        st.write(f"- Evaluación: {X_test_scaled.shape[0]} muestras")
        
        # AGREGAR: Información de escalado para debug
        st.write(f"🔍 **Información de escalado:**")
        st.write(f"- Scaler Y mean: {scaler_y.mean_[0]:.2f}")
        st.write(f"- Scaler Y scale: {scaler_y.scale_[0]:.2f}")
        st.write(f"- y_train original rango: [{y_train.min():.2f}, {y_train.max():.2f}]")
        st.write(f"- y_test original rango: [{y_test.min():.2f}, {y_test.max():.2f}]")
        st.write(f"- y_train_scaled rango: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # Crear y compilar modelo
        nowcaster = BayesianNowcaster()
        nowcaster.model = nowcaster.create_bayesian_model(X_train_scaled.shape[1], model_type)
        nowcaster.model = nowcaster.compile_model(nowcaster.model, learning_rate=training_config['learning_rate'])
        nowcaster.scaler_X = scaler_X
        nowcaster.scaler_y = scaler_y
        
        st.write(f"🧠 **Modelo creado:**")
        st.write(f"- Arquitectura: {MODEL_CONFIGS[model_type]['name']}")
        st.write(f"- Parámetros totales: {nowcaster.model.count_params():,}")
        
        # Entrenar modelo
        history = nowcaster.train_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            use_early_stopping=training_config['use_early_stopping'],
            early_stopping_patience=training_config['early_stopping_patience'],
            reduce_lr_patience=training_config['reduce_lr_patience']
        )
        
        # CORREGIR: Evaluar modelo global con datos correctamente escalados
        # El problema era que se pasaba y_test sin escalar pero X_test escalado
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        metrics = nowcaster.evaluate_model(X_test_scaled, y_test_scaled, n_samples=100)
        
        # IMPORTANTE: Desnormalizar las predicciones en metrics para visualización
        # Las métricas internas están en escala normalizada, necesitamos desnormalizarlas
        metrics_denormalized = {
            'mse': metrics['mse'] * (scaler_y.scale_[0] ** 2),  # MSE se escala por σ²
            'mae': metrics['mae'] * scaler_y.scale_[0],         # MAE se escala por σ
            'rmse': metrics['rmse'] * scaler_y.scale_[0],       # RMSE se escala por σ
            'r2': metrics['r2'],  # R² es invariante al escalado
            'mean_uncertainty': metrics['mean_uncertainty'] * scaler_y.scale_[0],
            'coverage_95': metrics['coverage_95'],  # Cobertura es invariante
            'interval_width': metrics['interval_width'] * scaler_y.scale_[0],
            'predictions_mean': scaler_y.inverse_transform(metrics['predictions_mean'].reshape(-1, 1)).flatten(),
            'predictions_std': metrics['predictions_std'] * scaler_y.scale_[0],
            'ci_lower': scaler_y.inverse_transform(metrics['ci_lower'].reshape(-1, 1)).flatten(),
            'ci_upper': scaler_y.inverse_transform(metrics['ci_upper'].reshape(-1, 1)).flatten(),
            'y_true': y_test,  # Valores reales sin escalar
            'debug_info': {
                **metrics['debug_info'],
                'scaling_applied': True,
                'scaler_mean': float(scaler_y.mean_[0]),
                'scaler_scale': float(scaler_y.scale_[0])
            }
        }
        
        # Guardar resultados en session_state
        st.session_state.global_bnn_model_results = {
            'model': nowcaster.model,
            'metrics': metrics_denormalized,  # Usar métricas desnormalizadas
            'history': history,
            'test_df': df_test,
            'sensores_train': sensores_train,
            'sensores_test': sensores_test,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'selected_features': selected_features,
            'model_type': model_type,
            'training_config': training_config
        }
        
        st.success("✅ Modelo bayesiano global entrenado exitosamente!")
        st.info(" Los resultados se muestran a continuación...")
        
        # AGREGAR: Verificación de métricas para debug
        st.write(f"🔍 **Verificación de métricas:**")
        st.write(f"- RMSE global (desnormalizado): {metrics_denormalized['rmse']:.2f} µg/m³")
        st.write(f"- RMSE global (normalizado): {metrics['rmse']:.3f}")
        st.write(f"- R² global: {metrics_denormalized['r2']:.3f}")
        st.write(f"- Cobertura IC 95%: {metrics_denormalized['coverage_95']:.1%}")
        st.write(f"- Predicciones desnormalizadas rango: [{metrics_denormalized['predictions_mean'].min():.2f}, {metrics_denormalized['predictions_mean'].max():.2f}]")
        st.write(f"- Valores reales rango: [{y_test.min():.2f}, {y_test.max():.2f}]")
        
        # AGREGAR: Validación completa de data leakage
        st.markdown("---")
        st.markdown("### 🔍 Metodología de División de Datos")
        
        # Calcular tamaños originales antes de división de validación
        original_train_size = len(X_train) + len(X_val)
        
        st.info(f"""
        ** División de datos aplicada:**
        - **Datos originales**: {original_train_size} muestras de entrenamiento, {X_test.shape[0]} de evaluación
        - **División temporal**: Datos antes/después de {fecha_division}
        - **División de validación**: 80% entrenamiento ({X_train.shape[0]} muestras), 20% validación ({X_val.shape[0]} muestras)
        - **Escalado**: Ajustado SOLO con los {X_train.shape[0]} muestras de entrenamiento final
        
        **✅ Metodología correcta:**
        - Los scalers se ajustan SOLO con el conjunto de entrenamiento final
        - Los datos de validación y test se transforman (NO se usan para ajustar)
        - No hay data leakage temporal ni de validación cruzada
        """)
        
        
        # Mostrar resumen final
        st.markdown("---")
        st.markdown("###  Resumen del Entrenamiento")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
            **✅ Modelo Entrenado**
            - Arquitectura: {MODEL_CONFIGS[model_type]['name']}
            - Parámetros: {nowcaster.model.count_params():,}
            - Sensores train: {len(sensores_train)}
            - Sensores test: {len(sensores_test)}
            """)
        
        with col2:
            st.info(f"""
            ** Datos Utilizados**
            - Train final: {X_train.shape[0]:,} muestras
            - Validación: {X_val.shape[0]:,} muestras  
            - Evaluación: {X_test.shape[0]:,} muestras
            - Características: {len(selected_features)}
            """)
        
        with col3:
            st.metric(
                "RMSE Global", 
                f"{metrics_denormalized['rmse']:.2f} µg/m³",
                help="Error cuadrático medio en escala original"
            )
            st.metric(
                "R² Global",
                f"{metrics_denormalized['r2']:.3f}",
                help="Coeficiente de determinación"
            )


def show_global_bayesian_sensor_analysis(test_df, model, selected_features, scaler_X, scaler_y, sensores_test):
    """Muestra análisis detallado por sensor para el modelo bayesiano global."""
    
    st.subheader(" Análisis por Sensor de Evaluación")
    
    # Calcular métricas por sensor
    sensor_metrics = []
    
    # Crear instancia local de nowcaster para prepare_nowcasting_data
    local_nowcaster = BayesianNowcaster()
    
    for sensor_id in sensores_test:
        sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
        
        if len(sensor_data) == 0:
            continue
        
        # Preparar datos del sensor específico
        X_sensor, y_sensor = local_nowcaster.prepare_nowcasting_data(
            sensor_data, selected_features
        )
        
        if len(X_sensor) == 0:
            continue
        
        # APLICAR EL MISMO ESCALADO QUE EN EL ENTRENAMIENTO
        X_sensor_scaled = scaler_X.transform(X_sensor)
        
        # Predicciones con incertidumbre con manejo robusto de errores
        try:
            nowcaster = BayesianNowcaster()
            nowcaster.model = model
            
            # Validar que el modelo existe
            if nowcaster.model is None:
                st.warning(f"⚠️ Modelo no disponible para sensor {sensor_id}")
                continue
            
            # Intentar predicción con manejo de errores
            y_pred_mean_scaled, y_pred_std_scaled = nowcaster.predict_with_uncertainty(
                X_sensor_scaled, n_samples=50  # Reducir samples para evitar problemas
            )
            
        except Exception as e:
            st.warning(f"⚠️ Error al procesar sensor {sensor_id}: {str(e)}")
            st.info("Intentando con predicción estándar como fallback...")
            
            try:
                # Fallback: predicción estándar sin Monte Carlo
                nowcaster = BayesianNowcaster()
                nowcaster.model = model
                y_pred_mean_scaled = model(X_sensor_scaled, training=False).numpy()
                y_pred_std_scaled = np.ones_like(y_pred_mean_scaled) * 0.1  # Incertidumbre mínima
            except Exception as e2:
                st.error(f"❌ Error crítico con sensor {sensor_id}: {str(e2)}")
                continue
        
        # Desnormalizar predicciones
        y_pred_mean = scaler_y.inverse_transform(y_pred_mean_scaled.reshape(-1, 1)).flatten()
        y_pred_std = y_pred_std_scaled.flatten() * scaler_y.scale_[0]
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_sensor, y_pred_mean))
        r2 = r2_score(y_sensor, y_pred_mean)
        mae = mean_absolute_error(y_sensor, y_pred_mean)
        
        # Métricas de incertidumbre
        ci_lower = y_pred_mean - 1.96 * y_pred_std
        ci_upper = y_pred_mean + 1.96 * y_pred_std
        within_ci = (y_sensor >= ci_lower) & (y_sensor <= ci_upper)
        coverage_95 = np.mean(within_ci)
        mean_uncertainty = np.mean(y_pred_std)
        
        sensor_metrics.append({
            'sensor_id': sensor_id,
            'n_samples': len(sensor_data),
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'coverage_95': coverage_95,
            'mean_uncertainty': mean_uncertainty,
            'no2_mean': y_sensor.mean(),
            'no2_std': y_sensor.std()
        })
    
    sensor_metrics_df = pd.DataFrame(sensor_metrics)
    
    # Validar que tenemos métricas
    if sensor_metrics_df.empty:
        st.error("❌ No se pudieron calcular métricas por sensor")
        return
    
    # Mostrar métricas resumidas en columnas
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
            "Cobertura IC 95%",
            f"{sensor_metrics_df['coverage_95'].mean():.1%}",
            f"±{sensor_metrics_df['coverage_95'].std():.1%}"
        )
    
    with col4:
        st.metric(
            "Incertidumbre Promedio",
            f"{sensor_metrics_df['mean_uncertainty'].mean():.2f} µg/m³",
            f"±{sensor_metrics_df['mean_uncertainty'].std():.2f}"
        )

    
    # Mostrar tabla de métricas
    with st.expander("📋 Métricas Detalladas por Sensor"):
        st.dataframe(
            sensor_metrics_df.style.format({
                'rmse': '{:.2f}',
                'r2': '{:.3f}',
                'mae': '{:.2f}',
                'coverage_95': '{:.1%}',
                'mean_uncertainty': '{:.2f}',
                'no2_mean': '{:.2f}',
                'no2_std': '{:.2f}'
            }),
            use_container_width=True
        )
    
    # Selector para análisis individual
    st.markdown("### 🔍 Análisis Detallado por Sensor")
    
    sensor_seleccionado = st.selectbox(
        "Selecciona sensor para análisis detallado:",
        sensores_test,
        key="global_bnn_analysis_sensor"
    )
    
    # Usar checkbox en lugar de botón para evitar rerun
    if st.checkbox("📈 Mostrar Análisis Detallado", key="show_detailed_bnn_analysis"):
        show_detailed_bayesian_sensor_analysis(test_df, model, selected_features, scaler_X, scaler_y, sensor_seleccionado)


def show_detailed_bayesian_sensor_analysis(test_df, model, selected_features, scaler_X, scaler_y, sensor_id):
    """Muestra análisis detallado para un sensor específico."""
    
    sensor_data = test_df[test_df['id_no2'] == sensor_id].copy()
    
    if len(sensor_data) == 0:
        st.error(f"No hay datos para el sensor {sensor_id}")
        return
    
    # Preparar datos del sensor
    nowcaster = BayesianNowcaster()
    X_sensor, y_sensor = nowcaster.prepare_nowcasting_data(sensor_data, selected_features)
    
    if len(X_sensor) == 0:
        st.error(f"No se pudieron preparar datos para el sensor {sensor_id}")
        return
    
    # Aplicar escalado
    X_sensor_scaled = scaler_X.transform(X_sensor)
    
    # Predicciones con incertidumbre
    nowcaster.model = model
    y_pred_mean_scaled, y_pred_std_scaled = nowcaster.predict_with_uncertainty(
        X_sensor_scaled, n_samples=100
    )
    
    # Desnormalizar
    y_pred_mean = scaler_y.inverse_transform(y_pred_mean_scaled.reshape(-1, 1)).flatten()
    y_pred_std = y_pred_std_scaled.flatten() * scaler_y.scale_[0]
    
    st.subheader(f" Análisis Detallado - Sensor {sensor_id}")
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_sensor, y_pred_mean))
    r2 = r2_score(y_sensor, y_pred_mean)
    mae = mean_absolute_error(y_sensor, y_pred_mean)
    
    # Métricas de incertidumbre
    ci_lower = y_pred_mean - 1.96 * y_pred_std
    ci_upper = y_pred_mean + 1.96 * y_pred_std
    within_ci = (y_sensor >= ci_lower) & (y_sensor <= ci_upper)
    coverage_95 = np.mean(within_ci)
    mean_uncertainty = np.mean(y_pred_std)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.2f} µg/m³")
    with col2:
        st.metric("R²", f"{r2:.3f}")
    with col3:
        st.metric("Cobertura IC 95%", f"{coverage_95:.1%}")
    with col4:
        st.metric("Incertidumbre Media", f"{mean_uncertainty:.2f} µg/m³")
    
    # Crear métricas para visualización
    sensor_metrics = {
        'predictions_mean': y_pred_mean,
        'predictions_std': y_pred_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'y_true': y_sensor
    }
    
    # Mostrar predicciones con incertidumbre
    show_uncertainty_predictions(sensor_data.iloc[-len(y_sensor):], sensor_metrics)


def show_bayesian_info_panel():
    """Muestra panel de información sobre el nowcasting bayesiano unificado."""
    with st.expander("ℹ️ Acerca del Nowcasting Bayesiano Unificado", expanded=False):
        st.markdown("""
        **🧠 Nowcasting Bayesiano Unificado**
        
        Este módulo permite entrenar y comparar dos tipos de modelos bayesianos:
        
        **Modelos Individuales:**
        - Un modelo bayesiano por sensor
        - Especializado en patrones locales específicos
        - Ideal para análisis detallado por ubicación
        - Cuantificación de incertidumbre personalizada
        
        **Modelos Globales:**
        - Un modelo entrenado con múltiples sensores
        - Aprende patrones universales transferibles
        - Ideal para nowcasting en nuevas ubicaciones
        - Incertidumbre robusta multi-sensor
        
        **Configuración Experimental:**
        - Selecciona sensores para entrenamiento vs evaluación
        - Evalúa transferibilidad entre ubicaciones
        - Análisis detallado de incertidumbre por sensor
        - Monte Carlo Dropout para cuantificación bayesiana
        
        **Ventajas del Enfoque Bayesiano:**
        - **Cuantificación de incertidumbre**: Intervalos de confianza
        - **Robustez**: Manejo de datos ruidosos
        - **Transferibilidad**: Aplicable a ciudades sin sensores
        - **Interpretabilidad**: Confianza en las predicciones
        
        **⚠️ Nota importante:**
        Los modelos NO utilizan valores históricos de NO₂, permitiendo su 
        aplicación directa a ciudades sin sensores de calidad del aire.
        """)


# ==================== FUNCIÓN PRINCIPAL UNIFICADA ====================

def bayesian_nowcasting_page():
    """Función principal unificada para la página de nowcasting bayesiano."""
        
    # Panel de información
    show_bayesian_info_panel()
    
    # Inicializar trainer unificado
    unified_nowcaster = BayesianUnifiedNowcaster()
    
    # Cargar datos
    if not st.session_state.bnn_unified_data_loaded:
        if st.button(" Cargar Dataset Completo", type="primary", key="unified_bnn_load_data"):
            with st.spinner("Cargando dataset completo..."):
                unified_nowcaster.df_master = unified_nowcaster.load_data()
                if not unified_nowcaster.df_master.empty:
                    # Aplicar preprocesamiento
                    unified_nowcaster.df_master = unified_nowcaster.individual_nowcaster.create_temporal_features(unified_nowcaster.df_master)
                    unified_nowcaster.df_master = unified_nowcaster.individual_nowcaster.convert_units(unified_nowcaster.df_master)
                    st.session_state.bnn_unified_data_loaded = True
                    st.success("¡Dataset cargado exitosamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    unified_nowcaster.df_master = unified_nowcaster.load_data()
    if not unified_nowcaster.df_master.empty:
        unified_nowcaster.df_master = unified_nowcaster.individual_nowcaster.create_temporal_features(unified_nowcaster.df_master)
        unified_nowcaster.df_master = unified_nowcaster.individual_nowcaster.convert_units(unified_nowcaster.df_master)
    
    if unified_nowcaster.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Mostrar overview
    unified_nowcaster.show_data_overview()
    
    # Selector de modo
    st.header(" Selecciona Tipo de Modelo")
    
    mode = st.radio(
        "Tipo de nowcasting:",
        ["Individual (por sensor)", "Global (multi-sensor)"],
        index=0 if st.session_state.bnn_unified_mode == 'individual' else 1,
        horizontal=True
    )
    
    # Actualizar estado
    if "Individual" in mode:
        st.session_state.bnn_unified_mode = 'individual'
    else:
        st.session_state.bnn_unified_mode = 'global'
    
    st.divider()
    
    # Mostrar interfaz según el modo
    if st.session_state.bnn_unified_mode == 'individual':
        show_individual_bayesian_training()
    else:
        show_global_bayesian_training(unified_nowcaster)


# ==================== FUNCIONES ORIGINALES PRESERVADAS ====================

# Mantener todas las funciones originales para compatibilidad
def show_info_panel():
    """Mantener función original para compatibilidad."""
    show_bayesian_info_panel()


# ==================== FUNCIONES REUTILIZABLES PARA CONFIGURACIÓN ====================

def show_variable_selection(df_master, key_prefix=""):
    """Función reutilizable para selección de variables por categorías."""
    st.markdown("### 🔧 Selección de Variables")
    
    # Crear tabs para categorías usando VARIABLE_CATEGORIES
    var_tabs = st.tabs(list(VARIABLE_CATEGORIES.keys()))
    
    selected_features = []
    for i, (category, vars_list) in enumerate(VARIABLE_CATEGORIES.items()):
        with var_tabs[i]:
            # Filtrar variables que existen en los datos
            available_vars = [var for var in vars_list if var in df_master.columns or 'sin' in var or 'cos' in var]
            
            # Configurar defaults - todas las variables disponibles
            default_vars = available_vars
            
            selected_in_category = st.multiselect(
                f"Variables de {category}",
                available_vars,
                default=default_vars,
                help=f"Selecciona las variables de {category.lower()} para el modelo",
                key=f"{key_prefix}vars_{category.replace(' ', '_').lower()}"
            )
            selected_features.extend(selected_in_category)
    
    return selected_features


def show_training_configuration(key_prefix=""):
    """Función reutilizable para configuración de entrenamiento."""
    st.markdown("### ⚙️ Configuración de Entrenamiento")
    
    # Crear dos columnas para los controles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parámetros Principales**")
        
        # Learning Rate
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Tasa de aprendizaje del optimizador. Valores más altos = convergencia más rápida pero menos estable",
            key=f"{key_prefix}learning_rate"
        )
        
        # Épocas
        epochs = st.number_input(
            "Número máximo de épocas",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            help="Número máximo de épocas de entrenamiento",
            key=f"{key_prefix}epochs"
        )
        
        # Batch Size
        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64, 128, 256],
            index=2,  # 64 por defecto
            help="Tamaño del lote para entrenamiento. Valores más grandes = entrenamiento más estable",
            key=f"{key_prefix}batch_size"
        )
    
    with col2:
        st.markdown("**🛑 Control de Parada**")
        
        # Early Stopping
        use_early_stopping = st.checkbox(
            "Activar Early Stopping",
            value=True,
            help="Detener entrenamiento automáticamente cuando no mejore la validación",
            key=f"{key_prefix}early_stopping"
        )
        
        # Paciencia Early Stopping
        if use_early_stopping:
            early_stopping_patience = st.number_input(
                "Paciencia Early Stopping",
                min_value=5,
                max_value=50,
                value=25,
                step=5,
                help="Número de épocas sin mejora antes de detener el entrenamiento",
                key=f"{key_prefix}es_patience"
            )
        else:
            early_stopping_patience = 25
        
        # Paciencia Reduce LR
        reduce_lr_patience = st.number_input(
            "Paciencia Reduce LR",
            min_value=3,
            max_value=30,
            value=12,
            step=2,
            help="Épocas sin mejora antes de reducir learning rate",
            key=f"{key_prefix}lr_patience"
        )
    
    # Crear configuración de entrenamiento
    training_config = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'use_early_stopping': use_early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'reduce_lr_patience': reduce_lr_patience
    }
    
    # # Mostrar resumen de configuración de entrenamiento
    # with st.expander("📋 Resumen de Configuración de Entrenamiento"):
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.write("**Optimización:**")
    #         st.write(f"- Learning Rate: {learning_rate}")
    #         st.write(f"- Batch Size: {batch_size}")
    #         st.write(f"- Épocas máx: {epochs}")
        
    #     with col2:
    #         st.write("**Control de Parada:**")
    #         st.write(f"- Early Stopping: {'✅' if use_early_stopping else '❌'}")
    #         if use_early_stopping:
    #             st.write(f"- Paciencia ES: {early_stopping_patience}")
    #         st.write(f"- Paciencia LR: {reduce_lr_patience}")
        
    #     with col3:
    #         st.write("**Estimación:**")
    #         if use_early_stopping:
    #             est_time = "5-30 min"
    #             est_epochs = f"20-{min(epochs, early_stopping_patience + 20)}"
    #         else:
    #             est_time = f"{epochs * 0.1:.0f}-{epochs * 0.3:.0f} min"
    #             est_epochs = str(epochs)
    #         st.write(f"- Tiempo estimado: {est_time}")
    #         st.write(f"- Épocas esperadas: {est_epochs}")
    
    return training_config


def show_dataset_info(df_clean, train_data, test_data, selected_features):
    """Función reutilizable para mostrar información del conjunto de datos."""
    st.markdown("### Información del Conjunto de Datos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Registros totales", len(df_clean))
    with col2:
        st.metric("Entrenamiento", len(train_data))
    with col3:
        st.metric("Evaluación", len(test_data))
    with col4:
        st.metric("Variables", len(selected_features))


def prepare_training_data(df_sensor, selected_features, fecha_division):
    """Función reutilizable para preparar datos de entrenamiento."""
    with st.spinner("Preparando datos para entrenamiento..."):
        df_processed = df_sensor.copy()
        
        # Limpiar datos
        df_clean = df_processed.dropna(subset=selected_features + ['no2_value'])
        
        if len(df_clean) < 1000:
            st.error("❌ Datos insuficientes después de limpiar NaN")
            return None, None, None
        
        # Dividir datos temporalmente
        fecha_division_dt = pd.to_datetime(fecha_division)
        train_data = df_clean[df_clean['fecha'] <= fecha_division_dt]
        test_data = df_clean[df_clean['fecha'] > fecha_division_dt]
        
        if len(train_data) < 500 or len(test_data) < 100:
            st.error("❌ No hay suficientes datos para entrenamiento o evaluación.")
            return None, None, None
    
    return df_clean, train_data, test_data


def show_model_configuration_summary(config):
    """Función reutilizable para mostrar configuración del modelo entrenado."""
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Sensor:** {config['sensor_id']}")
        st.info(f"**Arquitectura:** {MODEL_CONFIGS[config['model_type']]['name']}")
    with col2:
        st.info(f"**Características:** {len(config['features'])}")
        if 'training_config' in config:
            tc = config['training_config']
            st.info(f"**Learning Rate:** {tc['learning_rate']}")
    with col3:
        if 'training_config' in config:
            tc = config['training_config']
            st.info(f"**Épocas máx:** {tc['epochs']}")
            st.info(f"**Batch Size:** {tc['batch_size']}")


def show_complete_model_results(metrics, history, test_data, config, title="Resultados del Modelo Bayesiano"):
    """Función reutilizable para mostrar resultados completos del modelo."""
    st.header(f" {title}")
    
    # Mostrar configuración del modelo
    show_model_configuration_summary(config)
    
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
