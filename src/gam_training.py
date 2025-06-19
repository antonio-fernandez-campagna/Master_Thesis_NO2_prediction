"""
Módulo refactorizado para entrenamiento y análisis de modelos GAM (Generalized Additive Models).

Este módulo proporciona una interfaz integrada para entrenar y analizar modelos GAM
para predecir niveles de NO2 basándose en variables de tráfico, meteorológicas y temporales.
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import seaborn as sns
import joblib
import os
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings

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
        'weekend', 'season_sin', 'season_cos'
    ],
    "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
    "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp']
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
    'vmed': {'name': 'Velocidad Media', 'unit': 'km/h', 'typical_range': (0, 100)}
}

# COLUMNS_FOR_OUTLIERS = [
#     'no2_value', 'intensidad', 'carga', 'ocupacion', 'vmed',
#     'd2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp'
# ]

COLUMNS_FOR_OUTLIERS = [
    'no2_value'
]



# ==================== CLASE PRINCIPAL ====================

class GAMTrainer:
    """Clase principal para entrenamiento y análisis de modelos GAM."""
    
    def __init__(self):
        self.df_master = None
        self.model = None
        self.scaler_dict = {}
        self.scaler_target = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'training_data_loaded' not in st.session_state:
            st.session_state.training_data_loaded = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'training_config' not in st.session_state:
            st.session_state.training_config = {}
    
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
        
        # Viento (m/s -> km/h)
        if 'u10' in df.columns:
            df['u10'] = df['u10'] * 3.6
        if 'v10' in df.columns:
            df['v10'] = df['v10'] * 3.6
        
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
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
        
        elif method == 'zscore':
            from scipy.stats import zscore
            z_scores = df_filtered[COLUMNS_FOR_OUTLIERS].apply(zscore)
            condition = (z_scores.abs() < 3.0).all(axis=1)
            df_filtered = df_filtered[condition]
        
        elif method == 'quantiles':
            for col in COLUMNS_FOR_OUTLIERS:
                if col in df_filtered.columns:
                    lower = df_filtered[col].quantile(0.01)
                    upper = df_filtered[col].quantile(0.99)
                    df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]
        
        return df_filtered
    
    def split_data(self, df: pd.DataFrame, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide los datos en conjuntos de entrenamiento y prueba."""
        train_df = df[df['fecha'] < split_date]
        test_df = df[df['fecha'] >= split_date]
        return train_df, test_df
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Escala las características usando StandardScaler."""
        scaler_dict = {}
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        for feature in features:
            if feature in X_train.columns:
                scaler = StandardScaler()
                X_train_scaled[feature] = scaler.fit_transform(X_train[[feature]])
                X_test_scaled[feature] = scaler.transform(X_test[[feature]])
                scaler_dict[feature] = scaler
        
        return X_train_scaled, X_test_scaled, scaler_dict
    
    def scale_target(self, y_train: pd.Series) -> Tuple[np.ndarray, StandardScaler]:
        """Escala la variable objetivo."""
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        return y_scaled, scaler
    
    def train_gam_model(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: List[str]) -> LinearGAM:
        """Entrena el modelo GAM."""
        if len(feature_names) == 0:
            raise ValueError("No se han proporcionado características para el modelo")
        
        # Crear términos spline para cada característica
        terms = []
        for i in range(len(feature_names)):
            if any(temp in feature_names[i] for temp in ['sin', 'cos']):
                # Variables cíclicas: mínimo 4 splines (> spline_order=3)
                terms.append(s(i, n_splines=5, spline_order=2))
            else:
                # Otras variables: más splines para capturar no-linealidades
                terms.append(s(i, n_splines=8, spline_order=3))
        
        # Verificar que tenemos términos válidos
        if not terms:
            raise ValueError("No se pudieron crear términos GAM válidos")
        
        # Crear modelo GAM con regularización
        #try:
        # Construir la fórmula GAM correctamente
        if len(terms) == 1:
            gam_formula = terms[0]
        else:
            gam_formula = terms[0]
            for term in terms[1:]:
                gam_formula = gam_formula + term
        
        gam = LinearGAM(gam_formula, max_iter=200)
        
        # Entrenar modelo con manejo de errores
        gam.fit(X_train.values, y_train)
        gam.feature_names = feature_names
        
        return gam
        
        # except Exception as e:
        #     st.error(f"Error durante el entrenamiento GAM: {str(e)}")
        #     # Fallback: modelo más simple
        #     st.warning("Intentando con configuración más simple...")
            
        #     try:
        #         # Términos más simples
        #         simple_terms = []
        #         for i in range(len(feature_names)):
        #             simple_terms.append(s(i, n_splines=6, spline_order=2))
                
        #         # Construir fórmula simple
        #         if len(simple_terms) == 1:
        #             simple_formula = simple_terms[0]
        #         else:
        #             simple_formula = simple_terms[0]
        #             for term in simple_terms[1:]:
        #                 simple_formula = simple_formula + term
                
        #         gam_simple = LinearGAM(simple_formula, max_iter=100)
        #         gam_simple.fit(X_train.values, y_train)
        #         gam_simple.feature_names = feature_names
                
        #     except Exception as e2:
        #         st.error(f"Error en modelo simple: {str(e2)}")
        #         # Último recurso: modelo lineal básico
        #         st.warning("Usando modelo lineal básico como último recurso...")
                
        #         linear_terms = []
        #         for i in range(len(feature_names)):
        #             linear_terms.append(s(i, n_splines=4, spline_order=1))
                
        #         if len(linear_terms) == 1:
        #             linear_formula = linear_terms[0]
        #         else:
        #             linear_formula = linear_terms[0]
        #             for term in linear_terms[1:]:
        #                 linear_formula = linear_formula + term
                
        #         gam_linear = LinearGAM(linear_formula, max_iter=50)
        #         gam_linear.fit(X_train.values, y_train)
        #         gam_linear.feature_names = feature_names
                
                #return gam_linear
    
    def evaluate_model(self, model: LinearGAM, X_test: pd.DataFrame, y_test: pd.Series, scaler_target: StandardScaler) -> Dict:
        """Evalúa el modelo y calcula métricas."""
        y_pred_scaled = model.predict(X_test.values)
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'y_pred': y_pred,
            'residuals': y_test - y_pred
        }
        
        return metrics
    
    def save_model(self, model: LinearGAM, feature_names: List[str], scaler_dict: Dict, 
                   scaler_target: StandardScaler, sensor_id: str, outlier_method: str, 
                   preprocessing: str) -> str:
        """Guarda el modelo entrenado."""
        model_info = {
            'model': model,
            'feature_names': feature_names,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'variable_metadata': VARIABLE_METADATA
        }
        
        # Crear directorio si no existe
        os.makedirs('data/models', exist_ok=True)
        
        filename = f'data/models/gam_model_{sensor_id}_{outlier_method}_{preprocessing}.pkl'
        joblib.dump(model_info, filename)
        
        return filename
    
    def load_model(self, filepath: str) -> Optional[Dict]:
        """Carga un modelo guardado."""
        try:
            return joblib.load(filepath)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return None
    
    def get_partial_effects(self, model: LinearGAM, feature_names: List[str], 
                           values: np.ndarray, feature_name: str = None, 
                           sin_name: str = None, cos_name: str = None) -> List[float]:
        """Calcula efectos parciales para visualización."""
        effects = []
        
        try:
            for val in values:
                XX = np.zeros((1, len(feature_names)))
                
                if sin_name and cos_name:
                    # Variables cíclicas - usar el rango correcto para cada tipo
                    if 'hour' in sin_name:
                        period = 24
                    elif 'month' in sin_name:
                        period = 12
                    elif 'day_of_week' in sin_name:
                        period = 7
                    elif 'day_of_year' in sin_name:
                        period = 365.25
                    elif 'season' in sin_name:
                        period = 4
                    elif 'day' in sin_name and 'day_of' not in sin_name:
                        period = 31
                    else:
                        period = len(values)
                    
                    sin_val = np.sin(2 * np.pi * val / period)
                    cos_val = np.cos(2 * np.pi * val / period)
                    
                    if sin_name in feature_names and cos_name in feature_names:
                        XX[0, feature_names.index(sin_name)] = sin_val
                        XX[0, feature_names.index(cos_name)] = cos_val
                        
                        # Calcular efecto combinado
                        try:
                            sin_eff = model.partial_dependence(term=feature_names.index(sin_name), X=XX)
                            cos_eff = model.partial_dependence(term=feature_names.index(cos_name), X=XX)
                            effects.append(float(sin_eff + cos_eff))
                        except:
                            # Fallback: usar predicción completa
                            pred = model.predict(XX)
                            effects.append(float(pred[0]))
                    else:
                        effects.append(0.0)
                
                elif feature_name and feature_name in feature_names:
                    # Variable simple
                    XX[0, feature_names.index(feature_name)] = val
                    try:
                        eff = model.partial_dependence(term=feature_names.index(feature_name), X=XX)
                        effects.append(float(eff))
                    except:
                        # Fallback: usar predicción completa
                        pred = model.predict(XX)
                        effects.append(float(pred[0]))
                else:
                    effects.append(0.0)
                    
        except Exception as e:
            st.warning(f"Error calculando efectos parciales: {str(e)}")
            # Retornar efectos neutros
            effects = [0.0] * len(values)
        
        return effects
    
    def validate_gam_config(self, feature_names: List[str], X_train: pd.DataFrame) -> bool:
        """Valida la configuración del modelo GAM antes del entrenamiento."""
        try:
            st.write("🔍 **Validando configuración GAM:**")
            
            # Verificar que la lista de features no esté vacía
            if not feature_names:
                st.error("❌ Lista de features vacía")
                return False
            
            # Verificar que tenemos suficientes datos
            min_samples = max(20, len(feature_names) * 5)
            if len(X_train) < min_samples:
                st.error(f"❌ Datos insuficientes: {len(X_train)} < {min_samples} mínimo requerido")
                return False
            
            # Verificar que todas las features existen en X_train
            missing_in_data = [f for f in feature_names if f not in X_train.columns]
            if missing_in_data:
                st.error(f"❌ Features no encontradas en datos: {missing_in_data}")
                st.write("Columnas disponibles en X_train:")
                st.write(list(X_train.columns))
                return False
            
            # Verificar características de las variables
            cyclical_vars = [f for f in feature_names if any(temp in f for temp in ['sin', 'cos'])]
            regular_vars = [f for f in feature_names if f not in cyclical_vars]
            
            st.write(f"- ✅ Variables cíclicas: {len(cyclical_vars)} {cyclical_vars[:3]}{'...' if len(cyclical_vars) > 3 else ''}")
            st.write(f"- ✅ Variables regulares: {len(regular_vars)} {regular_vars[:3]}{'...' if len(regular_vars) > 3 else ''}")
            st.write(f"- ✅ Total features: {len(feature_names)}")
            st.write(f"- ✅ Muestras entrenamiento: {len(X_train)}")
            
            # Verificar varianza de las variables
            low_variance_vars = []
            infinite_vars = []
            for var in feature_names:
                if var in X_train.columns:
                    var_data = X_train[var]
                    
                    # Verificar infinitos
                    if np.isinf(var_data).any():
                        infinite_vars.append(var)
                        continue
                    
                    # Verificar varianza
                    var_val = var_data.var()
                    if var_val < 1e-8:
                        low_variance_vars.append((var, var_val))
            
            if infinite_vars:
                st.error(f"❌ Variables con valores infinitos: {infinite_vars}")
                return False
            
            if low_variance_vars:
                st.warning(f"⚠️ Variables con varianza muy baja: {[f'{var}: {val:.2e}' for var, val in low_variance_vars]}")
            
            # Verificar valores NaN
            nan_vars = []
            for var in feature_names:
                if var in X_train.columns:
                    nan_count = X_train[var].isna().sum()
                    if nan_count > 0:
                        nan_vars.append((var, nan_count))
            
            if nan_vars:
                st.error(f"❌ Variables con valores NaN: {[(var, count) for var, count in nan_vars]}")
                return False
            
            # Mostrar estadísticas de las primeras features
            st.write("📊 **Estadísticas de muestra:**")
            sample_features = feature_names[:5]  # Mostrar solo las primeras 5
            for var in sample_features:
                if var in X_train.columns:
                    data = X_train[var]
                    st.write(f"  - {var}: min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}, std={data.std():.3f}")
            
            st.success("✅ Configuración GAM válida")
            return True
            
        except Exception as e:
            st.error(f"❌ Error validando configuración: {str(e)}")
            st.exception(e)
            return False


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_model_metrics(metrics: Dict):
    """Muestra las métricas del modelo."""
    st.subheader("📊 Métricas del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "RMSE", 
            f"{metrics['rmse']:.2f} μg/m³",
            help="Error cuadrático medio - menor es mejor"
        )
    
    with col2:
        st.metric(
            "R² Score", 
            f"{metrics['r2']:.3f}",
            help="Coeficiente de determinación - más cercano a 1 es mejor"
        )
    
    with col3:
        st.metric(
            "MAE", 
            f"{metrics['mae']:.2f} μg/m³",
            help="Error absoluto medio - menor es mejor"
        )


def show_residual_analysis(residuals: np.ndarray):
    """Muestra análisis de residuos."""
    st.subheader("📈 Análisis de Residuos")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma de residuos
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title('Distribución de Residuos')
    axes[0].set_xlabel('Residuo (μg/m³)')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Gráfico Q-Q
    sm.qqplot(residuals, line='45', ax=axes[1], fit=True)
    axes[1].set_title('Gráfico Q-Q de Residuos')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_temporal_effects(trainer: GAMTrainer, model: LinearGAM, feature_names: List[str]):
    """Muestra efectos temporales del modelo."""
    st.subheader("🕐 Efectos Temporales")
    
    # Primera fila: Hora, Día de la semana, Mes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Efecto de la Hora**")
        hours = np.arange(0, 24)
        if 'hour_sin' in feature_names and 'hour_cos' in feature_names:
            hour_effects = trainer.get_partial_effects(
                model, feature_names, hours, 
                sin_name='hour_sin', cos_name='hour_cos'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(hours, hour_effects, marker='o', linewidth=2, markersize=4)
            ax.set_title('Efecto de la Hora en NO₂')
            ax.set_xlabel('Hora del día')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)
            plt.close(fig)
    
    with col2:
        st.write("**Efecto del Día de la Semana**")
        days = np.arange(0, 7)
        if 'day_of_week_sin' in feature_names and 'day_of_week_cos' in feature_names:
            day_effects = trainer.get_partial_effects(
                model, feature_names, days,
                sin_name='day_of_week_sin', cos_name='day_of_week_cos'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(days, day_effects, marker='o', linewidth=2, markersize=6)
            ax.set_title('Efecto del Día de la Semana')
            ax.set_xlabel('Día de la semana')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3)
            
            day_labels = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
            ax.set_xticks(range(7))
            ax.set_xticklabels(day_labels)
            
            # Resaltar fin de semana
            weekend_color = 'lightcoral' if len(day_effects) > 5 else 'lightblue'
            ax.axvspan(4.5, 6.5, alpha=0.2, color=weekend_color, label='Fin de semana')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    
    with col3:
        st.write("**Efecto del Mes**")
        months = np.arange(1, 13)
        if 'month_sin' in feature_names and 'month_cos' in feature_names:
            month_effects = trainer.get_partial_effects(
                model, feature_names, months,
                sin_name='month_sin', cos_name='month_cos'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(months, month_effects, marker='o', linewidth=2, markersize=4)
            ax.set_title('Efecto del Mes')
            ax.set_xlabel('Mes')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 13))
            
            # Etiquetas de meses
            month_labels = ['E', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
            ax.set_xticklabels(month_labels)
            st.pyplot(fig)
            plt.close(fig)
    
    # Segunda fila: Día del año, Estaciones, Efecto fin de semana
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.write("**Efecto del Día del Año**")
        if 'day_of_year_sin' in feature_names and 'day_of_year_cos' in feature_names:
            day_year_range = np.arange(1, 366, 10)  # Cada 10 días para mejor visualización
            day_year_effects = trainer.get_partial_effects(
                model, feature_names, day_year_range,
                sin_name='day_of_year_sin', cos_name='day_of_year_cos'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(day_year_range, day_year_effects, linewidth=2)
            ax.set_title('Efecto del Día del Año')
            ax.set_xlabel('Día del año')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3)
            
            # Marcar estaciones
            season_starts = [1, 80, 172, 266]  # Aproximado: invierno, primavera, verano, otoño
            season_names = ['Invierno', 'Primavera', 'Verano', 'Otoño']
            colors = ['lightblue', 'lightgreen', 'yellow', 'orange']
            
            for i, (start, name, color) in enumerate(zip(season_starts, season_names, colors)):
                end = season_starts[i+1] if i < 3 else 365
                ax.axvspan(start, end, alpha=0.1, color=color, label=name)
            
            ax.legend(loc='upper right', fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
    
    with col5:
        st.write("**Efecto de las Estaciones**")
        if 'season_sin' in feature_names and 'season_cos' in feature_names:
            seasons = np.arange(0, 4)
            season_effects = trainer.get_partial_effects(
                model, feature_names, seasons,
                sin_name='season_sin', cos_name='season_cos'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(seasons, season_effects, 
                         color=['lightblue', 'lightgreen', 'yellow', 'orange'],
                         alpha=0.7)
            ax.set_title('Efecto de las Estaciones')
            ax.set_xlabel('Estación')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3, axis='y')
            
            season_labels = ['Invierno', 'Primavera', 'Verano', 'Otoño']
            ax.set_xticks(range(4))
            ax.set_xticklabels(season_labels, rotation=45)
            
            # Añadir valores en las barras
            for bar, effect in zip(bars, season_effects):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{effect:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close(fig)
    
    with col6:
        st.write("**Efecto Fin de Semana**")
        if 'weekend' in feature_names:
            weekend_values = [0, 1]  # Laborable, Fin de semana
            weekend_effects = trainer.get_partial_effects(
                model, feature_names, weekend_values, feature_name='weekend'
            )
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(weekend_values, weekend_effects, 
                         color=['lightblue', 'lightcoral'],
                         alpha=0.7)
            ax.set_title('Efecto Fin de Semana')
            ax.set_xlabel('Tipo de día')
            ax.set_ylabel('Efecto en NO₂')
            ax.grid(True, alpha=0.3, axis='y')
            
            weekend_labels = ['Laborable', 'Fin de semana']
            ax.set_xticks(range(2))
            ax.set_xticklabels(weekend_labels)
            
            # Añadir valores en las barras
            for bar, effect in zip(bars, weekend_effects):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{effect:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.close(fig)


def show_variable_effects(trainer: GAMTrainer, model: LinearGAM, feature_names: List[str]):
    """Muestra efectos de variables específicas."""
    st.subheader("🔬 Efectos de Variables por Categoría")
    
    # Crear tabs para diferentes categorías de variables
    tab1, tab2, tab3 = st.tabs(["🚗 Variables de Tráfico", "🌤️ Variables Meteorológicas", "📊 Comparación General"])
    
    with tab1:
        st.write("### Efectos de Variables de Tráfico")
        
        traffic_variables = {
            'intensidad': {'range': (0, 1500), 'title': 'Intensidad de Tráfico', 'unit': 'veh/h', 'color': 'red'},
            'carga': {'range': (0, 100), 'title': 'Carga de Tráfico', 'unit': '%', 'color': 'orange'},
            'ocupacion': {'range': (0, 100), 'title': 'Ocupación Vial', 'unit': '%', 'color': 'purple'},
            'vmed': {'range': (0, 100), 'title': 'Velocidad Media', 'unit': 'km/h', 'color': 'blue'}
        }
        
        # Mostrar efectos de tráfico en una cuadrícula 2x2
        cols = st.columns(2)
        for i, (var, config) in enumerate(traffic_variables.items()):
            if var in feature_names:
                with cols[i % 2]:
                    min_val, max_val = config['range']
                    values = np.linspace(min_val, max_val, 50)
                    effects = trainer.get_partial_effects(
                        model, feature_names, values, feature_name=var
                    )
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(values, effects, color=config['color'], linewidth=2.5)
                    ax.fill_between(values, effects, alpha=0.3, color=config['color'])
                    ax.set_title(f"Efecto de {config['title']} en NO₂", fontsize=12, fontweight='bold')
                    ax.set_xlabel(f"{config['title']} ({config['unit']})")
                    ax.set_ylabel('Efecto en NO₂ (μg/m³)')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    # Añadir anotaciones sobre rangos típicos
                    if var == 'intensidad':
                        ax.axvspan(0, 300, alpha=0.1, color='green', label='Tráfico bajo')
                        ax.axvspan(300, 800, alpha=0.1, color='yellow', label='Tráfico medio')
                        ax.axvspan(800, 1500, alpha=0.1, color='red', label='Tráfico alto')
                        ax.legend(loc='upper right', fontsize=8)
                    elif var in ['carga', 'ocupacion']:
                        ax.axvspan(0, 30, alpha=0.1, color='green', label='Bajo')
                        ax.axvspan(30, 70, alpha=0.1, color='yellow', label='Medio')
                        ax.axvspan(70, 100, alpha=0.1, color='red', label='Alto')
                        ax.legend(loc='upper right', fontsize=8)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Mostrar estadísticas del efecto
                    max_effect = np.max(effects)
                    min_effect = np.min(effects)
                    effect_range = max_effect - min_effect
                    st.write(f"📊 **Rango de efecto**: {effect_range:.2f} μg/m³")
                    st.write(f"🔺 **Máximo efecto**: {max_effect:.2f} μg/m³")
                    st.write(f"🔻 **Mínimo efecto**: {min_effect:.2f} μg/m³")
    
    with tab2:
        st.write("### Efectos de Variables Meteorológicas")
        
        meteo_variables = {
            't2m': {'range': (-5, 40), 'title': 'Temperatura', 'unit': '°C', 'color': 'red'},
            'd2m': {'range': (-10, 30), 'title': 'Punto de Rocío', 'unit': '°C', 'color': 'blue'},
            'sp': {'range': (980, 1030), 'title': 'Presión Superficial', 'unit': 'hPa', 'color': 'green'},
            'tp': {'range': (0, 20), 'title': 'Precipitación', 'unit': 'mm', 'color': 'cyan'},
            'u10': {'range': (-20, 20), 'title': 'Viento U (10m)', 'unit': 'km/h', 'color': 'orange'},
            'v10': {'range': (-20, 20), 'title': 'Viento V (10m)', 'unit': 'km/h', 'color': 'purple'},
            'ssrd': {'range': (0, 800), 'title': 'Radiación Solar', 'unit': 'W/m²', 'color': 'yellow'},
            'ssr': {'range': (0, 600), 'title': 'Radiación Neta', 'unit': 'W/m²', 'color': 'gold'}
        }
        
        # Mostrar efectos meteorológicos en cuadrícula 2x4
        cols = st.columns(2)
        for i, (var, config) in enumerate(meteo_variables.items()):
            if var in feature_names:
                with cols[i % 2]:
                    min_val, max_val = config['range']
                    values = np.linspace(min_val, max_val, 50)
                    effects = trainer.get_partial_effects(
                        model, feature_names, values, feature_name=var
                    )
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(values, effects, color=config['color'], linewidth=2.5)
                    ax.fill_between(values, effects, alpha=0.3, color=config['color'])
                    ax.set_title(f"Efecto de {config['title']} en NO₂", fontsize=12, fontweight='bold')
                    ax.set_xlabel(f"{config['title']} ({config['unit']})")
                    ax.set_ylabel('Efecto en NO₂ (μg/m³)')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    # Añadir referencias específicas
                    if var == 't2m':
                        ax.axvline(x=0, color='blue', linestyle=':', alpha=0.7, label='Congelación')
                        ax.axvline(x=25, color='red', linestyle=':', alpha=0.7, label='Calor')
                        ax.legend(fontsize=8)
                    elif var == 'sp':
                        ax.axvline(x=1013.25, color='black', linestyle=':', alpha=0.7, label='Presión estándar')
                        ax.legend(fontsize=8)
                    elif var == 'tp':
                        ax.axvline(x=0.1, color='blue', linestyle=':', alpha=0.7, label='Lluvia ligera')
                        ax.axvline(x=2.5, color='red', linestyle=':', alpha=0.7, label='Lluvia fuerte')
                        ax.legend(fontsize=8)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Mostrar estadísticas del efecto
                    max_effect = np.max(effects)
                    min_effect = np.min(effects)
                    effect_range = max_effect - min_effect
                    st.write(f"📊 **Rango de efecto**: {effect_range:.2f} μg/m³")
                    
                    # Interpretación específica por variable
                    if var == 't2m':
                        if max_effect > min_effect:
                            st.write("🌡️ **Interpretación**: Mayor temperatura → Mayor NO₂")
                        else:
                            st.write("🌡️ **Interpretación**: Mayor temperatura → Menor NO₂")
                    elif var == 'tp':
                        if effects[0] > effects[-1]:
                            st.write("🌧️ **Interpretación**: Lluvia reduce NO₂ (lavado atmosférico)")
                        else:
                            st.write("🌧️ **Interpretación**: Relación compleja con precipitación")
    
    with tab3:
        st.write("### Comparación de Importancia de Variables")
        
        # Calcular importancia relativa basada en el rango de efectos
        variable_importance = {}
        
        all_variables = {}
        all_variables.update({k: v for k, v in traffic_variables.items() if k in feature_names})
        all_variables.update({k: v for k, v in meteo_variables.items() if k in feature_names})
        
        for var, config in all_variables.items():
            if var in feature_names:
                min_val, max_val = config['range']
                values = np.linspace(min_val, max_val, 30)
                effects = trainer.get_partial_effects(
                    model, feature_names, values, feature_name=var
                )
                effect_range = np.max(effects) - np.min(effects)
                variable_importance[var] = {
                    'range': effect_range,
                    'title': config['title'],
                    'category': 'Tráfico' if var in traffic_variables else 'Meteorología'
                }
        
        if variable_importance:
            # Ordenar por importancia
            sorted_vars = sorted(variable_importance.items(), key=lambda x: x[1]['range'], reverse=True)
            
            # Gráfico de barras de importancia
            fig, ax = plt.subplots(figsize=(12, 8))
            
            vars_names = [item[1]['title'] for item in sorted_vars]
            ranges = [item[1]['range'] for item in sorted_vars]
            categories = [item[1]['category'] for item in sorted_vars]
            
            colors = ['red' if cat == 'Tráfico' else 'blue' for cat in categories]
            
            bars = ax.barh(vars_names, ranges, color=colors, alpha=0.7)
            ax.set_xlabel('Rango de Efecto en NO₂ (μg/m³)')
            ax.set_title('Importancia Relativa de Variables (basada en rango de efectos)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Añadir valores en las barras
            for bar, range_val in zip(bars, ranges):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{range_val:.2f}', ha='left', va='center', fontweight='bold')
            
            # Leyenda
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Variables de Tráfico'),
                             Patch(facecolor='blue', alpha=0.7, label='Variables Meteorológicas')]
            ax.legend(handles=legend_elements, loc='lower right')
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Tabla resumen
            st.write("### 📋 Resumen de Importancia")
            import pandas as pd
            summary_df = pd.DataFrame([
                {
                    'Variable': item[1]['title'],
                    'Categoría': item[1]['category'],
                    'Rango de Efecto (μg/m³)': f"{item[1]['range']:.2f}",
                    'Importancia': '🔴 Alta' if item[1]['range'] > np.mean(ranges) else '🟡 Media' if item[1]['range'] > np.mean(ranges)/2 else '🟢 Baja'
                }
                for item in sorted_vars
            ])
            st.dataframe(summary_df, use_container_width=True)
            
            # Insights automáticos
            st.write("### 💡 Insights Automáticos")
            
            top_traffic = [item for item in sorted_vars if item[1]['category'] == 'Tráfico'][:2]
            top_meteo = [item for item in sorted_vars if item[1]['category'] == 'Meteorología'][:2]
            
            if top_traffic:
                st.write(f"🚗 **Variables de tráfico más influyentes**: {', '.join([item[1]['title'] for item in top_traffic])}")
            
            if top_meteo:
                st.write(f"🌤️ **Variables meteorológicas más influyentes**: {', '.join([item[1]['title'] for item in top_meteo])}")
            
            avg_traffic_effect = np.mean([item[1]['range'] for item in sorted_vars if item[1]['category'] == 'Tráfico'])
            avg_meteo_effect = np.mean([item[1]['range'] for item in sorted_vars if item[1]['category'] == 'Meteorología'])
            
            if avg_traffic_effect > avg_meteo_effect:
                st.write("📊 **Conclusión**: Las variables de tráfico tienen mayor impacto promedio en NO₂")
            else:
                st.write("📊 **Conclusión**: Las variables meteorológicas tienen mayor impacto promedio en NO₂")


def show_partial_dependence_plots(model: LinearGAM, feature_names: List[str]):
    """Muestra gráficos de dependencia parcial."""
    st.subheader("📊 Dependencias Parciales Detalladas")
    
    cols = 3
    rows = (len(feature_names) + cols - 1) // cols
    
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(feature_names):
                feature = feature_names[idx]
                with columns[col]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    try:
                        XX = model.generate_X_grid(term=idx)
                        pdep, confi = model.partial_dependence(term=idx, X=XX, width=0.95)
                        
                        ax.plot(XX[:, idx], pdep, label='Efecto parcial')
                        ax.fill_between(XX[:, idx], confi[:, 0], confi[:, 1], 
                                       alpha=0.2, label='IC 95%')
                        ax.legend()
                    except:
                        XX = model.generate_X_grid(term=idx)
                        pdep = model.partial_dependence(term=idx, X=XX)
                        ax.plot(XX[:, idx], pdep)
                    
                    ax.set_title(f'{feature}')
                    ax.set_ylabel('Efecto parcial')
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)


def show_info_panel():
    """Muestra panel de información sobre el entrenamiento GAM."""
    with st.expander("ℹ️ Acerca del Entrenamiento GAM", expanded=False):
        st.markdown("""
        **Modelos GAM (Generalized Additive Models)**
        
        Los GAM permiten modelar relaciones no lineales entre variables predictoras y NO₂.
        
        **Características:**
        - **Flexibilidad**: Captura relaciones no lineales automáticamente
        - **Interpretabilidad**: Cada variable contribuye de forma aditiva
        - **Robustez**: Maneja diferentes tipos de variables (continuas, cíclicas)
        
        **Proceso de entrenamiento:**
        1. **Preprocesamiento**: Creación de variables cíclicas y conversión de unidades
        2. **División temporal**: Separación por fechas (entrenamiento/evaluación)
        3. **Filtrado de outliers**: ⚠️ **Solo en datos de entrenamiento** (evita data leakage)
        4. **Escalado**: Normalización de variables para mejor convergencia
        5. **Entrenamiento**: Optimización de splines para cada variable
        
        **Buenas prácticas implementadas:**
        - ✅ **Outliers eliminados solo del training set** - Mantiene integridad de evaluación
        - ✅ **División temporal** - Simula predicción en tiempo real
        - ✅ **Variables cíclicas** - Captura patrones temporales complejos
        - ✅ **Escalado por conjunto** - Evita data leakage entre train/test
        
        **Variables utilizadas:**
        - **Temporales**: Hora, día, mes, estaciones (como funciones cíclicas)
        - **Tráfico**: Intensidad, carga, ocupación, velocidad
        - **Meteorológicas**: Temperatura, precipitación, presión, viento
        """)
        
        st.markdown("---")
        st.markdown("""
        **🔍 ¿Por qué outliers solo en entrenamiento?**
        
        - **Data leakage**: Usar información del test set contaminaría la evaluación
        - **Realismo**: En producción no conocemos los outliers futuros
        - **Robustez**: El modelo debe manejar datos anómalos en predicción
        - **Evaluación honesta**: Métricas reflejan rendimiento real
        """)


def show_interaction_analysis(trainer: GAMTrainer, model: LinearGAM, feature_names: List[str], X_test: pd.DataFrame):
    """Muestra análisis de interacciones entre variables."""
    st.subheader("🔄 Análisis de Interacciones entre Variables")
    
    # Seleccionar pares de variables importantes para análisis de interacción
    key_interactions = [
        ('intensidad', 't2m', 'Tráfico vs Temperatura'),
        ('intensidad', 'tp', 'Tráfico vs Precipitación'),
        ('t2m', 'sp', 'Temperatura vs Presión'),
        ('carga', 'u10', 'Carga Tráfico vs Viento U'),
        ('ocupacion', 'v10', 'Ocupación vs Viento V'),
        ('vmed', 'ssrd', 'Velocidad vs Radiación Solar')
    ]
    
    available_interactions = [
        (var1, var2, name) for var1, var2, name in key_interactions 
        if var1 in feature_names and var2 in feature_names
    ]
    
    if not available_interactions:
        st.warning("No hay suficientes variables disponibles para análisis de interacciones.")
        return
    
    # Crear tabs para diferentes tipos de interacciones
    tab1, tab2, tab3 = st.tabs([
        "🌡️ Tráfico-Meteorología", 
        "📊 Matriz de Correlaciones GAM", 
        "🎯 Escenarios Combinados"
    ])
    
    with tab1:
        st.write("### Interacciones Tráfico-Meteorología")
        
        # Mostrar heatmaps de interacciones
        for var1, var2, interaction_name in available_interactions[:4]:  # Mostrar máximo 4
            st.write(f"#### {interaction_name}")
            
            # Crear grilla de valores para ambas variables
            if var1 in VARIABLE_METADATA:
                range1 = VARIABLE_METADATA[var1]['typical_range']
            else:
                range1 = (X_test[var1].min(), X_test[var1].max())
            
            if var2 in VARIABLE_METADATA:
                range2 = VARIABLE_METADATA[var2]['typical_range']
            else:
                range2 = (X_test[var2].min(), X_test[var2].max())
            
            # Crear grilla más pequeña para eficiencia
            var1_vals = np.linspace(range1[0], range1[1], 20)
            var2_vals = np.linspace(range2[0], range2[1], 20)
            
            # Calcular efectos combinados
            interaction_effects = np.zeros((len(var2_vals), len(var1_vals)))
            
            for i, v2 in enumerate(var2_vals):
                for j, v1 in enumerate(var1_vals):
                    # Crear punto de datos con valores medios para otras variables
                    XX = np.zeros((1, len(feature_names)))
                    
                    # Establecer valores medios para todas las variables
                    for k, feat in enumerate(feature_names):
                        if feat == var1:
                            XX[0, k] = v1
                        elif feat == var2:
                            XX[0, k] = v2
                        elif feat in X_test.columns:
                            XX[0, k] = X_test[feat].mean()
                    
                    try:
                        pred = model.predict(XX)
                        interaction_effects[i, j] = pred[0]
                    except:
                        interaction_effects[i, j] = 0
            
            # Crear heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(interaction_effects, cmap='RdYlBu_r', aspect='auto', 
                          extent=[range1[0], range1[1], range2[0], range2[1]])
            
            ax.set_xlabel(f"{VARIABLE_METADATA.get(var1, {}).get('name', var1)} ({VARIABLE_METADATA.get(var1, {}).get('unit', '')})")
            ax.set_ylabel(f"{VARIABLE_METADATA.get(var2, {}).get('name', var2)} ({VARIABLE_METADATA.get(var2, {}).get('unit', '')})")
            ax.set_title(f"Efecto Combinado: {interaction_name}")
            
            # Añadir colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Predicción NO₂ (μg/m³)')
            
            # Añadir contornos
            contours = ax.contour(var1_vals, var2_vals, interaction_effects, 
                                colors='black', alpha=0.3, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Interpretación automática
            max_effect = np.max(interaction_effects)
            min_effect = np.min(interaction_effects)
            effect_range = max_effect - min_effect
            
            st.write(f"📊 **Rango de efectos combinados**: {effect_range:.2f} μg/m³")
            
            # Encontrar condiciones de máximo y mínimo
            max_idx = np.unravel_index(np.argmax(interaction_effects), interaction_effects.shape)
            min_idx = np.unravel_index(np.argmin(interaction_effects), interaction_effects.shape)
            
            max_v1 = var1_vals[max_idx[1]]
            max_v2 = var2_vals[max_idx[0]]
            min_v1 = var1_vals[min_idx[1]]
            min_v2 = var2_vals[min_idx[0]]
            
            st.write(f"🔺 **Máximo NO₂** ({max_effect:.2f} μg/m³): {var1}={max_v1:.1f}, {var2}={max_v2:.1f}")
            st.write(f"🔻 **Mínimo NO₂** ({min_effect:.2f} μg/m³): {var1}={min_v1:.1f}, {var2}={min_v2:.1f}")
    
    with tab2:
        st.write("### Matriz de Correlaciones de Efectos GAM")
        
        # Variables a analizar (evitar variables cíclicas y duplicadas)
        analysis_vars = []
        for var in feature_names:
            if (var in VARIABLE_METADATA and 
                not any(temp in var for temp in ['sin', 'cos']) and
                var not in analysis_vars):
                analysis_vars.append(var)
        
        # Limitar a máximo 8 variables para eficiencia
        analysis_vars = analysis_vars[:8]
        
        if len(analysis_vars) >= 2:
            st.write(f"🔍 **Analizando correlaciones entre**: {', '.join([VARIABLE_METADATA.get(var, {}).get('name', var) for var in analysis_vars])}")
            
            correlation_matrix = np.zeros((len(analysis_vars), len(analysis_vars)))
            
            # Calcular efectos parciales para cada variable de manera más robusta
            all_effects = {}
            for var in analysis_vars:
                if var in VARIABLE_METADATA:
                    range_var = VARIABLE_METADATA[var]['typical_range']
                    # Usar más puntos para mejor resolución
                    values = np.linspace(range_var[0], range_var[1], 50)
                    
                    # Calcular efectos parciales usando el método más robusto
                    try:
                        # Crear matriz base con valores medios
                        n_points = len(values)
                        XX_base = np.zeros((n_points, len(feature_names)))
                        
                        # Llenar con valores medios de X_test para todas las variables
                        for feat_idx, feat_name in enumerate(feature_names):
                            if feat_name in X_test.columns:
                                XX_base[:, feat_idx] = X_test[feat_name].mean()
                            elif any(temp in feat_name for temp in ['sin', 'cos']):
                                # Para variables cíclicas, usar 0 (valor neutro)
                                XX_base[:, feat_idx] = 0
                        
                        # Variar solo la variable de interés
                        var_idx = feature_names.index(var)
                        XX_base[:, var_idx] = values
                        
                        # Obtener predicciones
                        predictions = model.predict(XX_base)
                        
                        # Calcular efectos parciales como diferencia respecto al valor medio
                        mean_prediction = np.mean(predictions)
                        partial_effects = predictions - mean_prediction
                        
                        all_effects[var] = partial_effects
                        
                    except Exception as e:
                        st.warning(f"Error calculando efectos para {var}: {str(e)}")
                        # Usar efectos neutros en caso de error
                        all_effects[var] = np.zeros(len(values))
            
            # Calcular correlaciones entre efectos parciales
            valid_vars = [var for var in analysis_vars if var in all_effects and len(all_effects[var]) > 0]
            
            if len(valid_vars) >= 2:
                correlation_matrix = np.zeros((len(valid_vars), len(valid_vars)))
                
                for i, var1 in enumerate(valid_vars):
                    for j, var2 in enumerate(valid_vars):
                        if i == j:
                            correlation_matrix[i, j] = 1.0
                        else:
                            try:
                                # Verificar que los efectos no sean constantes
                                effects1 = all_effects[var1]
                                effects2 = all_effects[var2]
                                
                                if np.std(effects1) > 1e-10 and np.std(effects2) > 1e-10:
                                    corr = np.corrcoef(effects1, effects2)[0, 1]
                                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
                                else:
                                    correlation_matrix[i, j] = 0
                            except:
                                correlation_matrix[i, j] = 0
                
                # Crear heatmap de correlaciones
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                
                # Configurar etiquetas
                var_labels = [VARIABLE_METADATA.get(var, {}).get('name', var) for var in valid_vars]
                ax.set_xticks(range(len(valid_vars)))
                ax.set_yticks(range(len(valid_vars)))
                ax.set_xticklabels(var_labels, rotation=45, ha='right')
                ax.set_yticklabels(var_labels)
                
                # Añadir valores en las celdas
                for i in range(len(valid_vars)):
                    for j in range(len(valid_vars)):
                        color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
                        text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                     ha="center", va="center", color=color, fontweight='bold')
                
                ax.set_title("Correlaciones entre Efectos Parciales GAM")
                plt.colorbar(im, ax=ax, label='Correlación')
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Interpretación de correlaciones con filtros más estrictos
                st.write("### 🔍 Correlaciones Significativas")
                
                # Mostrar estadísticas de los efectos
                st.write("#### 📊 Estadísticas de Efectos Parciales")
                effects_stats = []
                for var in valid_vars:
                    effects = all_effects[var]
                    stats = {
                        'Variable': VARIABLE_METADATA.get(var, {}).get('name', var),
                        'Rango': f"{np.max(effects) - np.min(effects):.3f}",
                        'Std Dev': f"{np.std(effects):.3f}",
                        'Varianza': f"{np.var(effects):.3f}"
                    }
                    effects_stats.append(stats)
                
                import pandas as pd
                stats_df = pd.DataFrame(effects_stats)
                st.dataframe(stats_df, use_container_width=True)
                
                # Filtrar correlaciones significativas (evitar perfectas y diagonales)
                significant_corr_pairs = []
                for i in range(len(valid_vars)):
                    for j in range(i+1, len(valid_vars)):
                        corr_val = correlation_matrix[i, j]
                        # Filtros más estrictos: evitar correlaciones perfectas sospechosas
                        if (0.3 <= abs(corr_val) <= 0.95 and  # Rango razonable
                            np.std(all_effects[valid_vars[i]]) > 1e-6 and  # Efectos no constantes
                            np.std(all_effects[valid_vars[j]]) > 1e-6):
                            significant_corr_pairs.append((valid_vars[i], valid_vars[j], corr_val))
                
                if significant_corr_pairs:
                    st.write("**Correlaciones físicamente plausibles:**")
                    for var1, var2, corr in sorted(significant_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                        var1_name = VARIABLE_METADATA.get(var1, {}).get('name', var1)
                        var2_name = VARIABLE_METADATA.get(var2, {}).get('name', var2)
                        
                        # Interpretación contextual
                        interpretation = ""
                        if var1 in ['t2m', 'd2m'] and var2 in ['t2m', 'd2m']:
                            interpretation = " (relación meteorológica esperada)"
                        elif var1 in ['intensidad', 'carga', 'ocupacion', 'vmed'] and var2 in ['intensidad', 'carga', 'ocupacion', 'vmed']:
                            interpretation = " (variables de tráfico relacionadas)"
                        elif 'ssr' in [var1, var2] and 'ssrd' in [var1, var2]:
                            interpretation = " (radiaciones solares relacionadas)"
                        
                        if corr > 0:
                            st.write(f"📈 **{var1_name}** ↔ **{var2_name}**: efectos similares (r={corr:.2f}){interpretation}")
                        else:
                            st.write(f"📉 **{var1_name}** ↔ **{var2_name}**: efectos opuestos (r={corr:.2f}){interpretation}")
                else:
                    st.write("✅ **No se encontraron correlaciones artificiales**. Los efectos parciales son independientes.")
                
                # Advertencias sobre correlaciones sospechosas
                perfect_corr_pairs = []
                for i in range(len(valid_vars)):
                    for j in range(i+1, len(valid_vars)):
                        corr_val = correlation_matrix[i, j]
                        if abs(corr_val) > 0.98:  # Correlaciones casi perfectas
                            perfect_corr_pairs.append((valid_vars[i], valid_vars[j], corr_val))
                
                if perfect_corr_pairs:
                    st.warning("⚠️ **Correlaciones sospechosamente altas detectadas:**")
                    for var1, var2, corr in perfect_corr_pairs:
                        var1_name = VARIABLE_METADATA.get(var1, {}).get('name', var1)
                        var2_name = VARIABLE_METADATA.get(var2, {}).get('name', var2)
                        st.write(f"- {var1_name} vs {var2_name}: r={corr:.3f}")
                    st.write("Esto puede indicar:")
                    st.write("- Multicolinealidad en los datos")
                    st.write("- Efectos parciales muy similares")
                    st.write("- Posible sobreajuste del modelo")
                    
                    # Botón para diagnóstico detallado
                    #if st.button("🔬 Ejecutar Diagnóstico Detallado"):
                    diagnose_perfect_correlations(model, feature_names, X_test)
            
            else:
                st.warning("No hay suficientes variables válidas para análisis de correlaciones.")
        else:
            st.warning("No hay suficientes variables disponibles para análisis de correlaciones.")
    
    with tab3:
        st.write("### Escenarios de Condiciones Combinadas")
        
        # Definir escenarios típicos
        scenarios = {
            "🌅 Mañana Laborable": {
                "hour_sin": np.sin(2 * np.pi * 8 / 24),  # 8 AM
                "hour_cos": np.cos(2 * np.pi * 8 / 24),
                "weekend": 0,
                "intensidad": 800,  # Tráfico alto
                "t2m": 15,  # Temperatura moderada
                "tp": 0,  # Sin lluvia
            },
            "🌆 Tarde Laborable": {
                "hour_sin": np.sin(2 * np.pi * 18 / 24),  # 6 PM
                "hour_cos": np.cos(2 * np.pi * 18 / 24),
                "weekend": 0,
                "intensidad": 1000,  # Tráfico muy alto
                "t2m": 20,  # Temperatura cálida
                "tp": 0,  # Sin lluvia
            },
            "🌙 Noche Fin de Semana": {
                "hour_sin": np.sin(2 * np.pi * 22 / 24),  # 10 PM
                "hour_cos": np.cos(2 * np.pi * 22 / 24),
                "weekend": 1,
                "intensidad": 200,  # Tráfico bajo
                "t2m": 12,  # Temperatura fresca
                "tp": 0,  # Sin lluvia
            },
            "🌧️ Día Lluvioso": {
                "hour_sin": np.sin(2 * np.pi * 14 / 24),  # 2 PM
                "hour_cos": np.cos(2 * np.pi * 14 / 24),
                "weekend": 0,
                "intensidad": 400,  # Tráfico reducido por lluvia
                "t2m": 10,  # Temperatura baja
                "tp": 5,  # Lluvia moderada
            },
            "🔥 Día Caluroso Verano": {
                "hour_sin": np.sin(2 * np.pi * 15 / 24),  # 3 PM
                "hour_cos": np.cos(2 * np.pi * 15 / 24),
                "weekend": 0,
                "intensidad": 600,  # Tráfico moderado
                "t2m": 35,  # Temperatura muy alta
                "tp": 0,  # Sin lluvia
                "ssrd": 800,  # Radiación solar alta
            }
        }
        
        # Calcular predicciones para cada escenario
        scenario_results = {}
        
        for scenario_name, scenario_values in scenarios.items():
            # Crear vector de características
            XX = np.zeros((1, len(feature_names)))
            
            # Establecer valores del escenario
            for feat_idx, feat_name in enumerate(feature_names):
                if feat_name in scenario_values:
                    XX[0, feat_idx] = scenario_values[feat_name]
                elif feat_name in X_test.columns:
                    XX[0, feat_idx] = X_test[feat_name].mean()  # Valor medio para otras variables
            
            try:
                prediction = model.predict(XX)[0]
                scenario_results[scenario_name] = prediction
            except:
                scenario_results[scenario_name] = None
        
        # Mostrar resultados
        if scenario_results:
            # Gráfico de barras de escenarios
            valid_scenarios = {k: v for k, v in scenario_results.items() if v is not None}
            
            if valid_scenarios:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                scenarios_names = list(valid_scenarios.keys())
                predictions = list(valid_scenarios.values())
                
                colors = ['skyblue', 'orange', 'purple', 'lightblue', 'red'][:len(scenarios_names)]
                bars = ax.bar(scenarios_names, predictions, color=colors, alpha=0.7)
                
                ax.set_ylabel('Predicción NO₂ (μg/m³)')
                ax.set_title('Predicciones de NO₂ para Diferentes Escenarios')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Añadir línea de referencia WHO
                ax.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Límite WHO (40 μg/m³)')
                ax.legend()
                
                # Añadir valores en las barras
                for bar, pred in zip(bars, predictions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{pred:.1f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Análisis de escenarios
                st.write("### 📋 Análisis de Escenarios")
                
                max_scenario = max(valid_scenarios.items(), key=lambda x: x[1])
                min_scenario = min(valid_scenarios.items(), key=lambda x: x[1])
                
                st.write(f"🔺 **Peor escenario**: {max_scenario[0]} → {max_scenario[1]:.1f} μg/m³")
                st.write(f"🔻 **Mejor escenario**: {min_scenario[0]} → {min_scenario[1]:.1f} μg/m³")
                
                # Alertas por límites WHO
                dangerous_scenarios = [name for name, pred in valid_scenarios.items() if pred > 40]
                if dangerous_scenarios:
                    st.warning(f"⚠️ **Escenarios que superan límite WHO**: {', '.join(dangerous_scenarios)}")
                else:
                    st.success("✅ Todos los escenarios están por debajo del límite WHO")
                
                # Recomendaciones
                st.write("### 💡 Recomendaciones Basadas en Escenarios")
                
                if max_scenario[1] > 40:
                    st.write("🚨 **Alerta**: Condiciones de alto riesgo identificadas")
                    st.write("- Considerar restricciones de tráfico en horas pico")
                    st.write("- Monitoreo intensivo en días calurosos")
                
                if 'Día Lluvioso' in valid_scenarios and valid_scenarios['🌧️ Día Lluvioso'] < np.mean(list(valid_scenarios.values())):
                    st.write("🌧️ **Beneficio de la lluvia**: La precipitación reduce significativamente el NO₂")
                
                range_scenarios = max(valid_scenarios.values()) - min(valid_scenarios.values())
                st.write(f"📊 **Variabilidad por condiciones**: {range_scenarios:.1f} μg/m³ de diferencia entre escenarios")


def diagnose_perfect_correlations(model: LinearGAM, feature_names: List[str], X_test: pd.DataFrame):
    """Diagnostica las causas de correlaciones perfectas entre efectos parciales."""
    st.write("### 🔬 Diagnóstico de Correlaciones Perfectas")
    
    # Variables a diagnosticar
    analysis_vars = [var for var in feature_names 
                    if var in VARIABLE_METADATA and not any(temp in var for temp in ['sin', 'cos'])][:6]
    
    if len(analysis_vars) < 2:
        st.warning("No hay suficientes variables para diagnóstico.")
        return
    
    st.write("#### 1. Análisis de Multicolinealidad en Datos Originales")
    
    # Calcular correlaciones en datos originales
    original_corr_matrix = X_test[analysis_vars].corr()
    
    # Mostrar correlaciones altas en datos originales
    high_original_corr = []
    for i, var1 in enumerate(analysis_vars):
        for j, var2 in enumerate(analysis_vars):
            if i < j and abs(original_corr_matrix.iloc[i, j]) > 0.8:
                high_original_corr.append((var1, var2, original_corr_matrix.iloc[i, j]))
    
    if high_original_corr:
        st.warning("⚠️ **Multicolinealidad detectada en datos originales:**")
        for var1, var2, corr in high_original_corr:
            var1_name = VARIABLE_METADATA.get(var1, {}).get('name', var1)
            var2_name = VARIABLE_METADATA.get(var2, {}).get('name', var2)
            st.write(f"- {var1_name} ↔ {var2_name}: r={corr:.3f}")
        st.write("**Recomendación**: Considerar eliminar una de las variables correlacionadas.")
    else:
        st.success("✅ No hay multicolinealidad significativa en datos originales.")
    
    st.write("#### 2. Análisis de Efectos Parciales Individuales")
    
    # Analizar efectos parciales para cada variable
    effects_analysis = {}
    
    for var in analysis_vars[:4]:  # Limitar para eficiencia
        if var in VARIABLE_METADATA:
            range_var = VARIABLE_METADATA[var]['typical_range']
            values = np.linspace(range_var[0], range_var[1], 20)
            
            # Calcular efectos usando diferentes métodos
            try:
                # Método 1: Efectos parciales directos del modelo
                XX = np.zeros((len(values), len(feature_names)))
                for feat_idx, feat_name in enumerate(feature_names):
                    if feat_name == var:
                        XX[:, feat_idx] = values
                    elif feat_name in X_test.columns:
                        XX[:, feat_idx] = X_test[feat_name].mean()
                
                predictions = model.predict(XX)
                
                # Método 2: Usar partial_dependence del modelo (si está disponible)
                try:
                    var_idx = feature_names.index(var)
                    pd_effects = []
                    for val in values:
                        XX_single = np.zeros((1, len(feature_names)))
                        XX_single[0, var_idx] = val
                        for feat_idx, feat_name in enumerate(feature_names):
                            if feat_idx != var_idx and feat_name in X_test.columns:
                                XX_single[0, feat_idx] = X_test[feat_name].mean()
                        
                        pd_effect = model.partial_dependence(term=var_idx, X=XX_single)
                        pd_effects.append(pd_effect)
                    
                    effects_analysis[var] = {
                        'predictions': predictions,
                        'partial_dependence': np.array(pd_effects),
                        'values': values,
                        'range': np.max(predictions) - np.min(predictions),
                        'std': np.std(predictions),
                        'is_linear': np.corrcoef(values, predictions)[0, 1] > 0.95
                    }
                    
                except Exception as e:
                    effects_analysis[var] = {
                        'predictions': predictions,
                        'partial_dependence': None,
                        'values': values,
                        'range': np.max(predictions) - np.min(predictions),
                        'std': np.std(predictions),
                        'is_linear': np.corrcoef(values, predictions)[0, 1] > 0.95,
                        'error': str(e)
                    }
                    
            except Exception as e:
                st.error(f"Error analizando {var}: {str(e)}")
    
    # Mostrar resultados del análisis
    if effects_analysis:
        st.write("**Características de los efectos parciales:**")
        
        analysis_df_data = []
        for var, analysis in effects_analysis.items():
            var_name = VARIABLE_METADATA.get(var, {}).get('name', var)
            analysis_df_data.append({
                'Variable': var_name,
                'Rango de Efecto': f"{analysis['range']:.4f}",
                'Desv. Estándar': f"{analysis['std']:.4f}",
                'Es Lineal': "Sí" if analysis['is_linear'] else "No",
                'Tiene Error': "Sí" if 'error' in analysis else "No"
            })
        
        import pandas as pd
        analysis_df = pd.DataFrame(analysis_df_data)
        st.dataframe(analysis_df, use_container_width=True)
        
        # Identificar variables problemáticas
        problematic_vars = []
        for var, analysis in effects_analysis.items():
            if analysis['std'] < 1e-8:
                problematic_vars.append(f"{var} (efecto constante)")
            elif analysis['is_linear']:
                problematic_vars.append(f"{var} (efecto perfectamente lineal)")
        
        if problematic_vars:
            st.warning("⚠️ **Variables con efectos problemáticos:**")
            for prob_var in problematic_vars:
                st.write(f"- {prob_var}")
            st.write("**Posibles causas:**")
            st.write("- Modelo demasiado simple para la variable")
            st.write("- Datos insuficientes en el rango de la variable")
            st.write("- Variable no tiene efecto real en NO₂")
    
    st.write("#### 3. Recomendaciones")
    
    recommendations = []
    
    if high_original_corr:
        recommendations.append("🔄 **Reducir multicolinealidad**: Eliminar variables altamente correlacionadas")
    
    if any('error' in analysis for analysis in effects_analysis.values()):
        recommendations.append("⚙️ **Revisar configuración GAM**: Algunos términos pueden estar mal especificados")
    
    linear_vars = [var for var, analysis in effects_analysis.items() if analysis.get('is_linear', False)]
    if linear_vars:
        recommendations.append(f"📈 **Variables lineales detectadas**: {', '.join(linear_vars)} - Considerar términos lineales simples")
    
    constant_vars = [var for var, analysis in effects_analysis.items() if analysis.get('std', 1) < 1e-8]
    if constant_vars:
        recommendations.append(f"🔒 **Variables sin efecto**: {', '.join(constant_vars)} - Considerar eliminar del modelo")
    
    if not recommendations:
        recommendations.append("✅ **No se detectaron problemas graves** en el análisis de efectos parciales")
    
    for rec in recommendations:
        st.write(rec)


# ==================== FUNCIÓN PRINCIPAL ====================

def training_page():
    """Función principal del módulo de entrenamiento."""
    
    # Inicializar trainer
    trainer = GAMTrainer()
    
    # Panel de información
    show_info_panel()
    
    # Cargar datos
    if not st.session_state.training_data_loaded:
        if st.button("Cargar datos para entrenamiento", type="primary"):
            with st.spinner("Cargando datos de entrenamiento..."):
                trainer.df_master = trainer.load_data()
                if not trainer.df_master.empty:
                    st.session_state.training_data_loaded = True
                    st.success("Datos cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    trainer.df_master = trainer.load_data()
    
    if trainer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Configuración del modelo
    st.header("⚙️ Configuración del Modelo GAM")
    
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
            index=1  # zscore por defecto
        )
        
        # Preprocesamiento
        preprocessing = st.selectbox(
            "Preprocesamiento temporal",
            options=list(PREPROCESSING_OPTIONS.keys()),
            format_func=lambda x: PREPROCESSING_OPTIONS[x],
            index=0  # sin_cos por defecto
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
            
            selected_in_category = st.multiselect(
                f"Variables de {category}",
                available_vars,
                default=available_vars,
                help=f"Selecciona las variables de {category.lower()} para el modelo"
            )
            selected_features.extend(selected_in_category)
    
    if not selected_features:
        st.warning("Selecciona al menos una variable para continuar.")
        return
    
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
        # Aplicar transformaciones básicas (no outliers aún)
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
    model_filename = f'data/models/gam_model_{sensor_seleccionado}_{outlier_method}_{preprocessing}.pkl'
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
            st.info("No existe un modelo entrenado con esta configuración")
    
    with col2:
        train_button = st.button("🚀 Entrenar Nuevo Modelo", type="secondary")
    
    # Ejecutar análisis o entrenamiento
    if analyze_button and model_exists:
        with st.spinner("Cargando y analizando modelo..."):
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
                
                # Mostrar resultados
                show_model_metrics(metrics)
                show_residual_analysis(metrics['residuals'])
                
                # Tabs para análisis completo del modelo recién entrenado
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🕐 Efectos Temporales",
                    "🔬 Efectos de Variables", 
                    "📊 Dependencias Parciales",
                    "🔄 Análisis de Interacciones"
                ])
                
                with tab1:
                    show_temporal_effects(trainer, model, selected_features)
                
                with tab2:
                    show_variable_effects(trainer, model, selected_features)
                
                with tab3:
                    show_partial_dependence_plots(model, selected_features)
                
                with tab4:
                    show_interaction_analysis(trainer, model, selected_features, X_test)
    
    elif train_button:
        with st.spinner("Entrenando modelo GAM... Esto puede tardar varios minutos."):
            try:
                # Preparar datos de entrenamiento
                X_train = train_df[selected_features].copy()
                y_train = train_df['no2_value'].copy()
                X_test = test_df[selected_features].copy()
                y_test = test_df['no2_value'].copy()
                
                # Debug: mostrar información de las features
                st.write("🔍 **Debug - Features seleccionadas:**")
                st.write(f"- Número de features: {len(selected_features)}")
                st.write(f"- Features: {selected_features}")
                st.write(f"- Columnas en train_df: {list(train_df.columns)}")
                st.write(f"- Columnas en X_train: {list(X_train.columns)}")
                st.write(f"- Shape X_train: {X_train.shape}")
                st.write(f"- Shape y_train: {y_train.shape}")
                
                # Verificar que las features existen en los datos
                missing_features = [f for f in selected_features if f not in train_df.columns]
                if missing_features:
                    st.error(f"❌ Features faltantes en los datos: {missing_features}")
                    st.write("Columnas disponibles en train_df:")
                    st.write(list(train_df.columns))
                    return
                
                # Validar configuración antes del entrenamiento
                if not trainer.validate_gam_config(selected_features, X_train):
                    st.error("❌ Configuración inválida. No se puede entrenar el modelo.")
                    return
                
                # Escalar características
                X_train_scaled, X_test_scaled, scaler_dict = trainer.scale_features(
                    X_train, X_test, selected_features
                )
                
                # Escalar objetivo
                y_train_scaled, scaler_target = trainer.scale_target(y_train)
                
                # Entrenar modelo
                model = trainer.train_gam_model(X_train_scaled, y_train_scaled, selected_features)
                
                # Guardar modelo
                saved_path = trainer.save_model(
                    model, selected_features, scaler_dict, scaler_target,
                    sensor_seleccionado, outlier_method, preprocessing
                )
                
                st.success(f"¡Modelo entrenado y guardado exitosamente en {saved_path}!")
                
                # Evaluar modelo recién entrenado
                metrics = trainer.evaluate_model(model, X_test_scaled, y_test, scaler_target)
                
                # Mostrar resultados
                show_model_metrics(metrics)
                show_residual_analysis(metrics['residuals'])
                
                # Tabs para análisis completo del modelo recién entrenado
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🕐 Efectos Temporales",
                    "🔬 Efectos de Variables", 
                    "📊 Dependencias Parciales",
                    "🔄 Análisis de Interacciones"
                ])
                
                with tab1:
                    show_temporal_effects(trainer, model, selected_features)
                
                with tab2:
                    show_variable_effects(trainer, model, selected_features)
                
                with tab3:
                    show_partial_dependence_plots(model, selected_features)
                
                with tab4:
                    show_interaction_analysis(trainer, model, selected_features, X_test)
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {str(e)}")
                st.exception(e)
    
    # Pie de página
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Entrenamiento GAM para sensor {sensor_seleccionado} | Configuración: {OUTLIER_METHODS[outlier_method]} + {PREPROCESSING_OPTIONS[preprocessing]}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    training_page() 