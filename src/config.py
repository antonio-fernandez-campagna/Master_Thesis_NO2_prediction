"""
Configuración centralizada para todos los modelos de predicción de NO2.

Este módulo contiene todas las constantes, configuraciones y metadatos
utilizados por los diferentes algoritmos de machine learning.
"""

from typing import Dict, List
import streamlit as st

# ==================== MÉTODOS DE FILTRADO DE OUTLIERS ====================

OUTLIER_METHODS = {
    'iqr': 'Rango Intercuartílico (IQR)',
    'zscore': 'Z-Score (Desviación Estándar)',
    'quantiles': 'Percentiles Extremos',
    'none': 'Sin filtrado'
}

# ==================== OPCIONES DE PREPROCESAMIENTO ====================

PREPROCESSING_OPTIONS = {
    'sin_cos': 'Variables Cíclicas (Sin/Cos)',
    'none': 'Sin preprocesamiento'
}

# ==================== CATEGORÍAS DE VARIABLES ====================

VARIABLE_CATEGORIES = {
    "Variables Temporales": [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos',
        'weekend', 'season_sin', 'season_cos', 'hour', 'month', 'day_of_week', 'day_of_year'
    ],
    "Variables de Tráfico": ['intensidad', 'carga', 'ocupacion', 'vmed'],
    "Variables Meteorológicas": ['d2m', 't2m', 'ssr', 'ssrd', 'u10', 'v10', 'sp', 'tp', 'wind_speed', 'wind_direction']
}

# ==================== METADATOS DE VARIABLES ====================

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

# ==================== COLUMNAS PARA DETECCIÓN DE OUTLIERS ====================

COLUMNS_FOR_OUTLIERS = [
    'no2_value'
]

# ==================== CONFIGURACIONES DE MODELOS ====================

MODEL_CONFIGS = {
    'xgboost': {
        'name': 'XGBoost',
        'description': 'eXtreme Gradient Boosting',
        'icon': '🚀',
        'color': '#FF6B6B',
        'default_outlier_method': 'none',  # XGBoost maneja outliers naturalmente
        'default_preprocessing': 'none',   # XGBoost puede manejar variables temporales directamente
        'supports_feature_importance': True,
        'supports_early_stopping': True
    },
    'gam': {
        'name': 'GAM',
        'description': 'Generalized Additive Models',
        'icon': '📈',
        'color': '#4ECDC4',
        'default_outlier_method': 'iqr',
        'default_preprocessing': 'sin_cos',
        'supports_feature_importance': False,
        'supports_early_stopping': False
    },
    'bayesian': {
        'name': 'Bayesian NN',
        'description': 'Bayesian Neural Networks',
        'icon': '🧠',
        'color': '#45B7D1',
        'default_outlier_method': 'iqr',
        'default_preprocessing': 'sin_cos',
        'supports_feature_importance': False,
        'supports_early_stopping': True,
        'supports_uncertainty': True
    }
}

# ==================== CONFIGURACIONES DE STREAMLIT ====================

STREAMLIT_CONFIG = {
    'page_title': 'Predicción de NO₂',
    'page_icon': '🌍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# ==================== RUTAS DE ARCHIVOS ====================

FILE_PATHS = {
    'data': 'data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet',
    'models_dir': 'data/models',
    'figures_dir': 'data/figures'
}

# ==================== CONFIGURACIONES DE ENTRENAMIENTO ====================

TRAINING_CONFIGS = {
    'xgboost': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 50
    },
    'bayesian': {
        'epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.01,
        'early_stopping_patience': 25,
        'reduce_lr_patience': 12
    },
    'gam': {
        'n_splines': 25,
        'spline_order': 3,
        'reg_lambda': 0.6
    }
}

# ==================== FUNCIONES DE UTILIDAD PARA CONFIGURACIÓN ====================

def get_model_config(model_type: str) -> Dict:
    """Obtiene la configuración para un tipo de modelo específico."""
    return MODEL_CONFIGS.get(model_type, {})

def get_training_config(model_type: str) -> Dict:
    """Obtiene la configuración de entrenamiento para un tipo de modelo."""
    return TRAINING_CONFIGS.get(model_type, {})

def get_variable_categories() -> Dict[str, List[str]]:
    """Obtiene las categorías de variables disponibles."""
    return VARIABLE_CATEGORIES.copy()

def get_available_features(df_columns: List[str]) -> Dict[str, List[str]]:
    """
    Filtra las variables disponibles basándose en las columnas del DataFrame.
    
    Args:
        df_columns: Lista de columnas disponibles en el DataFrame
        
    Returns:
        Diccionario con categorías y variables disponibles
    """
    available_categories = {}
    
    for category, vars_list in VARIABLE_CATEGORIES.items():
        available_vars = []
        for var in vars_list:
            if var in df_columns or 'sin' in var or 'cos' in var:
                available_vars.append(var)
        
        if available_vars:  # Solo incluir categorías con variables disponibles
            available_categories[category] = available_vars
    
    return available_categories

def validate_feature_selection(selected_features: List[str]) -> bool:
    """Valida que la selección de features sea válida."""
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable para continuar.")
        return False
    
    # Validar que no haya conflictos entre variables cíclicas y no cíclicas
    temporal_vars = set(selected_features) & set(VARIABLE_CATEGORIES["Variables Temporales"])
    
    cyclical_vars = [var for var in temporal_vars if 'sin' in var or 'cos' in var]
    non_cyclical_vars = [var for var in temporal_vars if 'sin' not in var and 'cos' not in var]
    
    # Verificar si hay variables temporales base que también tienen versión cíclica seleccionada
    base_vars = {'hour', 'month', 'day_of_week', 'day_of_year'}
    cyclical_base_vars = set()
    
    for cyclical_var in cyclical_vars:
        for base_var in base_vars:
            if base_var in cyclical_var:
                cyclical_base_vars.add(base_var)
    
    conflicting_vars = set(non_cyclical_vars) & cyclical_base_vars
    
    if conflicting_vars:
        st.warning(f"⚠️ Variables conflictivas detectadas: {conflicting_vars}. "
                   f"No selecciones tanto la versión cíclica como la no cíclica de la misma variable temporal.")
        return False
    
    return True

def get_session_state_key(prefix: str, **kwargs) -> str:
    """
    Genera una clave única para session_state basada en parámetros.
    
    Args:
        prefix: Prefijo para la clave
        **kwargs: Parámetros adicionales para la clave
        
    Returns:
        Clave única como string
    """
    key_parts = [prefix]
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (list, tuple)):
            key_parts.append(f"{key}_{len(value)}")
        else:
            key_parts.append(f"{key}_{value}")
    
    return "_".join(str(part) for part in key_parts)

# ==================== CONFIGURACIONES DE STREAMLIT (ACTUALIZADO) ====================

PAGE_CONFIG = {
    'page_title': 'Dashboard Madrid NO₂',
    'page_icon': '🌍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': 'https://github.com/tu-repo/issues',
        'Report a bug': 'https://github.com/tu-repo/issues',
        'About': "Dashboard de análisis de NO₂ en Madrid - Tesis de Maestría"
    }
}

# ==================== CONFIGURACIÓN DE TABS ====================

TAB_CONFIG = {
    "🏠 Inicio": {
        'icon': '🏠',
        'description': 'Página principal con información del proyecto y guía de navegación',
        'requires_data': False
    },
    "Análisis NO₂": {
        'icon': '🌫️',
        'description': 'Análisis exploratorio de datos de concentración de NO₂',
        'requires_data': True
    },
    "Mapeo Sensores": {
        'icon': '🗺️',
        'description': 'Visualización geoespacial de sensores de calidad del aire y tráfico',
        'requires_data': True
    },
    "Correlaciones": {
        'icon': '🔗',
        'description': 'Análisis de correlaciones entre variables meteorológicas, tráfico y NO₂',
        'requires_data': True
    },
    "Entrenamiento GAM": {
        'icon': '📈',
        'description': 'Modelos aditivos generalizados para predicción de NO₂',
        'requires_data': True
    },
    "XGBoost Unificado": {
        'icon': '🚀',
        'description': 'Algoritmo de gradient boosting para predicción avanzada de NO₂',
        'requires_data': True
    },
    "Nowcasting Bayesiano": {
        'icon': '🧠',
        'description': 'Redes neuronales bayesianas para nowcasting con incertidumbre',
        'requires_data': True
    },
    "Comparación Modelos": {
        'icon': '⚖️',
        'description': 'Comparación y evaluación del rendimiento de diferentes modelos ML',
        'requires_data': True
    }
} 