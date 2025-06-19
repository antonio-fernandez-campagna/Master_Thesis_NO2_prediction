"""
Configuración centralizada para la aplicación de análisis de NO₂ en Madrid.

Este módulo contiene todas las constantes, configuraciones y metadatos
utilizados a lo largo de la aplicación.
"""

import streamlit as st
from typing import Dict, List, Any

# ==================== CONFIGURACIÓN DE LA APLICACIÓN ====================

PAGE_CONFIG = {
    "page_title": "Dashboard Madrid - Calidad del Aire y Tráfico",
    "page_icon": "🌍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ==================== INFORMACIÓN DEL PROYECTO ====================

PROJECT_INFO = {
    'title': 'Dashboard de Análisis de NO₂ en Madrid',
    'subtitle': 'Análisis integrado de contaminación atmosférica, tráfico y meteorología',
    'version': '2.0',
    'authors': ['Antonio Fernández'],
    'institution': 'Master Thesis - NO₂ Prediction',
    'year': '2024'
}

# ==================== RUTAS DE DATOS ====================

DATA_PATHS = {
    'NO2_DATA': '/Users/antoniofernandez/Projects/Master_Thesis_NO2_prediction/data/more_processed/no2_data_master.parquet',
    'TRAFFIC_NO2_DATA': '/Users/antoniofernandez/Projects/Master_Thesis_NO2_prediction/data/more_processed/no2_with_traffic_and_meteo.parquet',
    'SENSOR_MAPPING': '/Users/antoniofernandez/Projects/Master_Thesis_NO2_prediction/data/info/sensors_mapping.parquet',
    'MODELS_PATH': '/Users/antoniofernandez/Projects/Master_Thesis_NO2_prediction/models/',
    'FIGURES_PATH': '/Users/antoniofernandez/Projects/Master_Thesis_NO2_prediction/figures/'
}

# ==================== FUENTES DE DATOS ====================

DATA_SOURCES = {
    'Calidad del Aire': {
        'url': 'https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/',
        'granularity': 'Horaria',
        'description': 'Datos de concentración de NO₂ de las estaciones de medición de Madrid'
    },
    'Tráfico': {
        'url': 'https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/',
        'granularity': '15 minutos (agregado a horario)',
        'description': 'Intensidad, carga, ocupación y velocidad del tráfico rodado'
    },
    'Meteorología': {
        'url': 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels',
        'granularity': 'Horaria',
        'description': 'Variables meteorológicas de ERA5 (temperatura, viento, precipitación, etc.)'
    }
}

# ==================== HIPÓTESIS CIENTÍFICAS ====================

SCIENTIFIC_HYPOTHESIS = [
    {
        'title': '🌧️ La lluvia reduce los niveles de NO₂',
        'description': 'La precipitación arrastra contaminantes atmosféricos mediante deposición húmeda',
        'reference': 'Zhang et al. (2004) - Atmospheric Environment'
    },
    {
        'title': '🌬️ El viento dispersa el NO₂',
        'description': 'Mayor velocidad del viento aumenta la dispersión y reduce concentraciones',
        'reference': 'Kukkonen et al. (2003) - Atmospheric Environment'
    },
    {
        'title': '🌡️ Temperaturas altas favorecen la fotólisis',
        'description': 'El NO₂ se descompone con luz solar y temperaturas elevadas',
        'reference': 'Finlayson-Pitts & Pitts (2000) - Chemistry of the atmosphere'
    },
    {
        'title': '☀️ Radiación solar reduce NO₂',
        'description': 'La radiación solar descompone NO₂ → NO + O durante el día',
        'reference': 'Seinfeld & Pandis (2016) - Atmospheric Chemistry'
    },
    {
        'title': '🚗 Tráfico incrementa NO₂',
        'description': 'El tráfico rodado es la principal fuente de NO₂ en entornos urbanos',
        'reference': 'Cyrys et al. (2003) - Science of the Total Environment'
    },
    {
        'title': '🕒 Patrones temporales cíclicos',
        'description': 'NO₂ presenta ciclos diarios y semanales relacionados con actividad humana',
        'reference': 'Vardoulakis et al. (2003) - Atmospheric Environment'
    },
    {
        'title': '💨 Presión atmosférica y acumulación',
        'description': 'Altas presiones producen estancamiento y acumulación de contaminantes',
        'reference': 'Jacob & Winner (2009) - Atmospheric Environment'
    },
    {
        'title': '💧 Humedad modula la química del NO₂',
        'description': 'La humedad influye en deposición húmeda y formación de aerosoles',
        'reference': 'Beig et al. (2007) - Meteorology and air quality'
    }
]

# ==================== INFORMACIÓN DE VARIABLES ====================

VARIABLES_INFO = {
    'Temporales': {
        'description': 'Variables cíclicas (sin/cos) que capturan patrones temporales',
        'variables': ['Hora', 'Día de la semana', 'Mes', 'Día del año', 'Estación', 'Fin de semana'],
        'rationale': 'Las funciones trigonométricas preservan la continuidad temporal (ej: 23h → 0h)'
    },
    'Tráfico': {
        'description': 'Métricas de densidad y fluidez del tráfico rodado',
        'variables': ['Intensidad (veh/h)', 'Carga (%)', 'Ocupación (%)', 'Velocidad media (km/h)'],
        'rationale': 'El tráfico es la principal fuente antropogénica de NO₂ en Madrid'
    },
    'Meteorológicas': {
        'description': 'Variables atmosféricas que influyen en la dispersión y química del NO₂',
        'variables': ['Temperatura (°C)', 'Punto de rocío (°C)', 'Velocidad del viento (km/h)', 
                     'Presión (hPa)', 'Precipitación (mm)', 'Radiación solar (W/m²)'],
        'rationale': 'Controlan la dispersión, transporte y transformación química del NO₂'
    }
}

# ==================== CONFIGURACIÓN DE TABS ====================

TAB_CONFIG = {
    "🏠 Inicio": {
        "icon": "🏠",
        "description": "Página de bienvenida e introducción al proyecto",
        "module": "welcome_page"
    },
    "Análisis NO₂": {
        "icon": "🌍",
        "description": "Análisis temporal y espacial de niveles de NO₂",
        "module": "no2_analysis"
    },
    "Mapeo Sensores": {
        "icon": "🗺️", 
        "description": "Asignación entre sensores de NO₂ y tráfico",
        "module": "sensor_mapping"
    },
    "Correlaciones": {
        "icon": "📊",
        "description": "Análisis de correlaciones entre NO₂, tráfico y meteorología",
        "module": "correlations_analysis"
    },
    "Entrenamiento GAM": {
        "icon": "🤖",
        "description": "Entrenamiento y análisis de modelos GAM para predicción de NO₂",
        "module": "gam_training"
    },
    "XGBoost Unificado": {
        "icon": "⚡",
        "description": "Entrenamiento de modelos XGBoost individuales y globales con análisis comparativo",
        "module": "xgboost_unified"
    },
    "Nowcasting Bayesiano": {
        "icon": "🧠",
        "description": "Nowcasting de NO₂ con redes neuronales bayesianas e incertidumbre",
        "module": "bayesian_nowcasting"
    }
}

# ==================== CONFIGURACIÓN DE ESTILOS ====================

STYLE_CONFIG = {
    'primary_color': '#1f77b4',
    'secondary_color': '#ff7f0e',
    'success_color': '#2ca02c',
    'warning_color': '#d62728',
    'info_color': '#9467bd',
    'background_color': '#f0f8ff',
    'text_color': '#666'
}

# ==================== CONFIGURACIÓN DE CACHE ====================

CACHE_CONFIG = {
    'ttl': 3600,  # 1 hora
    'show_spinner': True,
    'suppress_st_warning': True
}

# ==================== MÉTRICAS Y LÍMITES ====================

METRICS_CONFIG = {
    'no2_limits': {
        'who_annual': 40,  # µg/m³ límite anual OMS
        'who_daily': 200,  # µg/m³ límite diario OMS
        'eu_annual': 40,   # µg/m³ límite anual UE
        'eu_hourly': 200   # µg/m³ límite horario UE
    },
    'traffic_thresholds': {
        'low_intensity': 500,
        'medium_intensity': 1500,
        'high_intensity': 3000
    }
}

# ==================== CONFIGURACIÓN DE MODELOS ====================

MODEL_CONFIG = {
    'gam': {
        'default_params': {
            'n_splines': 25,
            'spline_order': 3,
            'lam': 0.6
        },
        'feature_groups': {
            'temporal': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
            'meteorological': ['temperature_2m', 'dewpoint_2m', 'windspeed_10m', 'surface_pressure', 'precipitation'],
            'traffic': ['intensidad', 'carga', 'ocupacion', 'velocidad']
        }
    },
    'xgboost': {
        'default_params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    },
    'bayesian': {
        'default_params': {
            'hidden_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }
    }
}

# ==================== FUNCIONES DE UTILIDAD ====================

def get_data_path(key: str) -> str:
    """Obtiene la ruta de un archivo de datos."""
    return DATA_PATHS.get(key, "")

def get_tab_config(tab_name: str) -> Dict[str, Any]:
    """Obtiene la configuración de un tab específico."""
    return TAB_CONFIG.get(tab_name, {})

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Obtiene la configuración de un modelo específico."""
    return MODEL_CONFIG.get(model_type, {})

def get_style_config() -> Dict[str, str]:
    """Obtiene la configuración de estilos."""
    return STYLE_CONFIG

def get_cache_config() -> Dict[str, Any]:
    """Obtiene la configuración de cache."""
    return CACHE_CONFIG 