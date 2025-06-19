# Dashboard Madrid - Análisis de NO₂

## Estructura del Proyecto Refactorizado

Este directorio contiene el código fuente refactorizado para la aplicación de análisis de contaminación de NO₂ en Madrid.

### Archivos Principales

#### `app.py`
Aplicación principal de Streamlit que orquesta todos los módulos de análisis.

#### `config.py`
Configuración centralizada que contiene:
- Rutas de archivos de datos
- Configuración de la aplicación
- Constantes y parámetros
- Hipótesis científicas
- Información de variables

#### `utils.py`
Utilidades comunes reutilizables:
- Funciones de carga de datos con cache
- Funciones de visualización
- Procesamiento de datos
- Métricas y validaciones
- Manejo de errores

### Módulos de Análisis

#### `welcome_page.py`
Página de bienvenida e introducción al proyecto.

#### `no2_analysis.py`
Análisis temporal y espacial de niveles de NO₂.

#### `sensor_mapping.py`
Mapeo y asignación entre sensores de NO₂ y tráfico.

#### `correlations_analysis.py`
Análisis de correlaciones entre NO₂, tráfico y meteorología.

#### `gam_training.py`
Entrenamiento y análisis de modelos GAM (Generalized Additive Models).

#### `xgboost_training.py`
Entrenamiento y análisis de modelos XGBoost.

#### `bayesian_nowcasting.py`
Nowcasting de NO₂ con redes neuronales bayesianas e incertidumbre.

## Mejoras del Refactoring

### ✅ Eliminación de Duplicados
- Eliminados archivos con sufijo `copy`
- Eliminados archivos marcados como `NOT_USED`
- Eliminados archivos obsoletos no referenciados

### ✅ Nomenclatura Mejorada
- Nombres de archivo más claros y concisos
- Corrección de errores tipográficos (`analsis` → `analysis`)
- Nomenclatura consistente en inglés para módulos técnicos

### ✅ Estructura Modular
- Configuración centralizada en `config.py`
- Utilidades comunes en `utils.py`
- Separación clara de responsabilidades
- Importaciones organizadas

### ✅ Funciones Comunes Consolidadas
- Funciones de carga de datos centralizadas
- Funciones de visualización reutilizables
- Validaciones y manejo de errores consistente
- Cache optimizado para rendimiento

## Cómo Ejecutar

```bash
# Desde el directorio raíz del proyecto
streamlit run src/app.py
```

## Dependencias

La aplicación requiere las siguientes librerías principales:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `tensorflow` (para módulo bayesiano)
- `pygam` (para modelos GAM)

## Estructura de Datos

La aplicación espera los siguientes archivos de datos:
- `no2_data_master.parquet`: Datos principales de NO₂
- `no2_with_traffic_and_meteo.parquet`: Datos integrados con tráfico y meteorología
- `sensors_mapping.parquet`: Mapeo de sensores

## Configuración

Actualiza las rutas en `config.py` según tu configuración local:

```python
DATA_PATHS = {
    'NO2_DATA': 'ruta/a/tu/archivo/no2_data_master.parquet',
    'TRAFFIC_NO2_DATA': 'ruta/a/tu/archivo/no2_with_traffic_and_meteo.parquet',
    'SENSOR_MAPPING': 'ruta/a/tu/archivo/sensors_mapping.parquet',
    # ...
}
``` 