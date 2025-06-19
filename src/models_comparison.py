"""
Página de comparación de modelos para el Dashboard de Análisis de NO₂ en Madrid.

Este módulo proporciona una interfaz para comparar el rendimiento de diferentes
modelos de machine learning aplicados a la predicción de NO₂.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import datetime
from pathlib import Path

# Importar configuración centralizada
from src.config import (
    MODEL_CONFIGS, 
    VARIABLE_METADATA, 
    FILE_PATHS,
    get_model_config,
    get_session_state_key
)

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================

PAGE_CONFIG = {
    'title': 'Comparación de Modelos',
    'icon': '⚖️',
    'description': 'Análisis comparativo del rendimiento de modelos de ML para predicción de NO₂',
    'sidebar_sections': [
        'Configuración de Comparación',
        'Filtros de Datos',
        'Métricas de Evaluación'
    ]
}

# Métricas de evaluación disponibles
AVAILABLE_METRICS = {
    'mae': {'name': 'Error Absoluto Medio', 'unit': 'μg/m³', 'lower_is_better': True},
    'rmse': {'name': 'Error Cuadrático Medio', 'unit': 'μg/m³', 'lower_is_better': True},
    'r2': {'name': 'Coeficiente de Determinación', 'unit': '', 'lower_is_better': False},
    'mape': {'name': 'Error Porcentual Absoluto', 'unit': '%', 'lower_is_better': True},
    'bias': {'name': 'Sesgo Medio', 'unit': 'μg/m³', 'lower_is_better': True}
}

# ==================== FUNCIONES DE DATOS SIMULADOS ====================

def generate_sample_model_results() -> Dict[str, pd.DataFrame]:
    """
    Genera datos de ejemplo para demostrar la funcionalidad.
    En implementación real, cargaría resultados guardados de modelos.
    """
    np.random.seed(42)
    
    # Generar datos temporales
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    n_samples = len(dates)
    
    # Valores reales simulados (con patrones temporales)
    hour_effect = 10 * np.sin(2 * np.pi * dates.hour / 24)
    seasonal_effect = 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 8, n_samples)
    y_true = 30 + hour_effect + seasonal_effect + noise
    y_true = np.clip(y_true, 0, None)  # NO₂ no puede ser negativo
    
    models_data = {}
    
    # Simular resultados para cada modelo
    for model_name, config in MODEL_CONFIGS.items():
        # Cada modelo tiene diferentes características de error
        if model_name == 'xgboost':
            # XGBoost: Mejor en general, menor sesgo
            error_std = 6
            bias = 0.5
        elif model_name == 'gam':
            # GAM: Bueno pero menos flexible
            error_std = 8
            bias = -1.2
        elif model_name == 'bayesian':
            # Bayesian: Buena calibración, incertidumbre
            error_std = 7
            bias = 0.8
        else:
            error_std = 10
            bias = 2
        
        # Generar predicciones con error correlacionado
        model_error = np.random.normal(bias, error_std, n_samples)
        y_pred = y_true + model_error
        y_pred = np.clip(y_pred, 0, None)
        
        # Crear DataFrame para el modelo
        model_df = pd.DataFrame({
            'datetime': dates,
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_pred - y_true,
            'abs_error': np.abs(y_pred - y_true),
            'hour': dates.hour,
            'month': dates.month,
            'day_of_week': dates.dayofweek
        })
        
        models_data[model_name] = model_df
    
    return models_data

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de evaluación para un modelo."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Filtrar valores válidos
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {metric: np.nan for metric in AVAILABLE_METRICS.keys()}
    
    # Calcular métricas
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE (evitar división por cero)
    mask_nonzero = y_true_clean != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / y_true_clean[mask_nonzero])) * 100
    else:
        mape = np.nan
    
    bias = np.mean(y_pred_clean - y_true_clean)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'bias': bias
    }

# ==================== FUNCIONES DE SIDEBAR ====================

def create_comparison_sidebar() -> Dict:
    """Crea el sidebar con controles para la comparación de modelos."""
    st.sidebar.header("⚖️ Configuración de Comparación")
    
    # Selección de modelos a comparar
    st.sidebar.subheader("🤖 Modelos a Comparar")
    available_models = list(MODEL_CONFIGS.keys())
    
    selected_models = []
    for model_key in available_models:
        config = get_model_config(model_key)
        default_selected = model_key in ['xgboost', 'gam']  # Pre-seleccionar algunos
        
        if st.sidebar.checkbox(
            f"{config.get('icon', '🔧')} {config.get('name', model_key)}",
            value=default_selected,
            key=f"model_select_{model_key}"
        ):
            selected_models.append(model_key)
    
    if not selected_models:
        st.sidebar.warning("⚠️ Selecciona al menos un modelo")
        return {'selected_models': []}
    
    # Filtros temporales
    st.sidebar.subheader("📅 Filtros Temporales")
    
    date_filter = st.sidebar.selectbox(
        "Período de Análisis",
        options=['Todos los datos', 'Último mes', 'Últimos 3 meses', 'Último año', 'Personalizado'],
        index=0
    )
    
    custom_dates = None
    if date_filter == 'Personalizado':
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Desde", value=datetime.date(2023, 1, 1))
        with col2:
            end_date = st.date_input("Hasta", value=datetime.date(2023, 12, 31))
        custom_dates = (start_date, end_date)
    
    # Métricas de evaluación
    st.sidebar.subheader("📊 Métricas de Evaluación")
    
    selected_metrics = []
    for metric_key, metric_info in AVAILABLE_METRICS.items():
        default_selected = metric_key in ['mae', 'rmse', 'r2']
        
        if st.sidebar.checkbox(
            f"{metric_info['name']} ({metric_info['unit']})" if metric_info['unit'] else metric_info['name'],
            value=default_selected,
            key=f"metric_select_{metric_key}"
        ):
            selected_metrics.append(metric_key)
    
    # Opciones de visualización
    st.sidebar.subheader("📈 Opciones de Visualización")
    
    show_scatter = st.sidebar.checkbox("Gráfico de Dispersión", value=True)
    show_time_series = st.sidebar.checkbox("Series Temporales", value=True)
    show_residuals = st.sidebar.checkbox("Análisis de Residuos", value=False)
    show_feature_importance = st.sidebar.checkbox("Importancia de Variables", value=False)
    
    return {
        'selected_models': selected_models,
        'date_filter': date_filter,
        'custom_dates': custom_dates,
        'selected_metrics': selected_metrics,
        'show_scatter': show_scatter,
        'show_time_series': show_time_series,
        'show_residuals': show_residuals,
        'show_feature_importance': show_feature_importance
    }

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def create_metrics_comparison_table(models_data: Dict[str, pd.DataFrame], selected_models: List[str]) -> pd.DataFrame:
    """Crea una tabla comparativa de métricas."""
    results = {}
    
    for model in selected_models:
        if model in models_data:
            df = models_data[model]
            metrics = calculate_metrics(df['y_true'].values, df['y_pred'].values)
            results[model] = metrics
    
    # Convertir a DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # Formatear y ordenar por mejor rendimiento
    for metric in comparison_df.columns:
        if metric in AVAILABLE_METRICS:
            comparison_df[metric] = comparison_df[metric].round(3)
    
    return comparison_df

def plot_metrics_comparison(comparison_df: pd.DataFrame, selected_metrics: List[str]):
    """Crea gráficos de barras para comparar métricas."""
    if comparison_df.empty or not selected_metrics:
        st.warning("No hay datos para mostrar")
        return
    
    # Crear subplots
    n_metrics = len(selected_metrics)
    cols = min(2, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[AVAILABLE_METRICS[m]['name'] for m in selected_metrics],
        vertical_spacing=0.12
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for i, metric in enumerate(selected_metrics):
        row = i // cols + 1
        col = i % cols + 1
        
        if metric in comparison_df.columns:
            values = comparison_df[metric]
            models = comparison_df.index
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * rows,
        title_text="Comparación de Métricas por Modelo",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_comparison(models_data: Dict[str, pd.DataFrame], selected_models: List[str]):
    """Crea gráficos de dispersión para comparar predicciones vs valores reales."""
    if not selected_models:
        return
    
    fig = make_subplots(
        rows=1, cols=len(selected_models),
        subplot_titles=[f"{get_model_config(m).get('name', m)}" for m in selected_models],
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for i, model in enumerate(selected_models):
        if model in models_data:
            df = models_data[model].sample(min(1000, len(models_data[model])))  # Muestra para performance
            
            fig.add_trace(
                go.Scatter(
                    x=df['y_true'],
                    y=df['y_pred'],
                    mode='markers',
                    marker=dict(
                        color=colors[i % len(colors)],
                        opacity=0.6,
                        size=3
                    ),
                    name=model,
                    showlegend=False
                ),
                row=1, col=i+1
            )
            
            # Línea diagonal (predicción perfecta)
            min_val = min(df['y_true'].min(), df['y_pred'].min())
            max_val = max(df['y_true'].max(), df['y_pred'].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Predicción Perfecta',
                    showlegend=i == 0
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        height=400,
        title_text="Predicciones vs Valores Reales",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="NO₂ Real (μg/m³)")
    fig.update_yaxes(title_text="NO₂ Predicho (μg/m³)")
    
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series_comparison(models_data: Dict[str, pd.DataFrame], selected_models: List[str]):
    """Crea comparación de series temporales."""
    if not selected_models:
        return
    
    # Tomar una muestra representativa de datos (ej: una semana)
    sample_start = pd.Timestamp('2023-06-01')
    sample_end = pd.Timestamp('2023-06-08')
    
    fig = go.Figure()
    
    # Agregar valores reales (solo una vez)
    first_model = selected_models[0]
    if first_model in models_data:
        sample_df = models_data[first_model][
            (models_data[first_model]['datetime'] >= sample_start) &
            (models_data[first_model]['datetime'] <= sample_end)
        ]
        
        fig.add_trace(
            go.Scatter(
                x=sample_df['datetime'],
                y=sample_df['y_true'],
                mode='lines',
                name='Valores Reales',
                line=dict(color='black', width=2)
            )
        )
    
    # Agregar predicciones de cada modelo
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for i, model in enumerate(selected_models):
        if model in models_data:
            sample_df = models_data[model][
                (models_data[model]['datetime'] >= sample_start) &
                (models_data[model]['datetime'] <= sample_end)
            ]
            
            config = get_model_config(model)
            
            fig.add_trace(
                go.Scatter(
                    x=sample_df['datetime'],
                    y=sample_df['y_pred'],
                    mode='lines',
                    name=f"{config.get('icon', '🔧')} {config.get('name', model)}",
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dot')
                )
            )
    
    fig.update_layout(
        title="Comparación de Series Temporales (Muestra Semanal)",
        xaxis_title="Fecha y Hora",
        yaxis_title="Concentración NO₂ (μg/m³)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== FUNCIÓN PRINCIPAL ====================

def models_comparison_page():
    """Función principal de la página de comparación de modelos."""
    
    # Configurar sidebar
    config = create_comparison_sidebar()
    
    if not config['selected_models']:
        st.info("👈 Selecciona al menos un modelo en el panel lateral para comenzar la comparación.")
        return
    
    # Mostrar información de los modelos seleccionados
    st.subheader("🤖 Modelos Seleccionados")
    
    cols = st.columns(len(config['selected_models']))
    for i, model in enumerate(config['selected_models']):
        model_config = get_model_config(model)
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: {model_config.get('color', '#grey')}20; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h3 style="margin: 0; color: {model_config.get('color', '#grey')};">
                    {model_config.get('icon', '🔧')} {model_config.get('name', model)}
                </h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    {model_config.get('description', 'Modelo de Machine Learning')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Cargar o generar datos (en implementación real, cargarían datos guardados)
    with st.spinner("Cargando resultados de modelos..."):
        models_data = generate_sample_model_results()
    
    # Aplicar filtros temporales si es necesario
    # [Código de filtrado temporal aquí]
    
    # === SECCIÓN 1: TABLA COMPARATIVA DE MÉTRICAS ===
    st.header("📊 Comparación de Métricas")
    
    if config['selected_metrics']:
        comparison_df = create_metrics_comparison_table(models_data, config['selected_models'])
        
        if not comparison_df.empty:
            # Mostrar tabla
            st.dataframe(
                comparison_df[config['selected_metrics']],
                use_container_width=True
            )
            
            # Gráfico de barras
            plot_metrics_comparison(comparison_df, config['selected_metrics'])
            
            # Identificar mejor modelo por métrica
            st.subheader("🏆 Mejor Modelo por Métrica")
            best_models = {}
            
            for metric in config['selected_metrics']:
                if metric in comparison_df.columns:
                    metric_info = AVAILABLE_METRICS[metric]
                    if metric_info['lower_is_better']:
                        best_model = comparison_df[metric].idxmin()
                        best_value = comparison_df[metric].min()
                    else:
                        best_model = comparison_df[metric].idxmax()
                        best_value = comparison_df[metric].max()
                    
                    best_models[metric] = (best_model, best_value)
            
            cols = st.columns(len(best_models))
            for i, (metric, (best_model, best_value)) in enumerate(best_models.items()):
                with cols[i]:
                    model_config = get_model_config(best_model)
                    st.metric(
                        label=f"Mejor {AVAILABLE_METRICS[metric]['name']}",
                        value=f"{best_value:.3f}",
                        help=f"Modelo: {model_config.get('name', best_model)}"
                    )
    else:
        st.warning("⚠️ Selecciona al menos una métrica en el panel lateral.")
    
    # === SECCIÓN 2: VISUALIZACIONES ===
    st.header("📈 Análisis Visual")
    
    if config['show_scatter']:
        st.subheader("🎯 Precisión de Predicciones")
        plot_scatter_comparison(models_data, config['selected_models'])
    
    if config['show_time_series']:
        st.subheader("⏰ Evolución Temporal")
        plot_time_series_comparison(models_data, config['selected_models'])
    
    if config['show_residuals']:
        st.subheader("📊 Análisis de Residuos")
        st.info("🚧 Esta sección estará disponible en la próxima versión.")
    
    if config['show_feature_importance']:
        st.subheader("🔍 Importancia de Variables")
        st.info("🚧 Esta sección estará disponible en la próxima versión.")
    
    # === SECCIÓN 3: CONCLUSIONES ===
    with st.expander("📋 Resumen y Recomendaciones", expanded=False):
        st.markdown("""
        ### 🎯 Interpretación de Resultados
        
        - **MAE/RMSE**: Menores valores indican mejor precisión
        - **R²**: Valores más altos (cercanos a 1) indican mejor ajuste
        - **MAPE**: Porcentaje de error promedio
        - **Bias**: Tendencia sistemática de sobreestimación (+) o subestimación (-)
        
        ### 📊 Consideraciones para la Selección de Modelo
        
        1. **Precisión General**: Evaluar MAE y RMSE
        2. **Capacidad Explicativa**: Considerar R²
        3. **Sesgo Sistemático**: Verificar bias
        4. **Robustez**: Analizar consistencia temporal
        
        ### 💡 Próximos Pasos
        
        - Validación con datos externos
        - Análisis de incertidumbre
        - Evaluación en condiciones extremas
        - Implementación en producción
        """) 