"""
Página de bienvenida para el Dashboard de Análisis de NO₂ en Madrid.

Este módulo proporciona una introducción completa al proyecto, explicando
el contexto, los datos utilizados, la metodología y las funcionalidades disponibles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import datetime


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

PROJECT_INFO = {
    'title': 'Dashboard de Análisis de NO₂ en Madrid',
    'subtitle': 'Análisis integrado de contaminación atmosférica, tráfico y meteorología',
    'version': '2.0',
    'authors': ['Antonio Fernández'],
    'institution': 'Master Thesis - NO₂ Prediction',
    'year': '2024'
}

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

HYPOTHESIS = [
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


# ==================== FUNCIONES AUXILIARES ====================

def create_timeline_chart():
    """Crea un gráfico de línea temporal mostrando la evolución del proyecto."""
    timeline_data = [
        {'Date': '2024-01', 'Phase': 'Recopilación de datos', 'Progress': 100},
        {'Date': '2024-02', 'Phase': 'Limpieza y preprocesamiento', 'Progress': 100},
        {'Date': '2024-03', 'Phase': 'Análisis exploratorio', 'Progress': 100},
        {'Date': '2024-04', 'Phase': 'Mapeo de sensores', 'Progress': 100},
        {'Date': '2024-05', 'Phase': 'Modelos GAM', 'Progress': 100},
        {'Date': '2024-06', 'Phase': 'Modelos XGBoost', 'Progress': 100},
        {'Date': '2024-07', 'Phase': 'Dashboard interactivo', 'Progress': 100}
    ]
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.line(df, x='Date', y='Progress',
                  title='Cronología del Proyecto',
                  markers=True,
                  line_shape='spline')
    
    fig.update_layout(
        xaxis_title='Período',
        yaxis_title='Progreso (%)',
        yaxis=dict(range=[0, 110]),
        height=300
    )
    
    return fig


def create_data_summary_chart():
    """Crea un gráfico resumen de los datos disponibles."""
    data_summary = {
        'Fuente': ['Sensores NO₂', 'Sensores Tráfico', 'Meteorología'],
        'Registros': [150000, 500000, 50000],
        'Período': ['2018-2024', '2018-2024', '2018-2024'],
        'Cobertura': [85, 92, 98]
    }
    
    fig = go.Figure(data=[
        go.Bar(name='Registros (miles)', x=data_summary['Fuente'], 
               y=[x/1000 for x in data_summary['Registros']], yaxis='y', offsetgroup=1),
        go.Bar(name='Cobertura (%)', x=data_summary['Fuente'], 
               y=data_summary['Cobertura'], yaxis='y2', offsetgroup=2)
    ])

    fig.update_layout(
        title='Resumen de Datos Disponibles',
        xaxis_title='Fuente de Datos',
        yaxis=dict(title='Registros (miles)', side='left'),
        yaxis2=dict(title='Cobertura (%)', side='right', overlaying='y'),
        barmode='group',
        height=400
    )
    
    return fig


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_project_header():
    """Muestra el encabezado principal del proyecto."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; font-size: 3rem; font-weight: bold; margin-bottom: 0;">
            🌍 Dashboard de Análisis de NO₂
        </h1>
        <h2 style="color: #666; font-size: 1.5rem; margin-top: 0;">
            Madrid: Contaminación Atmosférica, Tráfico y Meteorología
        </h2>
        <div style="background: linear-gradient(90deg, #ff7f0e, #1f77b4, #2ca02c); height: 4px; margin: 1rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)


def show_project_overview():
    """Muestra la visión general del proyecto."""
    st.header("🎯 Visión General del Proyecto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Este dashboard integra **datos de calidad del aire, tráfico y meteorología** de Madrid 
        para comprender y predecir los niveles de **dióxido de nitrógeno (NO₂)**.
        
        **Objetivos principales:**
        - 📊 **Análisis exploratorio** de patrones temporales y espaciales
        - 🔗 **Correlaciones** entre tráfico, meteorología y contaminación
        - 🤖 **Modelado predictivo** con GAM y XGBoost
        - 🎯 **Nowcasting** de NO₂ para apoyo a la toma de decisiones
        
        **Período de análisis:** 2018-2024 (6 años de datos)
        """)
    
    with col2:
        st.info("""
        **💡 ¿Por qué NO₂?**
        
        - Principal contaminante del tráfico
        - Indicador de calidad del aire
        - Efectos en salud respiratoria
        - Regulado por normativa europea
        - Límite OMS: 40 μg/m³
        """)


def show_data_sources():
    """Muestra información sobre las fuentes de datos."""
    st.header("📊 Fuentes de Datos")
    
    # Gráfico resumen
    fig = create_data_summary_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalles de cada fuente
    for source, info in DATA_SOURCES.items():
        with st.expander(f"📋 Detalles: {source}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **Descripción:** {info['description']}
                
                **Granularidad:** {info['granularity']}
                
                **Fuente:** [Portal de Datos Abiertos]({info['url']})
                """)
            
            with col2:
                if source == 'Calidad del Aire':
                    st.metric("Estaciones NO₂", "24", help="Estaciones de medición activas")
                elif source == 'Tráfico':
                    st.metric("Sensores Tráfico", "4,000+", help="Sensores de intensidad distribuidos")
                else:
                    st.metric("Variables Meteo", "10", help="Variables meteorológicas principales")


def show_methodology():
    """Muestra la metodología del proyecto."""
    st.header("🔬 Metodología")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Procesamiento", "🔗 Mapeo", "📊 Variables", "🤖 Modelado"])
    
    with tab1:
        st.subheader("Procesamiento de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚦 Datos de Tráfico:**
            - Agregación de 15 min → 1 hora
            - **Intensidad:** Promedio aritmético
            - **Carga:** Promedio ponderado por intensidad
            - **Ocupación:** Promedio ponderado por intensidad
            - **Velocidad:** Promedio ponderado por intensidad
            """)
        
        with col2:
            st.markdown("""
            **🌬️ Datos de Calidad del Aire:**
            - Filtrado de valores erróneos
            - Eliminación de datos anteriores a 2018
            - Selección de estaciones con sensores de tráfico cercanos
            
            **🌡️ Datos Meteorológicos:**
            - Descarga desde ERA5 (Copernicus)
            - Conversión de unidades (K→°C, Pa→hPa, etc.)
            """)
    
    with tab2:
        st.subheader("Mapeo Espacial de Sensores")
        
        st.markdown("""
        **Criterios de asignación:**
        
        1. **Proximidad geográfica:** Sensores de tráfico a < 200m de estaciones NO₂
        2. **Disponibilidad de datos:** Selección del sensor con mayor cobertura temporal
        3. **Validación temporal:** Verificación de solapamiento en períodos de medición
        
        **Resultado:** 24 estaciones NO₂ con sensores de tráfico asignados
        """)
        
        st.info("💡 **Visualización disponible:** El módulo 'Mapeo Sensores' permite explorar estas asignaciones de forma interactiva.")
    
    with tab3:
        st.subheader("Variables del Modelo")
        
        for category, info in VARIABLES_INFO.items():
            st.markdown(f"**{category}:**")
            st.markdown(f"- *Descripción:* {info['description']}")
            st.markdown(f"- *Variables:* {', '.join(info['variables'])}")
            st.markdown(f"- *Justificación:* {info['rationale']}")
            st.markdown("")
    
    with tab4:
        st.subheader("Estrategia de Modelado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Modelos GAM (Generalized Additive Models)**
            
            - **Objetivo:** Comprensibilidad e interpretabilidad
            - **Fortalezas:** Visualización de relaciones no lineales
            - **Aplicación:** Análisis exploratorio de efectos
            - **Salidas:** Curvas de respuesta por variable
            """)
        
        with col2:
            st.markdown("""
            **⚡ Modelos XGBoost**
            
            - **Objetivo:** Precisión predictiva máxima
            - **Fortalezas:** Captura de interacciones complejas
            - **Aplicación:** Nowcasting operacional
            - **Salidas:** Predicciones + importancia de variables
            """)


def show_scientific_hypothesis():
    """Muestra las hipótesis científicas del proyecto."""
    st.header("🔬 Hipótesis Científicas")
    
    st.markdown("""
    El modelo se basa en **8 hipótesis principales** respaldadas por literatura científica:
    """)
    
    # Mostrar hipótesis en formato de cards
    for i, hypothesis in enumerate(HYPOTHESIS):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
        
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"""
                <div style="border: 2px solid #e1e1e1; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; height: 200px;">
                    <h4 style="color: #1f77b4; margin-top: 0;">{hypothesis['title']}</h4>
                    <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">{hypothesis['description']}</p>
                    <p style="font-size: 0.8rem; color: #999; margin-top: auto;"><em>📚 {hypothesis['reference']}</em></p>
                </div>
                """, unsafe_allow_html=True)


def show_dashboard_features():
    """Muestra las características del dashboard."""
    st.header("🎛️ Funcionalidades del Dashboard")
    
    features = {
        "🌍 Análisis NO₂": {
            "description": "Visualización temporal y espacial de concentraciones",
            "features": ["Mapas de calor interactivos", "Series temporales", "Filtros avanzados", "Descomposición estacional"]
        },
        "🗺️ Mapeo Sensores": {
            "description": "Asignación entre sensores de NO₂ y tráfico",
            "features": ["Mapas interactivos", "Análisis de continuidad", "Estadísticas de cobertura", "Validación espacial"]
        },
        "📊 Correlaciones": {
            "description": "Análisis de relaciones entre variables",
            "features": ["Matrices de correlación", "Gráficos de dispersión", "Series temporales comparativas", "Análisis de disponibilidad"]
        },
        "🤖 Entrenamiento GAM": {
            "description": "Modelos aditivos generalizados interpretativos",
            "features": ["Curvas de respuesta", "Análisis de residuos", "Evaluación temporal", "Interpretabilidad"]
        },
        "⚡ Entrenamiento XGBoost": {
            "description": "Modelos de gradient boosting para predicción",
            "features": ["Alta precisión predictiva", "Importancia de variables", "Validación cruzada", "Nowcasting"]
        }
    }
    
    for module, info in features.items():
        with st.expander(f"{module} - {info['description']}"):
            st.markdown("**Características principales:**")
            for feature in info['features']:
                st.markdown(f"- {feature}")


def show_getting_started():
    """Muestra la guía de inicio rápido."""
    st.header("🚀 Guía de Inicio Rápido")
    
    st.markdown("""
    **Flujo de trabajo recomendado:**
    
    1. **🌍 Análisis NO₂** → Explorar patrones temporales y espaciales
    2. **🗺️ Mapeo Sensores** → Validar asignaciones geográficas
    3. **📊 Correlaciones** → Identificar relaciones entre variables
    4. **🤖 Entrenamiento GAM** → Comprender efectos individuales
    5. **⚡ Entrenamiento XGBoost** → Desarrollar modelos predictivos
    """)
    
    st.info("""
    💡 **Tip:** Cada módulo incluye un panel de ayuda (ℹ️) con documentación detallada.
    """)
    
    # Timeline del proyecto
    fig = create_timeline_chart()
    st.plotly_chart(fig, use_container_width=True)


def show_technical_specs():
    """Muestra especificaciones técnicas."""
    st.header("⚙️ Especificaciones Técnicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Datos**
        - Período: 2018-2024
        - Granularidad: Horaria
        - Registros: ~700K
        - Variables: 20+
        """)
    
    with col2:
        st.markdown("""
        **🔧 Tecnologías**
        - Python 3.9+
        - Streamlit
        - Pandas, NumPy
        - Scikit-learn, XGBoost
        - Plotly, Matplotlib
        """)
    
    with col3:
        st.markdown("""
        **📈 Modelos**
        - GAM (pygam)
        - XGBoost
        - Validación temporal
        - Métricas: RMSE, MAE, R²
        """)


def show_project_footer():
    """Muestra el pie de página del proyecto."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p><strong>{PROJECT_INFO['title']}</strong></p>
            <p>{PROJECT_INFO['subtitle']}</p>
            <p>Versión {PROJECT_INFO['version']} | {PROJECT_INFO['year']}</p>
            <p>Desarrollado por: {', '.join(PROJECT_INFO['authors'])}</p>
            <p><em>{PROJECT_INFO['institution']}</em></p>
        </div>
        """, unsafe_allow_html=True)


# ==================== FUNCIÓN PRINCIPAL ====================

def welcome_page():
    """Función principal de la página de bienvenida."""
    
    # Contenido principal
    show_project_header()
    
    # Navegación por pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Proyecto", 
        "📊 Datos", 
        "🔬 Metodología", 
        "🎛️ Dashboard", 
        "🚀 Inicio"
    ])
    
    with tab1:
        show_project_overview()
        show_scientific_hypothesis()
    
    with tab2:
        show_data_sources()
    
    with tab3:
        show_methodology()
    
    with tab4:
        show_dashboard_features()
    
    with tab5:
        show_getting_started()
        show_technical_specs()
    
    # Pie de página
    show_project_footer()


if __name__ == "__main__":
    welcome_page() 