"""
Módulo refactorizado para análisis y visualización de datos de contaminación por NO2 en Madrid.

Este módulo proporciona una interfaz limpia y optimizada para analizar datos de 
contaminación por NO2, incluyendo mapas de calor y análisis temporal.
"""

import folium
import pandas as pd
import streamlit as st
import numpy as np
import leafmap.foliumap as leafmap
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import altair as alt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

POLLUTION_LEVELS = {
    'Bajo': {'threshold': 40, 'color': 'green'},
    'Medio': {'threshold': 100, 'color': 'orange'},
    'Alto': {'threshold': float('inf'), 'color': 'red'}
}

GRANULARITY_CONFIG = {
    'Horaria': {'freq': 'H', 'format': '%Y-%m-%d %H:%M', 'period': 24},
    'Diaria': {'freq': 'D', 'format': '%Y-%m-%d', 'period': 365},
    'Semanal': {'freq': 'W', 'format': '%Y-%m-%d', 'period': 52},
    'Mensual': {'freq': 'M', 'format': '%Y-%m', 'period': 12},
    'Anual': {'freq': 'Y', 'format': '%Y', 'period': 1}
}


# ==================== CLASE PRINCIPAL ====================

class NO2Analyzer:
    """Clase principal para el análisis de datos de NO2."""
    
    def __init__(self):
        self.df_original = None
        self.df_filtered = None
        self.global_min = None
        self.global_max = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'config' not in st.session_state:
            st.session_state.config = {}
    
    @st.cache_data(ttl=3600)
    def load_data(_self) -> pd.DataFrame:
        """Carga y preprocesa los datos de contaminación por NO2."""
        try:
            df = pd.read_csv('data/super_processed/6_df_air_data_and_locations_reduced.csv')
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
            return pd.DataFrame()
    
    def filter_data(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Aplica todos los filtros a los datos."""
        # Filtro por sensor
        if config.get('sensor') != 'Todos':
            df = df[df['id_no2'] == config['sensor']]
        
        # Filtro por fechas
        df = df[
            (df['fecha'].dt.date >= config['fecha_inicio']) & 
            (df['fecha'].dt.date <= config['fecha_fin'])
        ]
        
        # Filtro por nivel de contaminación
        if config.get('nivel'):
            if config['nivel'] == 'Bajo':
                df = df[df['no2_value'] <= 40]
            elif config['nivel'] == 'Medio':
                df = df[(df['no2_value'] > 40) & (df['no2_value'] <= 100)]
            elif config['nivel'] == 'Alto':
                df = df[df['no2_value'] > 100]
        
        # Filtro de outliers
        if config.get('filtrar_outliers', False):
            df = self._remove_outliers(df)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina valores extremos del DataFrame."""
        q1 = df['no2_value'].quantile(0.01)
        q3 = df['no2_value'].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df['no2_value'] >= lower_bound) & (df['no2_value'] <= upper_bound)]
    
    def apply_temporal_granularity(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Aplica granularidad temporal a los datos."""
        config = GRANULARITY_CONFIG[granularity]
        
        if granularity == 'Horaria':
            df['time_group'] = df['fecha'].dt.floor('H')
        elif granularity == 'Diaria':
            df['time_group'] = df['fecha'].dt.floor('D')
        elif granularity == 'Semanal':
            df['time_group'] = df['fecha'].dt.to_period('W').dt.to_timestamp()
        elif granularity == 'Mensual':
            df['time_group'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        else:  # Anual
            df['time_group'] = df['fecha'].dt.to_period('Y').dt.to_timestamp()
        
        # Agregar datos si no es horario
        if granularity != 'Horaria':
            df = df.groupby(['time_group', 'latitud', 'longitud']).agg({
                'no2_value': 'mean',
                'fecha': 'min',
                'id_no2': 'first'
            }).reset_index()
        
        return df
    
    def create_heatmap(self, df: pd.DataFrame) -> Optional[leafmap.Map]:
        """Crea un mapa de calor con los datos de NO2."""
        if df.empty:
            return None
        
        # Limitar puntos para rendimiento
        if len(df) > 2000:
            df = df.sample(2000)
        
        # Centrar mapa
        center = [df['latitud'].mean(), df['longitud'].mean()]
        
        # Crear mapa
        m = leafmap.Map(
            center=center,
            zoom=12,
            tiles="CartoDB positron",
            draw_control=False,
            measure_control=False,
            fullscreen_control=True
        )
        
        # Preparar datos para heatmap
        heat_data = []
        for _, row in df.iterrows():
            normalized_value = max(0.1, min(1, 
                (row['no2_value'] - self.global_min) / 
                (self.global_max - self.global_min) * 0.8 + 0.2
            ))
            heat_data.append([row['latitud'], row['longitud'], normalized_value])
        
        # Configurar parámetros
        radius = 15 if len(df) > 100 else 25
        blur = 10 if len(df) > 100 else 15
        
        # Añadir heatmap
        m.add_heatmap(
            data=heat_data,
            name="NO2 Heatmap",
            radius=radius,
            blur=blur
        )
        
        return m
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcula estadísticas básicas de los datos."""
        if df.empty:
            return {}
        
        stats = {
            'mean': df['no2_value'].mean(),
            'max': df['no2_value'].max(),
            'min': df['no2_value'].min(),
            'median': df['no2_value'].median(),
            'count': len(df)
        }
        
        # Determinar nivel de contaminación
        if stats['mean'] <= 40:
            stats['level'] = 'Bajo'
            stats['color'] = 'green'
        elif stats['mean'] <= 100:
            stats['level'] = 'Medio'
            stats['color'] = 'orange'
        else:
            stats['level'] = 'Alto'
            stats['color'] = 'red'
        
        return stats
    
    def generate_temporal_stats(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Genera estadísticas temporales para gráficos."""
        format_str = GRANULARITY_CONFIG[granularity]['format']
        
        stats_df = df.groupby('time_group').agg({
            'no2_value': ['mean', 'max', 'count']
        }).reset_index()
        
        stats_df.columns = ['time_group', 'no2_promedio', 'no2_max', 'num_readings']
        stats_df['fecha_str'] = stats_df['time_group'].dt.strftime(format_str)
        
        return stats_df


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_basic_stats(stats: Dict):
    """Muestra estadísticas básicas en formato visual."""
    if not stats:
        return
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
        <div style="text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 0.5rem;">
            <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
            <div style="font-size: 1.5rem; color: {stats['color']};">{stats['mean']:.1f} μg/m³</div>
        </div>
        <div style="text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 0.5rem;">
            <div style="font-size: 0.8rem; color: #666;">Máximo NO₂</div>
            <div style="font-size: 1.5rem; color: red;">{stats['max']:.1f} μg/m³</div>
        </div>
        <div style="text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 0.5rem;">
            <div style="font-size: 0.8rem; color: #666;">Nivel</div>
            <div style="font-size: 1.5rem; color: {stats['color']};">{stats['level']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_temporal_evolution(stats_df: pd.DataFrame, granularity: str):
    """Muestra gráfico de evolución temporal."""
    st.write("**Evolución temporal de NO₂**")
    st.write("La OMS recomienda que los niveles medios anuales de NO₂ no superen los 40 μg/m³ (línea roja).")
    
    format_str = GRANULARITY_CONFIG[granularity]['format']
    
    # Gráfico de línea
    line_chart = alt.Chart(stats_df).mark_line(point=True).encode(
        x=alt.X('time_group:T', title='Fecha', axis=alt.Axis(format=format_str)),
        y=alt.Y('no2_promedio:Q', title='NO₂ promedio (μg/m³)'),
        tooltip=[
            alt.Tooltip('fecha_str:N', title='Fecha'),
            alt.Tooltip('no2_promedio:Q', title='NO₂ promedio', format='.1f'),
            alt.Tooltip('no2_max:Q', title='NO₂ máximo', format='.1f'),
            alt.Tooltip('num_readings:Q', title='Nº de mediciones')
        ]
    ).properties(height=300)
    
    # Línea de límite OMS
    limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(
        color='red', strokeDash=[3, 3]
    ).encode(y='y:Q')
    
    st.altair_chart(line_chart + limit_line, use_container_width=True)


def show_histogram(df: pd.DataFrame):
    """Muestra histograma de distribución de NO2."""
    st.write("**Distribución de valores de NO₂**")
    
    if df.empty:
        st.warning("No hay datos disponibles para generar el histograma.")
        return
    
    # Añadir categoría de nivel
    df_with_level = df.copy()
    df_with_level['nivel'] = pd.cut(
        df_with_level['no2_value'], 
        bins=[0, 40, 100, float('inf')], 
        labels=['Bajo', 'Medio', 'Alto'],
        include_lowest=True
    )
    
    # Crear histograma
    hist = alt.Chart(df_with_level).mark_bar().encode(
        x=alt.X('no2_value:Q', bin=alt.Bin(step=5), title='Concentración de NO₂ (μg/m³)'),
        y=alt.Y('count():Q', title='Número de mediciones'),
        color=alt.Color('nivel:N', 
                       scale=alt.Scale(domain=['Bajo', 'Medio', 'Alto'], 
                                     range=['green', 'orange', 'red']),
                       legend=alt.Legend(title="Nivel de contaminación"))
    ).properties(height=300)
    
    # Líneas de referencia
    lines = alt.Chart(pd.DataFrame({'x': [40, 100]})).mark_rule(
        color='black', strokeDash=[3, 3]
    ).encode(x='x:Q')
    
    st.altair_chart(hist + lines, use_container_width=True)


def show_seasonal_decomposition(stats_df: pd.DataFrame, granularity: str):
    """Muestra descomposición estacional de la serie temporal."""
    st.markdown("### 📊 Descomposición de la serie temporal de NO₂")
    
    try:
        # Preparar datos
        df_decompose = stats_df.set_index('time_group')
        period = GRANULARITY_CONFIG[granularity]['period']
        
        # Verificar si hay suficientes datos
        if len(df_decompose) < 2 * period:
            st.warning("No hay suficientes datos para realizar la descomposición estacional.")
            return
        
        # Aplicar descomposición
        result = seasonal_decompose(df_decompose['no2_promedio'], model='additive', period=period)
        
        # Crear gráficos
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=axes[0], title="Serie Original", color="black")
        result.trend.plot(ax=axes[1], title="Tendencia", color="blue")
        result.seasonal.plot(ax=axes[2], title="Estacionalidad", color="green")
        result.resid.plot(ax=axes[3], title="Ruido (Residuos)", color="red")
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error en la descomposición: {str(e)}")
        st.info("Intenta con un rango de fechas más amplio o una granularidad diferente.")


def show_info_panel():
    """Muestra panel de información sobre el dashboard."""
    with st.expander("ℹ️ Acerca de este dashboard", expanded=False):
        st.markdown("""
        **Dashboard de Análisis de NO₂ en Madrid**
        
        Este dashboard permite analizar la evolución temporal de los niveles de NO₂ en Madrid.
        
        **Funcionalidades:**
        - Visualización de mapas de calor de concentraciones de NO₂
        - Análisis temporal con diferentes granularidades
        - Filtros por sensor, fechas y niveles de contaminación
        - Estadísticas descriptivas y gráficos de evolución
        - Descomposición estacional de series temporales
        
        **Niveles de referencia:**
        - **Bajo**: ≤ 40 μg/m³ (límite recomendado por la OMS)
        - **Medio**: 41-100 μg/m³
        - **Alto**: > 100 μg/m³
        """)


# ==================== FUNCIÓN PRINCIPAL ====================

def generate_analisis_no2():
    """Función principal de la aplicación."""
    
    # Inicializar analizador
    analyzer = NO2Analyzer()
    
    # Panel de información
    show_info_panel()
    
    # Cargar datos
    if not st.session_state.data_loaded:
        if st.button("Cargar datos de NO₂", type="primary"):
            with st.spinner("Cargando datos..."):
                analyzer.df_original = analyzer.load_data()
                if not analyzer.df_original.empty:
                    analyzer.global_min = analyzer.df_original['no2_value'].min()
                    analyzer.global_max = analyzer.df_original['no2_value'].max()
                    st.session_state.data_loaded = True
                    st.success("Datos cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    analyzer.df_original = analyzer.load_data()
    analyzer.global_min = analyzer.df_original['no2_value'].min()
    analyzer.global_max = analyzer.df_original['no2_value'].max()
    
    # Configuración de filtros
    st.sidebar.header("⚙️ Configuración")
    
    # Selector de sensor
    sensores = ["Todos"] + sorted(analyzer.df_original['id_no2'].unique())
    sensor_seleccionado = st.sidebar.selectbox("Sensor de NO₂", sensores)
    
    # Filtros de fecha
    fecha_min = analyzer.df_original['fecha'].min().date()
    fecha_max = analyzer.df_original['fecha'].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fecha_inicio = st.date_input("Fecha inicio", fecha_min, min_value=fecha_min, max_value=fecha_max)
    with col2:
        fecha_fin = st.date_input("Fecha fin", fecha_max, min_value=fecha_min, max_value=fecha_max)
    
    # Granularidad temporal
    granularity = st.sidebar.selectbox(
        "Granularidad temporal", 
        list(GRANULARITY_CONFIG.keys()),
        index=3  # Mensual por defecto
    )
    
    # Filtros adicionales
    nivel_contaminacion = st.sidebar.selectbox(
        "Nivel de contaminación", 
        ["Todos", "Bajo", "Medio", "Alto"]
    )
    
    filtrar_outliers = st.sidebar.checkbox(
        "Filtrar valores extremos", 
        help="Elimina el 2% de valores más extremos"
    )
    
    # Validar fechas
    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        return
    
    # Configurar filtros
    config = {
        'sensor': sensor_seleccionado,
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin,
        'granularity': granularity,
        'nivel': nivel_contaminacion if nivel_contaminacion != "Todos" else None,
        'filtrar_outliers': filtrar_outliers
    }
    
    # Procesar datos
    with st.spinner("Procesando datos..."):
        # Aplicar filtros
        df_filtered = analyzer.filter_data(analyzer.df_original, config)
        
        if df_filtered.empty:
            st.error("No hay datos disponibles para los filtros seleccionados.")
            return
        
        # Aplicar granularidad temporal
        df_processed = analyzer.apply_temporal_granularity(df_filtered, granularity)
        
        # Obtener grupos temporales
        time_groups = sorted(df_processed['time_group'].unique())
        
        if not time_groups:
            st.error("No hay suficientes datos para la granularidad seleccionada.")
            return
    
    # Interfaz principal
    col_map, col_stats = st.columns([3, 1])
    
    with col_map:
        st.header("🗺️ Mapa de concentraciones")
        
        # Selector de tiempo
        format_str = GRANULARITY_CONFIG[granularity]['format']
        selected_time = st.select_slider(
            "Selecciona el momento temporal",
            options=time_groups,
            format_func=lambda x: x.strftime(format_str),
            value=time_groups[0]
        )
        
        # Filtrar datos para el tiempo seleccionado
        df_time_filtered = df_processed[df_processed['time_group'] == selected_time]
        
        # Crear y mostrar mapa
        if not df_time_filtered.empty:
            mapa = analyzer.create_heatmap(df_time_filtered)
            if mapa:
                folium_static(mapa, height=500)
            else:
                st.info("No hay datos suficientes para generar el mapa.")
        else:
            st.info("No hay datos para el momento seleccionado.")
    
    with col_stats:
        st.header("📊 Estadísticas")
        if not df_time_filtered.empty:
            stats = analyzer.calculate_statistics(df_time_filtered)
            show_basic_stats(stats)
        else:
            st.info("No hay datos para mostrar estadísticas.")
    
    # Gráficos adicionales
    if not df_processed.empty:
        st.header("📈 Análisis temporal")
        
        # Generar estadísticas temporales
        stats_df = analyzer.generate_temporal_stats(df_processed, granularity)
        
        # Gráfico de evolución
        show_temporal_evolution(stats_df, granularity)
        
        # Histograma
        show_histogram(df_processed)
        
        # Descomposición estacional
        show_seasonal_decomposition(stats_df, granularity)
    
    # Pie de página
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Datos del Ayuntamiento de Madrid | Última actualización: {fecha_max.strftime('%d/%m/%Y')}
    </div>
    """, unsafe_allow_html=True)

