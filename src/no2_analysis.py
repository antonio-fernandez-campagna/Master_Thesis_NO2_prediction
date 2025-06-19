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
import time


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

# Constantes para normalización del mapa
GLOBAL_MIN_NO2 = 0
GLOBAL_MAX_NO2 = 200
WHO_LIMIT_NO2 = 40


# ==================== CLASE PRINCIPAL ====================

class NO2Analyzer:
    """Clase principal para el análisis de datos de NO2."""
    
    def __init__(self):
        self.df_original = None
        self.df_filtered = None
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
        
        # Preparar datos para heatmap con rango fijo
        heat_data = []
        markers_above_limit = []
        
        for _, row in df.iterrows():
            # Normalizar usando rango fijo de 0-200
            normalized_value = max(0.1, min(1.0, 
                (row['no2_value'] - GLOBAL_MIN_NO2) / 
                (GLOBAL_MAX_NO2 - GLOBAL_MIN_NO2)
            ))
            
            # Hacer la normalización más sensible para valores bajos
            # Esto hace que valores por debajo de 40 sean más visibles
            if row['no2_value'] <= WHO_LIMIT_NO2:
                # Para valores <= 40, usar una escala más comprimida (0.1 - 0.4)
                normalized_value = 0.1 + (row['no2_value'] / WHO_LIMIT_NO2) * 0.3
            else:
                # Para valores > 40, usar el resto de la escala (0.4 - 1.0)
                excess_value = row['no2_value'] - WHO_LIMIT_NO2
                max_excess = GLOBAL_MAX_NO2 - WHO_LIMIT_NO2
                normalized_value = 0.4 + (excess_value / max_excess) * 0.6
                
                # Añadir marcador para sensores que superan el límite OMS
                markers_above_limit.append({
                    'lat': row['latitud'],
                    'lon': row['longitud'],
                    'value': row['no2_value'],
                    'id': row.get('id_no2', 'N/A')
                })
            
            heat_data.append([row['latitud'], row['longitud'], normalized_value])
        
        # Configurar parámetros del heatmap
        radius = 20 if len(df) > 100 else 30
        blur = 15 if len(df) > 100 else 20
        
        # Añadir heatmap (sin gradiente personalizado para evitar errores)
        m.add_heatmap(
            data=heat_data,
            name="NO2 Heatmap",
            radius=radius,
            blur=blur
        )
        
        # Añadir marcadores para sensores que superan el límite OMS
        if markers_above_limit:
            for marker in markers_above_limit:
                # Crear popup con información del sensor
                popup_text = f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4 style="color: red; margin: 0;">⚠️ Límite OMS superado</h4>
                    <p><strong>Sensor ID:</strong> {marker['id']}</p>
                    <p><strong>NO₂:</strong> {marker['value']:.1f} μg/m³</p>
                    <p><strong>Límite OMS:</strong> {WHO_LIMIT_NO2} μg/m³</p>
                    <p style="color: red;"><strong>Exceso:</strong> {marker['value'] - WHO_LIMIT_NO2:.1f} μg/m³</p>
                </div>
                """
                
                folium.Marker(
                    location=[marker['lat'], marker['lon']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(
                        color='red', 
                        icon='exclamation-triangle',
                        prefix='fa'
                    )
                ).add_to(m)
        
        # Añadir leyenda personalizada
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 220px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">Niveles de NO₂</h4>
        <div style="margin: 5px 0;">
            <span style="color: #0066FF; font-size: 16px;">●</span> 0-{WHO_LIMIT_NO2} μg/m³ (Límite OMS)
        </div>
        <div style="margin: 5px 0;">
            <span style="color: #FFAA00; font-size: 16px;">●</span> {WHO_LIMIT_NO2}-100 μg/m³ (Moderado)
        </div>
        <div style="margin: 5px 0;">
            <span style="color: #FF0000; font-size: 16px;">●</span> >100 μg/m³ (Alto)
        </div>
        <div style="margin: 8px 0 0 0; padding-top: 5px; border-top: 1px solid #ccc; font-size: 12px; color: #666;">
            🔴 Marcador = Supera límite OMS
        </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
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
    <div style="margin: 1rem 0;">
        <div style="text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 0.5rem; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
            <div style="font-size: 1.5rem; color: {stats['color']};">{stats['mean']:.1f} μg/m³</div>
        </div>
        <div style="text-align: center; padding: 1rem; background-color: #f0f0f0; border-radius: 0.5rem; margin-bottom: 0.5rem;">
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


def show_temporal_evolution_map(analyzer, df_processed: pd.DataFrame, granularity: str):
    """Muestra la evolución temporal de NO2 en el mapa con animación automática."""
    
    # Obtener grupos temporales ordenados
    time_groups = sorted(df_processed['time_group'].unique())
    format_str = GRANULARITY_CONFIG[granularity]['format']
    
    if len(time_groups) < 2:
        st.warning("Se necesitan al menos 2 períodos temporales para mostrar la evolución.")
        return
    
    st.subheader("🎬 Evolución temporal animada")
    
    # Controles de animación
    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
    col5, _, _, _ = st.columns([7, 1, 1, 1])

    
    with col1:
        if st.button("▶️ Reproducir", key="play_btn"):
            st.session_state.is_playing = True
            st.session_state.current_index = 0
    
    with col2:
        if st.button("⏸️ Pausar", key="pause_btn"):
            st.session_state.is_playing = False
    
    with col3:
        if st.button("⏹️ Reiniciar", key="stop_btn"):
            st.session_state.is_playing = False
            st.session_state.current_index = 0
    
    with col5:
        velocidad = st.select_slider(
            "Velocidad",
            options=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            value=1.0,
            format_func=lambda x: f"{x}x"
        )
    
    # Inicializar estado de animación
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Contenedor para el mapa
    map_container = st.empty()
    info_container = st.empty()
    
    # Lógica de animación
    if st.session_state.is_playing:
        # Auto-avanzar
        if st.session_state.current_index < len(time_groups):
            current_time = time_groups[st.session_state.current_index]
            
            # Mostrar mapa actual
            df_current = df_processed[df_processed['time_group'] == current_time]
            mapa = analyzer.create_heatmap(df_current)
            
            with map_container.container():
                if mapa:
                    folium_static(mapa, height=500)
                else:
                    st.info("No hay datos para este período.")
            
            # Mostrar información temporal
            with info_container.container():
                progress = (st.session_state.current_index + 1) / len(time_groups)
                st.progress(progress)
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Período actual", current_time.strftime(format_str))
                with col_info2:
                    st.metric("Progreso", f"{st.session_state.current_index + 1}/{len(time_groups)}")
                with col_info3:
                    if not df_current.empty:
                        avg_no2 = df_current['no2_value'].mean()
                        st.metric("NO₂ promedio", f"{avg_no2:.1f} μg/m³")
            
            # Avanzar al siguiente frame
            time.sleep(1.0 / velocidad)
            st.session_state.current_index += 1
            
            if st.session_state.current_index >= len(time_groups):
                st.session_state.is_playing = False
                st.session_state.current_index = 0
                st.success("¡Animación completada!")
            
            st.rerun()
    
    else:
        # Modo manual - mostrar frame actual
        current_time = time_groups[st.session_state.current_index]
        df_current = df_processed[df_processed['time_group'] == current_time]
        
        with map_container.container():
            mapa = analyzer.create_heatmap(df_current)
            if mapa:
                folium_static(mapa, height=500)
            else:
                st.info("No hay datos para este período.")
        
        with info_container.container():
            progress = (st.session_state.current_index + 1) / len(time_groups)
            st.progress(progress)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Período actual", current_time.strftime(format_str))
            with col_info2:
                # Controles manuales
                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.button("⬅️ Anterior", disabled=st.session_state.current_index == 0):
                        st.session_state.current_index = max(0, st.session_state.current_index - 1)
                        st.rerun()
                with col_next:
                    if st.button("➡️ Siguiente", disabled=st.session_state.current_index >= len(time_groups) - 1):
                        st.session_state.current_index = min(len(time_groups) - 1, st.session_state.current_index + 1)
                        st.rerun()
            with col_info3:
                if not df_current.empty:
                    avg_no2 = df_current['no2_value'].mean()
                    st.metric("NO₂ promedio", f"{avg_no2:.1f} μg/m³")
    
    # Información adicional
    st.info(f"💡 **Tip**: Usa los controles para navegar por {len(time_groups)} períodos temporales de {granularity.lower()}")


# ==================== FUNCIÓN PRINCIPAL ====================

def generar_analisis_no2():
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
                    st.session_state.data_loaded = True
                    st.success("Datos cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    analyzer.df_original = analyzer.load_data()
    
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
        
        if df_processed.empty:
            st.error("No hay suficientes datos para la granularidad seleccionada.")
            return
    
    # Mapa de evolución temporal
    st.header("🗺️ Evolución temporal de concentraciones")
    show_temporal_evolution_map(analyzer, df_processed, granularity)
    
    # Estadísticas globales del período
    st.header("📊 Resumen del período seleccionado")
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    stats_global = analyzer.calculate_statistics(df_processed)
    
    with col_stats1:
        st.metric("NO₂ promedio", f"{stats_global['mean']:.1f} μg/m³")
    with col_stats2:
        st.metric("NO₂ máximo", f"{stats_global['max']:.1f} μg/m³")
    with col_stats3:
        st.metric("NO₂ mínimo", f"{stats_global['min']:.1f} μg/m³")
    with col_stats4:
        st.metric("Total mediciones", f"{stats_global['count']:,}")
    
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


if __name__ == "__main__":
    generar_analisis_no2() 