"""
Módulo refactorizado para análisis y visualización del mapeo entre sensores de NO2 y tráfico en Madrid.

Este módulo proporciona una interfaz limpia para visualizar las asignaciones entre
sensores de calidad del aire y sensores de tráfico, incluyendo análisis de continuidad temporal.
"""

import folium
import pandas as pd
import random
import streamlit as st
import numpy as np
import leafmap.foliumap as leafmap
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import altair as alt
from typing import List, Tuple, Dict, Optional


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

SENSOR_COLORS = [
    'blue', 'green', 'orange', 'purple', 'darkred', 'lightred', 
    'beige', 'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue'
]

MAP_CONFIG = {
    'default_zoom': 12,
    'tiles': 'OpenStreetMap',
    'no2_marker_radius': 12,
    'traffic_icon': '🚦'
}


# ==================== CLASE PRINCIPAL ====================

class SensorMappingAnalyzer:
    """Clase principal para el análisis de mapeo de sensores NO2 y tráfico."""
    
    def __init__(self):
        self.df_mapping = None
        self.df_traffic_locations = None
        self.sensor_colors = {}
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'mapping_data_loaded' not in st.session_state:
            st.session_state.mapping_data_loaded = False
        if 'selected_traffic_sensor' not in st.session_state:
            st.session_state.selected_traffic_sensor = None
    
    @st.cache_data(ttl=3600)
    def load_mapping_data(_self) -> pd.DataFrame:
        """Carga los datos de mapeo entre sensores NO2 y tráfico."""
        try:
            df = pd.read_parquet('data/super_processed/7_5_no2_with_1traffic_id.parquet')
            return df
        except Exception as e:
            st.error(f"Error al cargar datos de mapeo: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_traffic_locations(_self) -> pd.DataFrame:
        """Carga las ubicaciones de los sensores de tráfico."""
        try:
            return pd.read_excel('data/super_processed/1_all_traffic_sensors.xlsx')
        except Exception as e:
            st.error(f"Error al cargar ubicaciones de tráfico: {str(e)}")
            return pd.DataFrame()
    
    def prepare_mapping_data(self) -> pd.DataFrame:
        """Prepara y procesa los datos de mapeo."""
        # Cargar datos
        df = self.load_mapping_data()
        df_traffic = self.load_traffic_locations()
        df_traffic.rename(columns = {'latitud':'latitud_trafico', 'longitud':'longitud_trafico'}, inplace = True)

        
        if df.empty or df_traffic.empty:
            return pd.DataFrame()
        
        # Eliminar duplicados y preparar datos
        df_unique = df.drop_duplicates(subset=['id_no2']).copy()
        
        # Asegurar tipos de datos consistentes
        df_traffic['id_trafico'] = df_traffic['id_trafico'].astype(str)
        df_unique['id_trafico'] = df_unique['id_trafico'].astype(str)
        
        # Merge con ubicaciones de tráfico
        df_merged = df_unique.merge(
            df_traffic, 
            on='id_trafico', 
            how='left',
            suffixes=('_no2', '_trafico')
        )
        
        return df_merged
    
    def assign_colors_to_sensors(self, df: pd.DataFrame) -> None:
        """Asigna colores únicos a cada sensor NO2."""
        unique_no2_sensors = df['id_no2'].unique()
        
        for sensor in unique_no2_sensors:
            if sensor not in self.sensor_colors:
                self.sensor_colors[sensor] = random.choice(SENSOR_COLORS)
    
    def create_sensor_mapping_map(self, df: pd.DataFrame) -> folium.Map:
        """Crea un mapa con el mapeo de sensores NO2 y tráfico."""
        if df.empty:
            return None
                        
        # Centrar mapa
        map_center = [df['latitud_no2'].mean(), df['longitud_no2'].mean()]
        
        # Crear mapa
        m = folium.Map(
            location=map_center,
            zoom_start=MAP_CONFIG['default_zoom'],
            tiles=MAP_CONFIG['tiles']
        )
        
        # Asignar colores
        self.assign_colors_to_sensors(df)
        
        # Añadir marcadores de NO2
        for _, row in df.iterrows():
            color = self.sensor_colors.get(row['id_no2'], 'blue')
            
            folium.CircleMarker(
                location=[row['latitud_no2'], row['longitud_no2']],
                radius=MAP_CONFIG['no2_marker_radius'],
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"<b>Sensor NO2</b><br>ID: {row['id_no2']}<br>Lat: {row['latitud_no2']:.4f}<br>Lon: {row['longitud_no2']:.4f}",
                tooltip=f"NO2 Sensor: {row['id_no2']}"
            ).add_to(m)
        
        # Añadir marcadores de tráfico
        for _, row in df.iterrows():
            if pd.notna(row['latitud_trafico']) and pd.notna(row['longitud_trafico']):
                folium.Marker(
                    location=[row['latitud_trafico'], row['longitud_trafico']],
                    popup=f"<b>Sensor Tráfico</b><br>ID: {row['id_trafico']}<br>Asignado a NO2: {row['id_no2']}<br>Lat: {row['latitud_trafico']:.4f}<br>Lon: {row['longitud_trafico']:.4f}",
                    tooltip=f"Tráfico: {row['id_trafico']} → NO2: {row['id_no2']}",
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12pt; text-align: center;">{MAP_CONFIG["traffic_icon"]}</div>'
                    )
                ).add_to(m)
        
        return m
    
    def get_traffic_sensors_list(self, df: pd.DataFrame) -> List[str]:
        """Obtiene la lista de sensores de tráfico disponibles."""
        return sorted(df['id_trafico'].dropna().unique().tolist())
    
    def analyze_data_continuity(self, traffic_sensor_id: str) -> Optional[pd.DataFrame]:
        """Analiza la continuidad de datos para un sensor de tráfico específico."""
        df = self.load_mapping_data()
        
        if df.empty:
            return None
        
        # Filtrar por sensor de tráfico
        df_sensor = df[df['id_trafico'] == str(traffic_sensor_id)].copy()
        
        if df_sensor.empty:
            return None
        
        # Procesar fechas
        df_sensor['fecha'] = pd.to_datetime(df_sensor['fecha'])
        
        # Generar rango completo de fechas
        fecha_inicio = df_sensor['fecha'].min()
        fecha_fin = df_sensor['fecha'].max()
        
        if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
            return None
        
        # Crear DataFrame con rango completo
        rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='H')
        df_continuity = pd.DataFrame({
            'fecha': rango_fechas,
            'dato_presente': 0
        })
        
        # Marcar datos presentes
        df_continuity.loc[df_continuity['fecha'].isin(df_sensor['fecha']), 'dato_presente'] = 1
        
        # Añadir estadísticas
        total_hours = len(df_continuity)
        present_hours = df_continuity['dato_presente'].sum()
        coverage_percentage = (present_hours / total_hours) * 100
        
        df_continuity['cobertura_total'] = coverage_percentage
        df_continuity['horas_totales'] = total_hours
        df_continuity['horas_con_datos'] = present_hours
        
        return df_continuity
    
    def calculate_mapping_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcula estadísticas del mapeo de sensores."""
        if df.empty:
            return {}
        
        stats = {
            'total_no2_sensors': df['id_no2'].nunique(),
            'total_traffic_sensors': df['id_trafico'].nunique(),
            'mapped_pairs': len(df),
            'avg_distance': df.get('distancia_metros', pd.Series()).mean() if 'distancia_metros' in df.columns else None,
            'date_range': {
                'start': df['fecha'].min() if 'fecha' in df.columns else None,
                'end': df['fecha'].max() if 'fecha' in df.columns else None
            }
        }
        
        return stats


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_mapping_info():
    """Muestra información sobre el mapeo de sensores."""
    with st.expander("ℹ️ Acerca del mapeo de sensores", expanded=False):
        st.markdown("""
        **Mapeo de Sensores NO₂ y Tráfico**
        
        Esta sección muestra la asignación entre sensores de calidad del aire (NO₂) y sensores de tráfico en Madrid.
        
        **Funcionalidades:**
        - Visualización de ubicaciones de sensores en el mapa
        - Análisis de continuidad temporal de datos
        - Estadísticas de cobertura y disponibilidad
        - Filtros por sensor y período temporal
        
        **Elementos del mapa:**
        - 🔵 **Círculos coloreados**: Sensores de NO₂
        - 🚦 **Iconos de tráfico**: Sensores de tráfico asignados
        - **Líneas**: Conexiones entre sensores emparejados
        """)


def show_continuity_chart(df_continuity: pd.DataFrame, sensor_id: str):
    """Muestra gráfico de continuidad de datos."""
    if df_continuity is None or df_continuity.empty:
        st.warning("No hay datos de continuidad disponibles para este sensor.")
        return
    
    # Estadísticas generales
    coverage = df_continuity['cobertura_total'].iloc[0]
    total_hours = df_continuity['horas_totales'].iloc[0]
    present_hours = df_continuity['horas_con_datos'].iloc[0]
    
    # Mostrar estadísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cobertura de datos", f"{coverage:.1f}%")
    with col2:
        st.metric("Horas con datos", f"{present_hours:,}")
    with col3:
        st.metric("Horas totales", f"{total_hours:,}")
    
    # Gráfico de continuidad
    chart = alt.Chart(df_continuity).mark_line(point=True).encode(
        x=alt.X('fecha:T', title='Fecha y Hora'),
        y=alt.Y('dato_presente:Q', 
                scale=alt.Scale(domain=[0, 1]),
                title='Datos disponibles (1: Sí, 0: No)'),
        tooltip=[
            alt.Tooltip('fecha:T', title='Fecha'),
            alt.Tooltip('dato_presente:Q', title='Dato presente')
        ]
    ).properties(
        title=f'Continuidad de datos - Sensor de Tráfico {sensor_id}',
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)


def show_mapping_statistics(stats: Dict):
    """Muestra estadísticas del mapeo."""
    if not stats:
        return
    
    st.header("📊 Estadísticas del Mapeo")
    st.write("Para cada sensor de NO2, se han seleccionado todos los sensores de tráfico que están a menos de 200 metros de distancia y elegido solo uno, que tiene más datos")
        
    st.metric("Sensores NO₂", stats.get('total_no2_sensors', 'N/A'))
    st.metric("Sensores Tráfico", stats.get('total_traffic_sensors', 'N/A'))
    st.metric("Pares Mapeados", stats.get('mapped_pairs', 'N/A'))


# ==================== FUNCIÓN PRINCIPAL ====================

def generar_mapa_asignaciones():
    """Función principal para la visualización del mapeo de sensores."""
    
    # Inicializar analizador
    analyzer = SensorMappingAnalyzer()
    
    # Panel de información
    show_mapping_info()
    
    # Configuración en sidebar
    st.sidebar.header("⚙️ Configuración de Mapeo")
    
    # Cargar datos automáticamente
    if not st.session_state.mapping_data_loaded:
        with st.spinner("Cargando datos de mapeo..."):
            analyzer.df_mapping = analyzer.load_mapping_data()
            analyzer.df_traffic_locations = analyzer.load_traffic_locations()
            
            if not analyzer.df_mapping.empty:
                st.session_state.mapping_data_loaded = True
                st.success("Datos de mapeo cargados correctamente!")
            else:
                st.error("Error al cargar los datos de mapeo.")
                return
    
    # Preparar datos procesados
    df_processed = analyzer.prepare_mapping_data()
    
    if df_processed.empty:
        st.error("No hay datos disponibles para mostrar el mapeo.")
        return
    
    # Filtros en sidebar
    show_statistics = st.sidebar.checkbox("Mostrar estadísticas", value=True)
    show_continuity = st.sidebar.checkbox("Mostrar análisis de continuidad", value=True)
    
    # Interfaz principal
    col_map, col_info = st.columns([3, 1])
    
    with col_map:
        st.header("🗺️ Mapa de Asignaciones")
        
        # Crear y mostrar mapa
        mapa = analyzer.create_sensor_mapping_map(df_processed)
        if mapa:
            folium_static(mapa, height=500)
        else:
            st.error("No se pudo generar el mapa.")
    
    with col_info:
        
        # Mostrar estadísticas si está habilitado
        if show_statistics:
            stats = analyzer.calculate_mapping_statistics(df_processed)
            show_mapping_statistics(stats)
    
    # Análisis de continuidad
    if show_continuity:
        st.subheader("📊 Análisis de Continuidad Temporal")
        
        # Obtener lista de sensores de tráfico
        traffic_sensors = analyzer.get_traffic_sensors_list(df_processed)
        
        if traffic_sensors:
            # Selector de sensor con slider
            col_selector, col_button = st.columns([3, 1])
            
            with col_selector:
                selected_sensor_index = st.slider(
                    "Selecciona sensor de tráfico",
                    min_value=0,
                    max_value=len(traffic_sensors) - 1,
                    value=0,
                    format=f"Sensor: {traffic_sensors[0] if traffic_sensors else 'N/A'}"
                )
                
                selected_sensor = traffic_sensors[selected_sensor_index]
                st.write(f"**Sensor seleccionado:** {selected_sensor}")
            
            #with col_button:
            if st.button("Analizar Continuidad", type="primary"):
                with st.spinner("Analizando continuidad de datos..."):
                    df_continuity = analyzer.analyze_data_continuity(selected_sensor)
                    show_continuity_chart(df_continuity, selected_sensor)
        else:
            st.info("No hay sensores de tráfico disponibles para análisis.")
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Mapeo de sensores basado en proximidad geográfica | Datos del Ayuntamiento de Madrid
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    generar_mapa_asignaciones() 