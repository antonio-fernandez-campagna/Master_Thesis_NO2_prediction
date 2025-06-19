"""
Módulo refactorizado para análisis de relación entre sensores de NO2 y tráfico en Madrid.

Este módulo proporciona una interfaz integrada para analizar correlaciones entre
datos de contaminación por NO2, tráfico y variables meteorológicas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta
import seaborn as sns
import altair as alt
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


# ==================== CONFIGURACIÓN Y CONSTANTES ====================

GRANULARITY_CONFIG = {
    'Horaria': {'freq': 'H', 'format': '%Y-%m-%d %H:%M'},
    'Diaria': {'freq': 'D', 'format': '%Y-%m-%d'},
    'Semanal': {'freq': 'W', 'format': '%Y-%m-%d'},
    'Mensual': {'freq': 'M', 'format': '%Y-%m'}
}

VARIABLE_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']

TRAFFIC_VARIABLES = ['intensidad', 'carga', 'ocupacion']
METEO_VARIABLES = ['d2m', 't2m', 'sst', 'ssrd', 'u10', 'v10', 'sp', 'tp']

VARIABLE_LABELS = {
    'intensidad': 'Intensidad de tráfico',
    'carga': 'Carga de tráfico',
    'ocupacion': 'Ocupación vial',
    'd2m': 'Temperatura punto rocío (°C)',
    't2m': 'Temperatura 2m (°C)',
    'sst': 'Temp. superficie mar (°C)',
    'ssrd': 'Radiación solar (W/m²)',
    'u10': 'Viento U 10m (m/s)',
    'v10': 'Viento V 10m (m/s)',
    'sp': 'Presión superficie (Pa)',
    'tp': 'Precipitación total (mm)'
}


# ==================== CLASE PRINCIPAL ====================

class SensorAnalyzer:
    """Clase principal para análisis de sensores de NO2 y tráfico."""
    
    def __init__(self):
        self.df_master = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión."""
        if 'sensor_data_loaded' not in st.session_state:
            st.session_state.sensor_data_loaded = False
        if 'sensor_config' not in st.session_state:
            st.session_state.sensor_config = {}
    
    @st.cache_data(ttl=3600)
    def load_data(_self) -> pd.DataFrame:
        """Carga y preprocesa los datos de sensores con caché."""
        try:
            df = pd.read_parquet('data/super_processed/7_4_no2_with_traffic_and_1meteo_and_1trafic_id.parquet')
            df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
            return pd.DataFrame()
    
    def filter_and_aggregate_data(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Filtra y agrega los datos según la configuración."""
        # Filtrar por sensor y fechas
        df_filtered = df[
            (df["id_no2"] == config['sensor']) & 
            (df["fecha"].dt.date >= config['fecha_inicio']) & 
            (df["fecha"].dt.date <= config['fecha_fin'])
        ].copy()
        
        # Filtrar por sensor de tráfico si está especificado
        if config.get('sensor_trafico'):
            df_filtered = df_filtered[df_filtered["id_trafico"] == config['sensor_trafico']]
        
        # Aplicar granularidad temporal
        granularity = config['granularity']
        freq = GRANULARITY_CONFIG[granularity]['freq']
        
        if granularity == "Horaria":
            df_filtered["time_group"] = df_filtered["fecha"].dt.floor(freq)
            df_aggregated = df_filtered.copy()
        else:
            if granularity in ["Semanal", "Mensual"]:
                df_filtered["time_group"] = df_filtered["fecha"].dt.to_period(freq).dt.to_timestamp()
            else:
                df_filtered["time_group"] = df_filtered["fecha"].dt.floor(freq)
            
            # Agregar datos
            agg_vars = ["no2_value"] + config['variables']
            df_aggregated = df_filtered.groupby("time_group").agg({
                var: "mean" for var in agg_vars
            }).reset_index()
        
        return df_aggregated
    
    def create_time_series_plot(self, df: pd.DataFrame, variable: str, color: str) -> plt.Figure:
        """Genera un gráfico de series temporales con dos ejes."""
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Eje NO2
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('NO₂ (μg/m³)', color='tab:blue')
        ax1.plot(df['time_group'], df['no2_value'],
                 color='tab:blue', marker='o', label='NO₂', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.axhline(y=40, color='r', linestyle='--', alpha=0.7, label='Límite OMS')
        
        # Eje variable
        ax2 = ax1.twinx()
        variable_label = VARIABLE_LABELS.get(variable, variable.capitalize())
        ax2.set_ylabel(variable_label, color=color)
        ax2.plot(df['time_group'], df[variable],
                 color=color, marker='x', label=variable, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Formateo
        fig.autofmt_xdate()
        plt.title(f'Comparativa NO₂ vs {variable_label}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, variable: str, color: str) -> plt.Figure:
        """Genera un scatter plot con línea de regresión."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calcular correlación
        correlation = df[variable].corr(df['no2_value'])
        
        # Crear scatter plot
        sns.scatterplot(x=df[variable], y=df['no2_value'], color=color, ax=ax, alpha=0.7)
        sns.regplot(x=df[variable], y=df['no2_value'], 
                   scatter=False, color=color, ax=ax, line_kws={'linewidth': 2})
        
        # Formateo
        variable_label = VARIABLE_LABELS.get(variable, variable.capitalize())
        plt.title(f'{variable_label} vs NO₂\nCorrelación: {correlation:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(variable_label)
        plt.ylabel('NO₂ (μg/m³)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_altair_scatter(self, df: pd.DataFrame, variable: str, color: str) -> alt.Chart:
        """Genera un gráfico interactivo de dispersión con Altair."""
        variable_label = VARIABLE_LABELS.get(variable, variable.capitalize())
        
        scatter_chart = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X(variable, title=variable_label),
            y=alt.Y('no2_value', title='NO₂ (μg/m³)'),
            tooltip=[
                alt.Tooltip(variable, title=variable_label, format='.2f'),
                alt.Tooltip('no2_value', title='NO₂', format='.2f'),
                alt.Tooltip('time_group', title='Fecha')
            ]
        ).properties(width=400, height=300)
        
        regression = scatter_chart.transform_regression(
            variable, 'no2_value'
        ).mark_line(color=color, strokeDash=[4, 2], size=2)
        
        return (scatter_chart + regression).resolve_scale(color='independent')
    
    def create_correlation_matrix(self, df: pd.DataFrame, variables: List[str]) -> plt.Figure:
        """Muestra una matriz de correlación entre las variables seleccionadas."""
        cols = ["no2_value"] + variables
        corr_data = df[cols].corr()
        
        # Renombrar columnas para mejor visualización
        rename_dict = {var: VARIABLE_LABELS.get(var, var.capitalize()) for var in variables}
        rename_dict['no2_value'] = 'NO₂'
        corr_data = corr_data.rename(index=rename_dict, columns=rename_dict)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, fmt=".3f",
                   square=True, linewidths=.5, ax=ax, cbar_kws={'label': 'Correlación'})
        plt.title('Matriz de Correlación: NO₂ vs Variables Seleccionadas', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def analyze_data_availability(self, df_filtered: pd.DataFrame) -> Dict:
        """Analiza y calcula métricas de disponibilidad de datos."""
        if df_filtered.empty:
            return {}
        
        fecha_min = df_filtered["fecha"].min()
        fecha_max = df_filtered["fecha"].max()
        
        # Generar rango completo de horas
        todas_horas = pd.date_range(
            start=fecha_min.replace(hour=0, minute=0, second=0),
            end=fecha_max.replace(hour=23, minute=59, second=59),
            freq='H'
        )
        
        # Crear DataFrame completo y marcar datos disponibles
        df_completo = pd.DataFrame(index=todas_horas)
        df_completo.index.name = 'hora'
        df_filtrado_hora = df_filtered.set_index("fecha")
        df_completo['tiene_datos'] = df_completo.index.isin(df_filtrado_hora.index).astype(int)
        
        # Calcular métricas
        total_horas = len(todas_horas)
        horas_con_datos = df_completo['tiene_datos'].sum()
        porcentaje_completitud = (horas_con_datos / total_horas) * 100
        
        # Disponibilidad por hora del día y día de la semana
        df_completo['dia_semana'] = df_completo.index.dayofweek
        pivot_data = df_completo.pivot_table(
            values='tiene_datos', 
            index=df_completo.index.hour,
            columns='dia_semana', 
            aggfunc='mean'
        ) * 100
        
        return {
            'total_horas': total_horas,
            'horas_con_datos': horas_con_datos,
            'porcentaje_completitud': porcentaje_completitud,
            'pivot_data': pivot_data,
            'fecha_inicio': fecha_min,
            'fecha_fin': fecha_max
        }


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def show_availability_analysis(disponibilidad: Dict):
    """Renderiza las métricas y gráficos de disponibilidad."""
    if not disponibilidad:
        st.warning("No hay datos suficientes para analizar la disponibilidad.")
        return
    
    st.subheader("📊 Análisis de disponibilidad de datos")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de horas", 
            f"{disponibilidad['total_horas']:,}",
            help="Número total de horas en el período seleccionado"
        )
    
    with col2:
        st.metric(
            "Horas con datos", 
            f"{disponibilidad['horas_con_datos']:,}",
            help="Número de horas con datos disponibles"
        )
    
    with col3:
        st.metric(
            "Completitud", 
            f"{disponibilidad['porcentaje_completitud']:.1f}%",
            help="Porcentaje de datos disponibles"
        )
    
    with col4:
        horas_faltantes = disponibilidad['total_horas'] - disponibilidad['horas_con_datos']
        st.metric(
            "Datos faltantes", 
            f"{horas_faltantes:,}",
            delta=f"-{100-disponibilidad['porcentaje_completitud']:.1f}%",
            delta_color="inverse",
            help="Número de horas sin datos"
        )
    
    # Heatmap de disponibilidad
    st.subheader("Patrón de disponibilidad por hora y día")
    
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        disponibilidad['pivot_data'], 
        annot=False, 
        cmap='RdYlGn', 
        vmin=0, 
        vmax=100, 
        xticklabels=dias_semana, 
        yticklabels=range(24),
        cbar_kws={'label': 'Disponibilidad (%)'},
        ax=ax
    )
    
    plt.title('Disponibilidad de datos por hora del día y día de la semana', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Día de la semana')
    plt.ylabel('Hora del día')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Información adicional
    with st.expander("ℹ️ Interpretación del análisis de disponibilidad"):
        st.markdown("""
        **¿Cómo interpretar este análisis?**
        
        - **Verde intenso**: Alta disponibilidad de datos (>80%)
        - **Amarillo**: Disponibilidad moderada (40-80%)
        - **Rojo**: Baja disponibilidad de datos (<40%)
        
        **Patrones comunes:**
        - Los sensores pueden tener mantenimiento programado en ciertas horas
        - Algunos días de la semana pueden tener menos cobertura
        - Las interrupciones pueden indicar problemas técnicos o climatológicos
        """)


def show_info_panel():
    """Muestra panel de información sobre el análisis de sensores."""
    with st.expander("ℹ️ Acerca de este análisis", expanded=False):
        st.markdown("""
        **Análisis de Sensores: Tráfico y NO₂**
        
        Este módulo permite analizar la relación entre los datos de tráfico, variables meteorológicas 
        y los niveles de NO₂ en Madrid.
        
        **Funcionalidades:**
        - Análisis de correlaciones entre NO₂ y múltiples variables
        - Visualización de series temporales comparativas
        - Gráficos de dispersión con líneas de regresión
        - Análisis de disponibilidad temporal de datos
        - Matrices de correlación interactivas
        
        **Variables disponibles:**
        - **Tráfico**: Intensidad, carga, ocupación
        - **Meteorológicas**: Temperatura, humedad, presión, viento, precipitación
        
        **Granularidades temporales:**
        - Horaria, diaria, semanal y mensual
        """)


# ==================== FUNCIÓN PRINCIPAL ====================

def analisis_sensores():
    """Función principal del análisis de sensores."""
    
    # Inicializar analizador
    analyzer = SensorAnalyzer()
    
    # Panel de información
    show_info_panel()
    
    # Cargar datos
    if not st.session_state.sensor_data_loaded:
        if st.button("Cargar datos de sensores", type="primary"):
            with st.spinner("Cargando datos de sensores y tráfico..."):
                analyzer.df_master = analyzer.load_data()
                if not analyzer.df_master.empty:
                    st.session_state.sensor_data_loaded = True
                    st.success("Datos cargados correctamente!")
                    st.rerun()
        return
    
    # Recuperar datos
    analyzer.df_master = analyzer.load_data()
    
    if analyzer.df_master.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Configuración en sidebar
    st.sidebar.header("⚙️ Configuración del análisis")
    
    # Selección de sensores
    sensores = sorted(analyzer.df_master["id_no2"].unique())
    sensor_seleccionado = st.sidebar.selectbox("Sensor de NO₂", sensores, index=0)
    
    # Sensor de tráfico asociado
    sensores_trafico = sorted(analyzer.df_master[
        analyzer.df_master["id_no2"] == sensor_seleccionado
    ]["id_trafico"].unique())
    
    if sensores_trafico:
        sensor_trafico = st.sidebar.selectbox(
            "Sensor de tráfico asociado", 
            sensores_trafico, 
            disabled=True,
            help="Sensor de tráfico más cercano al sensor de NO₂ seleccionado"
        )
    else:
        sensor_trafico = None
        st.sidebar.warning("No hay sensores de tráfico asociados")
    
    # Filtros de fecha
    fecha_min = analyzer.df_master["fecha"].min().date()
    fecha_max = analyzer.df_master["fecha"].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fecha_inicio = st.date_input(
            "Fecha inicial", 
            fecha_min, 
            min_value=fecha_min, 
            max_value=fecha_max
        )
    with col2:
        fecha_fin = st.date_input(
            "Fecha final", 
            fecha_max, 
            min_value=fecha_min, 
            max_value=fecha_max
        )
    
    if fecha_inicio > fecha_fin:
        st.sidebar.error("La fecha inicial debe ser anterior a la final.")
        fecha_fin = fecha_inicio + timedelta(days=7)
    
    # Granularidad temporal
    granularity = st.sidebar.selectbox(
        "Granularidad temporal", 
        list(GRANULARITY_CONFIG.keys()),
        index=3  # Mensual por defecto
    )
    
    # Selección de variables
    variables_disponibles = TRAFFIC_VARIABLES + METEO_VARIABLES
    variables_disponibles = [
        var for var in variables_disponibles 
        if var in analyzer.df_master.columns
    ]
    
    variables_seleccionadas = st.sidebar.multiselect(
        "Variables a analizar", 
        variables_disponibles, 
        default=variables_disponibles[:4],
        help="Selecciona las variables que quieres comparar con NO₂"
    )
    
    if not variables_seleccionadas:
        st.warning("Selecciona al menos una variable para continuar.")
        return
    
    # Configurar análisis
    config = {
        'sensor': sensor_seleccionado,
        'sensor_trafico': sensor_trafico,
        'fecha_inicio': fecha_inicio,
        'fecha_fin': fecha_fin,
        'granularity': granularity,
        'variables': variables_seleccionadas
    }
    
    # Procesar datos
    with st.spinner("Procesando datos..."):
        df_aggregated = analyzer.filter_and_aggregate_data(analyzer.df_master, config)
    
    if df_aggregated.empty:
        st.error("No hay datos disponibles para los filtros seleccionados.")
        return
    
    # Mostrar información del conjunto de datos
    st.header(f"📊 Análisis del sensor {sensor_seleccionado}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Períodos analizados", len(df_aggregated))
    with col2:
        st.metric("NO₂ promedio", f"{df_aggregated['no2_value'].mean():.1f} μg/m³")
    with col3:
        st.metric("NO₂ máximo", f"{df_aggregated['no2_value'].max():.1f} μg/m³")
    with col4:
        dias_analisis = (fecha_fin - fecha_inicio).days + 1
        st.metric("Días de análisis", dias_analisis)
    
    # Tabs de visualización
    tab1, tab2, tab3 = st.tabs([
        "📈 Análisis temporal", 
        "🔄 Correlaciones", 
        "📋 Disponibilidad de datos"
    ])
    
    # Tab 1: Análisis temporal
    with tab1:
        st.subheader("Series temporales de NO₂ y variables seleccionadas")
        
        # Organizar gráficos en dos columnas
        for i in range(0, len(variables_seleccionadas), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                variable = variables_seleccionadas[i]
                color = VARIABLE_COLORS[i % len(VARIABLE_COLORS)]
                fig = analyzer.create_time_series_plot(df_aggregated, variable, color)
                st.pyplot(fig)
            
            if i + 1 < len(variables_seleccionadas):
                with col2:
                    variable = variables_seleccionadas[i + 1]
                    color = VARIABLE_COLORS[(i + 1) % len(VARIABLE_COLORS)]
                    fig = analyzer.create_time_series_plot(df_aggregated, variable, color)
                    st.pyplot(fig)
    
    # Tab 2: Correlaciones
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Matriz de correlación")
            fig_corr = analyzer.create_correlation_matrix(df_aggregated, variables_seleccionadas)
            st.pyplot(fig_corr)
        
        with col2:
            st.subheader("Estadísticas de correlación")
            
            # Tabla de correlaciones
            correlations = []
            for var in variables_seleccionadas:
                corr = df_aggregated[var].corr(df_aggregated['no2_value'])
                correlations.append({
                    'Variable': VARIABLE_LABELS.get(var, var.capitalize()),
                    'Correlación': corr,
                    'Interpretación': 'Fuerte' if abs(corr) > 0.7 else 'Moderada' if abs(corr) > 0.3 else 'Débil'
                })
            
            df_corr = pd.DataFrame(correlations)
            st.dataframe(df_corr, use_container_width=True)
        
        # Subtabs para gráficos de dispersión
        scatter_tab1, scatter_tab2 = st.tabs(["Gráficos estáticos", "Gráficos interactivos"])
        
        with scatter_tab1:
            st.subheader("Gráficos de dispersión con regresión")
            for i in range(0, len(variables_seleccionadas), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    variable = variables_seleccionadas[i]
                    color = VARIABLE_COLORS[i % len(VARIABLE_COLORS)]
                    fig = analyzer.create_scatter_plot(df_aggregated, variable, color)
                    st.pyplot(fig)
                
                if i + 1 < len(variables_seleccionadas):
                    with col2:
                        variable = variables_seleccionadas[i + 1]
                        color = VARIABLE_COLORS[(i + 1) % len(VARIABLE_COLORS)]
                        fig = analyzer.create_scatter_plot(df_aggregated, variable, color)
                        st.pyplot(fig)
        
        with scatter_tab2:
            st.subheader("Gráficos interactivos de dispersión")
            for i in range(0, len(variables_seleccionadas), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    variable = variables_seleccionadas[i]
                    color = VARIABLE_COLORS[i % len(VARIABLE_COLORS)]
                    chart = analyzer.create_altair_scatter(df_aggregated, variable, color)
                    st.altair_chart(chart, use_container_width=True)
                
                if i + 1 < len(variables_seleccionadas):
                    with col2:
                        variable = variables_seleccionadas[i + 1]
                        color = VARIABLE_COLORS[(i + 1) % len(VARIABLE_COLORS)]
                        chart = analyzer.create_altair_scatter(df_aggregated, variable, color)
                        st.altair_chart(chart, use_container_width=True)
    
    # Tab 3: Disponibilidad de datos
    with tab3:
        # Necesitamos los datos originales filtrados para el análisis de disponibilidad
        df_original_filtered = analyzer.df_master[
            (analyzer.df_master["id_no2"] == sensor_seleccionado) & 
            (analyzer.df_master["fecha"].dt.date >= fecha_inicio) & 
            (analyzer.df_master["fecha"].dt.date <= fecha_fin)
        ].copy()
        
        if sensor_trafico:
            df_original_filtered = df_original_filtered[
                df_original_filtered["id_trafico"] == sensor_trafico
            ]
        
        disponibilidad = analyzer.analyze_data_availability(df_original_filtered)
        show_availability_analysis(disponibilidad)
    
    # Pie de página
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Análisis de sensor {sensor_seleccionado} | Período: {fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    analisis_sensores() 