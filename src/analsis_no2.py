"""
Módulo para análisis y visualización de datos de contaminación por NO2 en Madrid.

Este módulo proporciona funciones para cargar, procesar y visualizar datos de 
contaminación por NO2, incluyendo mapas de calor, análisis temporal y generación
de timelapses.
"""

import folium
import pandas as pd
import streamlit as st
import numpy as np
import leafmap.foliumap as leafmap
import imageio.v3
import os
import tempfile
import gc
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import altair as alt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# -------------------- FUNCIONES DE CARGA Y PREPROCESAMIENTO DE DATOS --------------------

@st.cache_data(ttl=3600)
def cargar_datos_air() -> pd.DataFrame:
    """
    Carga y preprocesa los datos de contaminación por NO2 con caché.
    
    Returns:
        pd.DataFrame: DataFrame con datos de contaminación por NO2 preprocesados.
    """
    df = pd.read_parquet('data/more_processed/air_data.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def filtrar_datos_por_fecha(
    df: pd.DataFrame, 
    fecha_inicio: datetime.date, 
    fecha_fin: datetime.date
) -> pd.DataFrame:
    """
    Filtra un DataFrame por rango de fechas.
    
    Args:
        df: DataFrame con columna 'fecha'
        fecha_inicio: Fecha inicial del filtro
        fecha_fin: Fecha final del filtro
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    return df[
        (df["fecha"].dt.date >= fecha_inicio) & 
        (df["fecha"].dt.date <= fecha_fin)
    ].copy()


def aplicar_granularidad_temporal(
    df: pd.DataFrame, 
    granularity: str
) -> Tuple[pd.DataFrame, str]:
    """
    Aplica una granularidad temporal al DataFrame y devuelve el formato adecuado.
    
    Args:
        df: DataFrame con columna 'fecha'
        granularity: Granularidad temporal ('Horaria', 'Diaria', etc.)
        
    Returns:
        Tuple[pd.DataFrame, str]: DataFrame modificado y formato para visualización
    """
    if granularity == "Horaria":
        df["time_group"] = df["fecha"].dt.floor("H")
        slider_format = "%Y-%m-%d %H:%M"
    elif granularity == "Diaria":
        df["time_group"] = df["fecha"].dt.floor("D")
        slider_format = "%Y-%m-%d"
    elif granularity == "Semanal":
        df["time_group"] = df["fecha"].dt.to_period("W").dt.to_timestamp()
        slider_format = "%Y-%m-%d"
    elif granularity == "Mensual":
        df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
        slider_format = "%Y-%m"
    else:  # Anual
        df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
        slider_format = "%Y"
    
    return df, slider_format


def agregar_datos_por_tiempo(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Agrega los datos por la granularidad temporal especificada.
    
    Args:
        df: DataFrame con columna 'time_group'
        granularity: Granularidad temporal ('Horaria', 'Diaria', etc.)
        
    Returns:
        pd.DataFrame: DataFrame agregado
    """
    if granularity != "Horaria":
        return df.groupby(["time_group", "latitud", "longitud"]).agg(
            no2_value=("no2_value", "mean"),
            fecha=("fecha", "min")
        ).reset_index()
    return df


def filtrar_datos_por_tiempo_seleccionado(
    df: pd.DataFrame, 
    selected_time: datetime, 
    granularity: str
) -> pd.DataFrame:
    """
    Filtra los datos para un momento temporal específico.
    
    Args:
        df: DataFrame con columna 'time_group'
        selected_time: Momento temporal seleccionado
        granularity: Granularidad temporal
        
    Returns:
        pd.DataFrame: DataFrame filtrado para el momento seleccionado
    """
    try:
        if granularity == "Mensual":
            return df[df["time_group"].dt.to_period("M") == pd.Period(selected_time, "M")]
        elif granularity == "Anual":
            return df[df["time_group"].dt.to_period("Y") == pd.Period(selected_time, "Y")]
        elif granularity == "Semanal":
            return df[df["time_group"].dt.to_period("W") == pd.Period(selected_time, "W")]
        else:
            return df[df["time_group"] == selected_time]
    except Exception as e:
        st.error(f"Error al filtrar datos por tiempo: {str(e)}")
        return pd.DataFrame()


def filtrar_por_nivel_contaminacion(
    df: pd.DataFrame, 
    nivel: Optional[str]
) -> pd.DataFrame:
    """
    Filtra los datos según el nivel de contaminación.
    
    Args:
        df: DataFrame con columna 'no2_value'
        nivel: Nivel de contaminación ('Bajo', 'Medio', 'Alto', None)
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    if nivel is None:
        return df
        
    if nivel == "Bajo":
        return df[df['no2_value'] <= 40]
    elif nivel == "Medio":
        return df[(df['no2_value'] > 40) & (df['no2_value'] <= 100)]
    elif nivel == "Alto":
        return df[df['no2_value'] > 100]
    
    return df


def calcular_estadisticas_no2(df: pd.DataFrame) -> Tuple[float, float, str, str]:
    """
    Calcula estadísticas básicas de NO2 y determina el nivel de contaminación.
    
    Args:
        df: DataFrame con columna 'no2_value'
        
    Returns:
        Tuple[float, float, str, str]: Media, máximo, nivel y color asociado
    """
    avg_no2 = df["no2_value"].mean()
    max_no2 = df["no2_value"].max()
    
    if avg_no2 <= 40:
        nivel = "Bajo"
        color = "green"
    elif avg_no2 <= 100:
        nivel = "Medio"
        color = "orange"
    else:
        nivel = "Alto"
        color = "red"
        
    return avg_no2, max_no2, nivel, color


def generar_estadisticas_temporales(
    df: pd.DataFrame, 
    slider_format: str
) -> pd.DataFrame:
    """
    Genera estadísticas temporales para gráficos.
    
    Args:
        df: DataFrame con columna 'time_group'
        slider_format: Formato para mostrar las fechas
        
    Returns:
        pd.DataFrame: DataFrame con estadísticas temporales
    """
    stats_df = df.groupby("time_group").agg(
        no2_promedio=("no2_value", "mean"),
        no2_max=("no2_value", "max"),
        num_readings=("no2_value", "count")
    ).reset_index()
    
    stats_df["fecha_str"] = stats_df["time_group"].dt.strftime(slider_format)
    stats_df["time_group"] = pd.to_datetime(stats_df["time_group"])
    
    return stats_df


# -------------------- FUNCIONES DE VISUALIZACIÓN --------------------

def crear_mapa_con_heatmap(
    df_selected: pd.DataFrame, 
    global_min: float, 
    global_max: float, 
    nivel_contaminacion: Optional[str] = None
) -> Optional[leafmap.Map]:
    """
    Crea un mapa Folium con heatmap de contaminación NO2.
    
    Args:
        df_selected: DataFrame con datos para visualizar
        global_min: Valor mínimo global para normalización
        global_max: Valor máximo global para normalización
        nivel_contaminacion: Filtro opcional por nivel de contaminación
        
    Returns:
        Optional[leafmap.Map]: Mapa con heatmap o None si no hay datos
    """
    if df_selected.empty:
        return None
        
    # Aplicar filtro por nivel si es necesario
    df_filtered = filtrar_por_nivel_contaminacion(df_selected, nivel_contaminacion)
    if df_filtered.empty:
        return None
    
    # Centrar el mapa en los datos
    map_center = [df_filtered["latitud"].mean(), df_filtered["longitud"].mean()]

    # Crear mapa base
    m = leafmap.Map(
        center=map_center,
        zoom=12,
        tiles="CartoDB positron",
        draw_control=False,
        measure_control=False,
        fullscreen_control=True
    )
    
    # Limitar número de puntos para mejor rendimiento
    max_points = 2000
    if len(df_filtered) > max_points:
        df_filtered = df_filtered.sample(max_points)

    # Configurar parámetros de heatmap según cantidad de datos
    radius = 15 if len(df_filtered) > 100 else 25
    blur = 10 if len(df_filtered) > 100 else 15

    # Normalizar valores una sola vez para evitar operaciones repetidas
    heat_data = []
    for _, row in df_filtered.iterrows():
        normalized_value = max(0.1, min(1, (row['no2_value'] - global_min) / (global_max - global_min) * 0.8 + 0.2))
        heat_data.append([row['latitud'], row['longitud'], normalized_value])
    
    # Añadir heatmap al mapa
    m.add_heatmap(
        data=heat_data,
        name="NO2 Heatmap",
        radius=radius,
        blur=blur,
    )

    return m


def mostrar_estadisticas_basicas(avg_no2: float, max_no2: float, nivel: str, color: str) -> None:
    """
    Muestra estadísticas básicas de NO2 con formato visual mejorado.
    
    Args:
        avg_no2: Valor promedio de NO2
        max_no2: Valor máximo de NO2
        nivel: Nivel de contaminación ('Bajo', 'Medio', 'Alto')
        color: Color para representar el nivel
    """
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 1rem;">
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 100%; margin-bottom: 0.5rem;">
                <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
                <div style="font-size: 1.5rem; color: {color};">{avg_no2:.1f} μg/m³</div>
            </div>
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 100%; margin-bottom: 0.5rem;">
                <div style="font-size: 0.8rem; color: #666;">Máximo NO₂</div>
                <div style="font-size: 1.5rem; color: red">{max_no2:.1f} μg/m³</div>
            </div>
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 100%;">
                <div style="font-size: 0.8rem; color: #666;">Nivel</div>
                <div style="font-size: 1.5rem; color: {color};">{nivel}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def mostrar_grafico_evolucion_temporal(stats_df: pd.DataFrame, slider_format: str) -> None:
    """
    Muestra un gráfico de línea con la evolución temporal de NO2.
    
    Args:
        stats_df: DataFrame con estadísticas temporales
        slider_format: Formato para mostrar las fechas
    """
    st.write("**Evolución temporal de NO₂**")
    st.write("La OMS recomienda que los niveles medios anuales de NO₂ no superen los 40 μg/m³ (línea roja).")
    
    # Crear gráfico de línea
    line_chart = alt.Chart(stats_df).mark_line(point=True).encode(
        x=alt.X('time_group:T', title='Fecha', axis=alt.Axis(format=slider_format)),
        y=alt.Y('no2_promedio:Q', title='NO₂ promedio (μg/m³)'),
        tooltip=[
            alt.Tooltip('fecha_str:N', title='Fecha'),
            alt.Tooltip('no2_promedio:Q', title='NO₂ promedio', format='.1f'),
            alt.Tooltip('no2_max:Q', title='NO₂ máximo', format='.1f'),
            alt.Tooltip('num_readings:Q', title='Nº de mediciones')
        ]
    ).properties(height=200)
    
    # Añadir línea de límite recomendado
    limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(
        color='red', 
        strokeDash=[3, 3]
    ).encode(y='y:Q')
    
    # Mostrar gráfico
    st.altair_chart(line_chart + limit_line, use_container_width=True)


def mostrar_descomposicion_serie_temporal(stats_df: pd.DataFrame, slider_format: str) -> None:
    """
    Muestra la descomposición estacional de la serie temporal de NO2.

    Args:
        stats_df: DataFrame con estadísticas temporales.
        slider_format: Formato para mostrar las fechas.
    """

    st.markdown("### 📊 Descomposición de la serie temporal de NO₂")
    st.write("**Descomposición estacional de la serie temporal de NO₂**")
    
    # Asegurar que los datos están indexados por fecha
    df = stats_df.copy()
    df.set_index('time_group', inplace=True)
    
    # Aplicar descomposición estacional
    # Determinar el periodo basado en el formato del slider
    if slider_format == "%Y-%m":
        period = 12  # Mensual
    elif slider_format == "%Y-%m-%d":
        period = 365  # Diario
    else:
        period = 7  # Semanal o cualquier otro caso

    # Aplicar descomposición estacional
    result = seasonal_decompose(df['no2_promedio'], model='additive', period=period)
    
    # Graficar resultados
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Serie Original", color="black")
    result.trend.plot(ax=axes[1], title="Tendencia", color="blue")
    result.seasonal.plot(ax=axes[2], title="Estacionalidad", color="green")
    result.resid.plot(ax=axes[3], title="Ruido (Residuos)", color="red")
    
    plt.tight_layout()
    st.pyplot(fig)


def mostrar_histograma_no2(df: pd.DataFrame) -> None:
    """
    Muestra un histograma con la distribución de valores de NO2 coloreado por niveles
    de contaminación, con un estilo visual consistente.
    
    Args:
        df: DataFrame con columna 'no2_value' (ya filtrado si es necesario)
    """
    st.write("**Distribución de valores de NO₂**")
    
    if df.empty:
        st.warning("No hay datos disponibles para generar el histograma.")
        return
    
    # Calcular estadísticas para mostrar
    media = df['no2_value'].mean()
    mediana = df['no2_value'].median()
    max_value = df['no2_value'].max()
    min_value = df['no2_value'].min()
    
    # Determinar niveles y colores para las estadísticas
    if media <= 40:
        nivel_media = "Bajo"
        color_media = "green"
    elif media <= 100:
        nivel_media = "Medio"
        color_media = "orange"
    else:
        nivel_media = "Alto"
        color_media = "red"
    
    # Color para el máximo siempre es rojo
    if max_value <= 40:
        color_max = "green"
    elif max_value <= 100:
        color_max = "orange"
    else:
        color_max = "red"
    
    # Determinar rango adecuado para el eje X
    x_min = max(0, min_value * 0.9)
    x_max = max_value * 1.1
    
    # Calcular bins con valores redondos para mejor interpretación
    bin_size = 5
    if max_value - min_value > 200:
        bin_size = 10
    elif max_value - min_value < 50:
        bin_size = 2
    
    # Añadir categoría de nivel para colorear
    df_with_level = df.copy()
    df_with_level['nivel'] = pd.cut(
        df_with_level['no2_value'], 
        bins=[0, 40, 100, float('inf')], 
        labels=['Bajo', 'Medio', 'Alto'],
        include_lowest=True
    )
    
    # Definir esquema de colores según los niveles
    color_scale = alt.Scale(
        domain=['Bajo', 'Medio', 'Alto'],
        range=['green', 'orange', 'red']
    )
    
    # Crear histograma mejorado con colores por nivel
    hist = alt.Chart(df_with_level).mark_bar().encode(
        x=alt.X('no2_value:Q', 
                bin=alt.Bin(step=bin_size),
                title='Concentración de NO₂ (μg/m³)',
                scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('count():Q', 
                title='Número de mediciones',
                scale=alt.Scale(zero=True)),
        color=alt.Color('nivel:N', 
                       scale=color_scale,
                       legend=alt.Legend(title="Nivel de contaminación"))
    ).properties(
        height=200
    )
    
    # Líneas verticales para los límites
    limit_line_40 = alt.Chart(pd.DataFrame({'x': [40]})).mark_rule(
        color='black', 
        strokeDash=[3, 3]
    ).encode(x='x:Q')
    
    limit_line_100 = alt.Chart(pd.DataFrame({'x': [100]})).mark_rule(
        color='black', 
        strokeDash=[3, 3]
    ).encode(x='x:Q')
    
    # Línea vertical para la media
    mean_line = alt.Chart(pd.DataFrame({'x': [media]})).mark_rule(
        color='blue', 
        strokeDash=[5, 3]
    ).encode(x='x:Q')
    
    # Texto para las líneas
    limit_40_text = alt.Chart(pd.DataFrame({'x': [40], 'y': [0], 'text': ['Límite OMS (40)']})).mark_text(
        color='black',
        align='left',
        baseline='bottom',
        dx=5,
        dy=-5,
        fontSize=10
    ).encode(x='x:Q', y='y:Q', text='text:N')
    
    limit_100_text = alt.Chart(pd.DataFrame({'x': [100], 'y': [0], 'text': ['Límite crítico (100)']})).mark_text(
        color='black',
        align='left',
        baseline='bottom',
        dx=5,
        dy=-15,
        fontSize=10
    ).encode(x='x:Q', y='y:Q', text='text:N')
    
    mean_text = alt.Chart(pd.DataFrame({'x': [media], 'y': [0], 'text': [f'Media: {media:.1f}']})).mark_text(
        color='blue',
        align='right',
        baseline='bottom',
        dx=-5,
        dy=-5,
        fontSize=10
    ).encode(x='x:Q', y='y:Q', text='text:N')
    
    # Mostrar histograma con líneas
    chart = hist + limit_line_40 + limit_line_100 + mean_line + limit_40_text + limit_100_text + mean_text
    st.altair_chart(chart, use_container_width=True)
    
    # Mostrar estadísticas en el mismo estilo visual que las estadísticas básicas
    st.markdown(f"""
        <div style="display: flex; flex-direction: row; justify-content: space-between; margin-top: 1rem;">
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 24%; margin-right: 0.5rem;">
                <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
                <div style="font-size: 1.5rem; color: {color_media};">{media:.1f} μg/m³</div>
            </div>
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 24%; margin-right: 0.5rem;">
                <div style="font-size: 0.8rem; color: #666;">Mediana NO₂</div>
                <div style="font-size: 1.5rem; color: {color_media};">{mediana:.1f} μg/m³</div>
            </div>
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 24%; margin-right: 0.5rem;">
                <div style="font-size: 0.8rem; color: #666;">Mínimo NO₂</div>
                <div style="font-size: 1.5rem; color: green;">{min_value:.1f} μg/m³</div>
            </div>
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 24%;">
                <div style="font-size: 0.8rem; color: #666;">Máximo NO₂</div>
                <div style="font-size: 1.5rem; color: {color_max};">{max_value:.1f} μg/m³</div>
            </div>
        </div>
        <div style="display: flex; flex-direction: row; justify-content: center; margin-top: 0.5rem;">
            <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
                <div style="font-size: 0.8rem; color: #666;">Nivel promedio</div>
                <div style="font-size: 1.5rem; color: {color_media};">{nivel_media}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Leyenda de niveles
    st.markdown(f"""
    <div style="margin-top: 15px; font-size: 0.8rem; color: #666; display: flex; flex-direction: row; justify-content: center; gap: 20px;">
        <div><span style="color: green; font-weight: bold;">■</span> Bajo (≤ 40 μg/m³)</div>
        <div><span style="color: orange; font-weight: bold;">■</span> Medio (41-100 μg/m³)</div>
        <div><span style="color: red; font-weight: bold;">■</span> Alto (> 100 μg/m³)</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- FUNCIONES PARA GENERACIÓN DE TIMELAPSE --------------------

def generar_timelapse(
    df: pd.DataFrame, 
    time_groups: List[datetime], 
    slider_format: str, 
    global_min: float, 
    global_max: float, 
    modo: str = "gif", 
    fps: int = 2, 
    nivel_contaminacion: Optional[str] = None
) -> Optional[bytes]:
    """
    Genera un GIF o video (mp4) a partir de mapas con datos de diferentes momentos.
    
    Args:
        df: DataFrame con todos los datos
        time_groups: Lista de grupos temporales únicos
        slider_format: Formato para mostrar las fechas
        global_min: Valor mínimo global para normalización
        global_max: Valor máximo global para normalización
        modo: Formato de salida ('gif' o 'mp4')
        fps: Frames por segundo
        nivel_contaminacion: Filtro opcional por nivel de contaminación
        
    Returns:
        Optional[bytes]: Datos binarios del timelapse o None si falla
    """
    with st.spinner('Generando timelapse... Este proceso puede tardar unos minutos.'):
        frames = []
        temp_frames = []
        temp_dir = tempfile.mkdtemp()
        
        # Limitar número de frames si hay demasiados
        if len(time_groups) > 30:
            step = len(time_groups) // 30
            time_groups = time_groups[::step]
            
        total_frames = len(time_groups)
        progress_bar = st.progress(0)
        
        # Procesar cada frame
        for i, t in enumerate(time_groups):
            # Actualizar barra de progreso
            progress_bar.progress((i + 1) / total_frames)
            
            # Filtrar datos para este momento temporal
            df_t = filtrar_datos_para_timelapse(df, t, slider_format)
            
            if df_t.empty:
                continue
            
            # Limitar tamaño de datos para mejorar rendimiento
            if len(df_t) > 1000:
                df_t = df_t.sample(1000)
            
            # Crear mapa para este momento
            m = crear_mapa_con_heatmap(df_t, global_min, global_max, nivel_contaminacion)
            
            if m is None:
                continue
                
            # Añadir título con timestamp
            añadir_titulo_a_mapa(m, t, slider_format)
            
            # Guardar cada mapa como imagen temporal
            temp_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
            temp_frames.append(temp_file)
            
            try:
                # Capturar imagen del mapa
                capturar_imagen_de_mapa(m, temp_file)
                
                # Añadir frame a la lista
                img = imageio.v3.imread(temp_file)
                frames.append(img)
                
                # Liberar memoria
                del m, img
                gc.collect()
                
            except Exception as e:
                st.error(f"Error al generar frame {i}: {str(e)}")
                continue
        
        progress_bar.empty()
        
        # Verificar si se generaron frames
        if not frames:
            st.warning("No se pudieron generar frames para el timelapse.")
            return None
            
        # Generar archivo final
        return guardar_timelapse(frames, temp_dir, modo, fps, temp_frames)


def filtrar_datos_para_timelapse(
    df: pd.DataFrame, 
    t: Union[datetime, Any], 
    slider_format: str
) -> pd.DataFrame:
    """
    Filtra datos para un momento temporal específico del timelapse.
    
    Args:
        df: DataFrame con datos
        t: Momento temporal
        slider_format: Formato para mostrar las fechas
        
    Returns:
        pd.DataFrame: DataFrame filtrado para el momento t
    """
    if isinstance(t, datetime):
        if isinstance(df["time_group"].iloc[0], datetime):
            # Si ambos son datetime, comparar directamente
            return df[df["time_group"] == t]
        else:
            # Convertir la columna time_group a datetime para comparar
            return df[pd.to_datetime(df["time_group"]) == t]
    else:
        # Para comparaciones con fechas o periodos
        time_format = "%Y-%m" if slider_format == "%Y-%m" else "%Y-%m-%d" if "-%d" in slider_format else "%Y"
        t_str = t.strftime(time_format) if hasattr(t, 'strftime') else str(t)
        
        if "time_group_str" not in df.columns:
            df["time_group_str"] = df["time_group"].dt.strftime(time_format)
        
        return df[df["time_group_str"] == t_str]


def añadir_titulo_a_mapa(m: leafmap.Map, t: Union[datetime, Any], slider_format: str) -> None:
    """
    Añade un título con timestamp al mapa.
    
    Args:
        m: Mapa al que añadir el título
        t: Momento temporal
        slider_format: Formato para mostrar las fechas
    """
    title_html = f'''
        <h3 style="position:absolute;z-index:1000;left:50px;top:10px;background-color:white;
        padding:5px;border-radius:5px;box-shadow:0 0 5px rgba(0,0,0,0.3);">
        {t.strftime(slider_format) if hasattr(t, 'strftime') else str(t)}
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))


def capturar_imagen_de_mapa(m: leafmap.Map, temp_file: str) -> None:
    """
    Captura una imagen de un mapa folium.
    
    Args:
        m: Mapa a capturar
        temp_file: Ruta del archivo temporal para guardar la imagen
    
    Raises:
        Exception: Si hay un error al capturar la imagen
    """
    # Guardar mapa como HTML
    m.save(temp_file.replace('.png', '.html'))
    
    # Intentar usar Selenium para convertir HTML a imagen
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.set_window_size(1280, 1024)
        driver.get('file://' + temp_file.replace('.png', '.html'))
        driver.save_screenshot(temp_file)
        driver.quit()
    except Exception as e:
        # Si falla con Selenium, intentar con otra biblioteca
        st.warning(f"No se pudo usar Selenium: {str(e)}. Usando método alternativo...")
        try:
            import imgkit
            imgkit.from_file(temp_file.replace('.png', '.html'), temp_file)
        except Exception as inner_e:
            raise Exception(f"No se pudo capturar la imagen: {str(inner_e)}")


def guardar_timelapse(
    frames: List[np.ndarray], 
    temp_dir: str, 
    modo: str, 
    fps: int, 
    temp_frames: List[str]
) -> Optional[bytes]:
    """
    Guarda un timelapse a partir de frames capturados.
    
    Args:
        frames: Lista de frames (imágenes)
        temp_dir: Directorio temporal para guardar archivos
        modo: Formato de salida ('gif' o 'mp4')
        fps: Frames por segundo
        temp_frames: Lista de rutas de archivos temporales
        
    Returns:
        Optional[bytes]: Datos binarios del timelapse o None si falla
    """
    output_file = os.path.join(temp_dir, f"timelapse.{modo}")
    
    try:
        if modo == "gif":
            imageio.v3.imwrite(output_file, frames, loop=0, fps=fps)
        else:  # mp4
            imageio.v3.imwrite(
                output_file, 
                frames, 
                fps=fps, 
                codec='h264', 
                output_params=['-pix_fmt', 'yuv420p']
            )
            
        # Leer el archivo para devolverlo
        with open(output_file, 'rb') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error al guardar el timelapse: {str(e)}")
        return None
    finally:
        # Limpiar archivos temporales
        for f in temp_frames:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    os.remove(f.replace('.png', '.html'))
                except:
                    pass


# -------------------- SECCIÓN DE UI Y CONTROLES --------------------

def mostrar_info_dashboard() -> None:
    """Muestra información contextual sobre el dashboard."""
    with st.expander("ℹ️ Acerca de este dashboard", expanded=False):
        st.markdown("""
        <div class="info-box">
        <p>Este dashboard permite analizar la evolución temporal de los niveles de NO₂ en Madrid a distintas escalas temporales.</p>
        <p><strong>Cómo usar:</strong></p>
        <ul>
            <li>Selecciona un rango de fechas para filtrar los datos</li>
            <li>Elige la granularidad temporal (horaria, diaria, semanal, mensual o anual)</li>
            <li>Utiliza el slider para visualizar la concentración de NO₂ en momentos específicos</li>
            <li>Puedes generar timelapses para observar la evolución en el tiempo</li>
            <li>Explora las estadísticas complementarias en los gráficos inferiores</li>
        </ul>
        <p><strong>Nota:</strong> El NO₂ es un contaminante asociado principalmente al tráfico rodado. La OMS recomienda no superar los 40 μg/m³ de media anual.</p>
        </div>
        """, unsafe_allow_html=True)


def mostrar_seccion_timelapse(
    df: pd.DataFrame,
    time_groups: List[datetime],
    slider_format: str,
    global_min: float,
    global_max: float,
    config: Dict[str, Any]
) -> None:
    """
    Muestra la sección para generar y descargar timelapse.
    
    Args:
        df: DataFrame con datos procesados
        time_groups: Lista de grupos temporales
        slider_format: Formato para mostrar fechas
        global_min: Valor mínimo global para normalización
        global_max: Valor máximo global para normalización
        config: Configuración con opciones de timelapse
    """
    st.markdown('<div class="sub-header">🎬 Timelapse de evolución temporal</div>', unsafe_allow_html=True)
    col1, _ = st.columns([1, 2])
    
    with col1:
        if st.button("🎬 Generar Timelapse", key="generate_timelapse"):
            try:
                timelapse_data = generar_timelapse(
                    df, 
                    time_groups, 
                    slider_format, 
                    global_min, 
                    global_max, 
                    modo=config["timelapse_format"], 
                    fps=config["fps"], 
                    nivel_contaminacion=config["nivel"]
                )
                
                if timelapse_data:
                    st.session_state.timelapse = timelapse_data
                    st.session_state.timelapse_format = config["timelapse_format"]
                    st.success("✅ Timelapse generado correctamente!")
                else:
                    st.error("❌ No se pudo generar el timelapse.")
            except Exception as e:
                st.error(f"Error al generar timelapse: {str(e)}")
                st.info("Asegúrate de tener instaladas las dependencias necesarias (Selenium o imgkit).")

        # Botón de descarga si hay timelapse generado
        if 'timelapse' in st.session_state and st.session_state.timelapse:
            filename = f"timelapse_no2_{config['fecha_inicio']}_{config['fecha_fin']}.{st.session_state.timelapse_format}"
            st.download_button(
                label=f"⬇️ Descargar {st.session_state.timelapse_format.upper()}",
                data=st.session_state.timelapse,
                file_name=filename,
                mime="image/gif" if st.session_state.timelapse_format == "gif" else "video/mp4"
            )
        else:
            st.info("Haz clic en 'Generar Timelapse' para crear una visualización animada de la evolución de NO₂ en el tiempo.")


def mostrar_pie_pagina(fecha_max: datetime) -> None:
    """
    Muestra un pie de página con información de actualización.
    
    Args:
        fecha_max: Fecha máxima de los datos
    """
    st.markdown(f"""
    <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
        Datos proporcionados por el Ayuntamiento de Madrid. Última actualización: {fecha_max.strftime("%d/%m/%Y")}
    </div>
    """, unsafe_allow_html=True)


# -------------------- FUNCIÓN PRINCIPAL --------------------

def generar_analisis_no2() -> None:
    """
    Función principal para la visualización y análisis de datos de NO₂ en Madrid.
    """
    st.markdown('<div class="sub-header">🌍 Análisis de niveles de NO₂ en Madrid</div>', unsafe_allow_html=True)
    

    # Mostrar información del dashboard
    mostrar_info_dashboard()

    # 1. INICIALIZACIÓN DE SESSION STATE - Lo primero y más importante
    inicializar_session_state()
    
    # 2. CARGA DE DATOS (solo una vez)
    if not st.session_state.data_loaded:
        if st.button("Cargar datos de NO₂"):
            
            with st.spinner('Cargando datos...'):
                try:
                    # Cargar datos
                    df_original = cargar_datos_air()
                    
                    # Guardar datos en session_state
                    st.session_state.df_original = df_original
                    st.session_state.global_min = df_original["no2_value"].min()
                    st.session_state.global_max = df_original["no2_value"].max()
                    st.session_state.data_loaded = True
                    
                    # Inicializar valores por defecto para la configuración
                    fecha_min = df_original["fecha"].min().date()
                    fecha_max = df_original["fecha"].max().date()
                    st.session_state.config = {
                        "sensor": "Todos",
                        "fecha_inicio": fecha_min,
                        "fecha_fin": fecha_max,
                        "granularity": "Mensual",
                        "nivel": None,
                        "nivel_display": "Todos",
                        "show_stats": True,
                        "filtrar_outliers": False,
                        "timelapse_format": "gif",
                        "fps": 2
                    }
                    
                    st.rerun()  # Forzar recarga
                except Exception as e:
                    st.error(f"Error al cargar los datos: {str(e)}")
                    return
    elif "df_original" not in st.session_state:
        st.warning("Los datos deben ser cargados. Por favor, haz clic en 'Cargar datos de NO₂'.")
        st.session_state.data_loaded = False
        return
    
    # 3. SI LOS DATOS ESTÁN CARGADOS, MOSTRAR LA INTERFAZ
    if st.session_state.data_loaded:
        df_original = st.session_state.df_original
        global_min = st.session_state.global_min
        global_max = st.session_state.global_max

        # Crear controles de configuración
        st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        with st.container():
            col1, col2_main = st.columns([1, 3])
            
            # Panel de configuración
            with col1:
                # Variables para tracking de cambios
                old_config = st.session_state.config.copy()
                
                # Cargar controles de configuración
                sensores = ["Todos"] + sorted(df_original["id_no2"].unique())
                sensor_index = sensores.index(st.session_state.config["sensor"]) if st.session_state.config["sensor"] in sensores else 0
                
                # Usar callbacks para actualizar valores - esto es clave
                def on_sensor_change():
                    st.session_state.config["sensor"] = st.session_state.sensor_widget
                    st.session_state.need_reprocess = True

                    # Forzar actualización de todas las visualizaciones y estadísticas
                    st.session_state.need_refresh_map = True
                    st.session_state.need_refresh_hist = True

                    # Limpiar caché de visualizaciones al cambiar el sensor
                    for key in list(st.session_state.keys()):
                        if key.startswith(("mapa_", "stats_", "estadisticas_", "histograma_")):
                            del st.session_state[key]
                                
                sensor_seleccionado = st.selectbox(
                    "Selecciona un sensor de NO₂",
                    sensores,
                    index=sensor_index,
                    key="sensor_widget",
                    on_change=on_sensor_change
                )
                
                # Filtro de fechas con callbacks
                st.markdown("#### 📅 Rango de fechas")
                fecha_min = df_original["fecha"].min().date()
                fecha_max = df_original["fecha"].max().date()
                
                def on_fecha_inicio_change():
                    st.session_state.config["fecha_inicio"] = st.session_state.fecha_inicio_widget
                    st.session_state.need_reprocess = True
                
                def on_fecha_fin_change():
                    st.session_state.config["fecha_fin"] = st.session_state.fecha_fin_widget
                    st.session_state.need_reprocess = True
                
                st.date_input(
                    "Fecha inicial", 
                    st.session_state.config["fecha_inicio"], 
                    min_value=fecha_min, 
                    max_value=fecha_max,
                    key="fecha_inicio_widget",
                    on_change=on_fecha_inicio_change
                )
                
                st.date_input(
                    "Fecha final", 
                    st.session_state.config["fecha_fin"], 
                    min_value=fecha_min, 
                    max_value=fecha_max,
                    key="fecha_fin_widget",
                    on_change=on_fecha_fin_change
                )
                
                # Validar fechas
                if st.session_state.config["fecha_inicio"] > st.session_state.config["fecha_fin"]:
                    st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
                    st.session_state.config["fecha_fin"] = st.session_state.config["fecha_inicio"] + timedelta(days=7)
                    st.session_state.need_reprocess = True
                
                # Granularidad y filtros con callbacks
                st.markdown("#### ⏱️ Agregación y filtro")
                granularidad_options = ["Horaria", "Diaria", "Semanal", "Mensual", "Anual"]
                
                def on_granularity_change():
                    st.session_state.config["granularity"] = st.session_state.granularity_widget
                    st.session_state.need_reprocess = True
                
                st.radio(
                    "Granularidad", 
                    granularidad_options,
                    index=granularidad_options.index(st.session_state.config["granularity"]),
                    key="granularity_widget",
                    on_change=on_granularity_change,
                    horizontal=True
                )
                
                nivel_options = ["Todos", "Bajo", "Medio", "Alto"]
                nivel_index = nivel_options.index(st.session_state.config.get("nivel_display", "Todos"))
                
                def on_nivel_change():
                    st.session_state.config["nivel_display"] = st.session_state.nivel_widget
                    st.session_state.config["nivel"] = None if st.session_state.nivel_widget == "Todos" else st.session_state.nivel_widget
                    # Invalida cache de visualizaciones
                    st.session_state.need_refresh_map = True
                
                st.selectbox(
                    "Nivel de contaminación", 
                    nivel_options, 
                    index=nivel_index,
                    key="nivel_widget",
                    on_change=on_nivel_change
                )
                
                # Opciones de visualización con callbacks
                st.markdown("#### 🔆 Opciones de visualización y timelapse")
                
                def on_stats_change():
                    st.session_state.config["show_stats"] = st.session_state.show_stats_widget
                
                def on_outliers_change():
                    # Actualizar valor en config
                    st.session_state.config["filtrar_outliers"] = st.session_state.filtrar_outliers_widget
                    
                    # Forzar reprocesamiento
                    st.session_state.need_reprocess = True
                    st.session_state.need_refresh_map = True
                    st.session_state.need_refresh_hist = True
                    
                    # Limpiar caché
                    for key in list(st.session_state.keys()):
                        if key.startswith(("mapa_", "stats_", "estadisticas_", "histograma_")):
                            del st.session_state[key]
                
                st.checkbox(
                    "Filtrar valores extremos (outliers)", 
                    value=st.session_state.config["filtrar_outliers"],
                    key="filtrar_outliers_widget",
                    on_change=on_outliers_change,
                    help="Elimina el 2% de valores más extremos para mejorar la visualización"
                )
                
                timelapse_options = ["gif", "mp4"]
                
                def on_timelapse_format_change():
                    st.session_state.config["timelapse_format"] = st.session_state.timelapse_format_widget
                
                def on_fps_change():
                    st.session_state.config["fps"] = st.session_state.fps_widget
                
                st.radio(
                    "Formato de salida", 
                    timelapse_options,
                    index=timelapse_options.index(st.session_state.config["timelapse_format"]),
                    key="timelapse_format_widget",
                    on_change=on_timelapse_format_change,
                    horizontal=True
                )
                
                st.slider(
                    "Velocidad (fps)", 
                    min_value=1, 
                    max_value=10, 
                    value=st.session_state.config["fps"],
                    key="fps_widget",
                    on_change=on_fps_change
                )

                # Botón para forzar actualización
                if st.button("Actualizar visualizaciones", key="forzar_actualizacion"):
                    st.session_state.need_reprocess = True
                    st.session_state.need_refresh_map = True
                    st.session_state.need_refresh_hist = True
                    st.rerun()  # Esto está bien porque no es un callback
            
            # Área principal de visualización
            with col2_main:
                # Verificar si necesitamos reprocesar los datos
                if "df_processed" not in st.session_state or st.session_state.need_reprocess:
                    with st.spinner("Procesando datos..."):
                        # Verificar que df_original existe
                        if "df_original" not in st.session_state:
                            st.error("Los datos no están cargados. Por favor, carga los datos primero.")
                            st.session_state.data_loaded = False
                            return
                        
                        df_original = st.session_state.df_original
                        
                        # Filtrar por sensor seleccionado
                        if st.session_state.config["sensor"] != "Todos":
                            df_original_filtered = df_original[df_original["id_no2"] == st.session_state.config["sensor"]]
                        else:
                            df_original_filtered = df_original
                        
                        # Filtrar por fechas
                        df = filtrar_datos_por_fecha(
                            df_original_filtered, 
                            st.session_state.config["fecha_inicio"], 
                            st.session_state.config["fecha_fin"]
                        )
                        
                        # Guardar el dataframe original filtrado por fecha para reutilizarlo
                        st.session_state.df_original_date_filtered = df.copy()
                        
                        # Ahora aplicar el filtro de outliers si está activado
                        if st.session_state.config["filtrar_outliers"]:
                            df = filtrar_outliers_para_visualizacion(df)
                        
                        if df.empty:
                            st.error("⚠️ No hay datos disponibles para el rango seleccionado.")
                            return

                        # Continuar con el resto del procesamiento...
                        df, slider_format = aplicar_granularidad_temporal(df, st.session_state.config["granularity"])
                        df = agregar_datos_por_tiempo(df, st.session_state.config["granularity"])
                        
                        # Guardar en session_state
                        st.session_state.df_processed = df
                        st.session_state.slider_format = slider_format
                        
                        # Obtener grupos temporales para el slider
                        time_groups = sorted(df["time_group"].unique())
                        if len(time_groups) == 0:
                            st.error("⚠️ No hay suficientes datos para la granularidad seleccionada.")
                            return
                        
                        st.session_state.time_groups = time_groups
                        
                        # Resetear el tiempo seleccionado cuando cambia la granularidad
                        if "selected_time" in st.session_state:
                            del st.session_state.selected_time
                        
                        # Ya no necesitamos reprocesar
                        st.session_state.need_reprocess = False
                        
                        # Pero necesitamos refrescar visualizaciones
                        st.session_state.need_refresh_map = True
                        st.session_state.need_refresh_hist = True
                else:
                    # Recuperar de session_state
                    df = st.session_state.df_processed
                    slider_format = st.session_state.slider_format
                    time_groups = st.session_state.time_groups

                # Sección de mapa
                st.markdown('<div class="sub-header">🗺️ Mapa de concentraciones de NO₂</div>', unsafe_allow_html=True)
                
                # Selector de tiempo con callback
                if "selected_time" not in st.session_state:
                    st.session_state.selected_time = time_groups[0]
                
                slider_values = [
                    t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t 
                    for t in time_groups
                ]
                
                # Buscar el índice actual para mantener la selección
                try:
                    current_index = slider_values.index(st.session_state.selected_time)
                except (ValueError, TypeError):
                    current_index = 0
                
                def on_time_change():
                    st.session_state.selected_time = st.session_state.time_slider
                    st.session_state.need_refresh_map = True
                
                selected_time = st.select_slider(
                    "Selecciona el momento temporal",
                    options=slider_values,
                    format_func=lambda x: x.strftime(slider_format) if hasattr(x, 'strftime') else str(x),
                    value=slider_values[current_index],
                    key="time_slider",
                    on_change=on_time_change
                )
                
                st.markdown(
                    f"📅 **Mostrando datos para:** "
                    f"{selected_time.strftime(slider_format) if hasattr(selected_time, 'strftime') else selected_time}"
                )
                
                # Procesar datos para el tiempo seleccionado
                if "df_selected" not in st.session_state or st.session_state.need_refresh_map:
                    with st.spinner("Filtrando datos para la visualización..."):
                        df_selected = filtrar_datos_por_tiempo_seleccionado(
                            df, selected_time, st.session_state.config["granularity"]
                        )
                        st.session_state.df_selected = df_selected
                        st.session_state.need_refresh_map = False
                else:
                    df_selected = st.session_state.df_selected
                
                # Mostrar mapa con datos filtrados
                if not df_selected.empty:
                    import streamlit.components.v1 as components

                    try:
                        # Generar clave única para el mapa
                        mapa_key = f"mapa_{selected_time}_{st.session_state.config['nivel']}_{st.session_state.config['filtrar_outliers']}"
                        
                        # Siempre regenerar el mapa si hay cambios importantes
                        if mapa_key not in st.session_state or st.session_state.need_refresh_map:
                            with st.spinner("Generando mapa..."):
                                # Siempre partir del dataframe original para este tiempo
                                df_para_mapa = df_selected.copy()
                                
                                # Aplicar filtro SOLO si está activado
                                if st.session_state.config["filtrar_outliers"]:
                                    df_para_mapa = filtrar_outliers_para_visualizacion(df_para_mapa)
                                    st.info(f"Se han filtrado los valores extremos (outliers) para mejorar la visualización.")

                                # Crear mapa con los datos correctos
                                m = crear_mapa_con_heatmap(
                                    df_para_mapa, 
                                    global_min, 
                                    global_max, 
                                    st.session_state.config["nivel"]
                                )
                                
                                # Guardar en session_state y marcar como procesado
                                st.session_state[mapa_key] = m
                                st.session_state.need_refresh_map = False
                        else:
                            m = st.session_state[mapa_key]
                        
                        if m:
                            with st.container():
                                col1, col2 = st.columns([4, 1])

                                with col1:
                                    # Renderizar mapa
                                    folium_static(m, height=600)
                                
                                with col2:
                                    # Calcular y mostrar estadísticas
                                    stats_key = f"stats_{selected_time}_{st.session_state.config['filtrar_outliers']}"
                                    
                                    if stats_key not in st.session_state or st.session_state.need_refresh_map:
                                        # Siempre partir del dataframe original para este tiempo
                                        df_stats = df_selected.copy()
                                        
                                        # Aplicar filtro SOLO si está activado
                                        if st.session_state.config["filtrar_outliers"]:
                                            df_stats = filtrar_outliers_para_visualizacion(df_stats)
                                        
                                        avg_no2, max_no2, nivel, color = calcular_estadisticas_no2(df_stats)
                                        st.session_state[stats_key] = (avg_no2, max_no2, nivel, color)
                                    else:
                                        avg_no2, max_no2, nivel, color = st.session_state[stats_key]
                                    
                                    mostrar_estadisticas_basicas(avg_no2, max_no2, nivel, color)

                    except Exception as e:
                        st.error(f"Error al crear el mapa: {str(e)}")
                        st.info("Intenta con un rango de fechas diferente o una granularidad distinta.")
                else:
                    st.info("ℹ️ No hay datos disponibles para el momento seleccionado.")

        # Sección de estadísticas (fuera del contenedor principal)
        if st.session_state.config["show_stats"] and not df.empty:
            st.markdown("## 📊 Estadísticas")
            
            try:
                # Verificar si ya tenemos estadísticas generadas
                stats_key = f"estadisticas_{st.session_state.config['granularity']}_{st.session_state.config['fecha_inicio']}_{st.session_state.config['fecha_fin']}"
                
                if stats_key not in st.session_state or st.session_state.need_reprocess:
                    # Generar estadísticas temporales
                    stats_df = generar_estadisticas_temporales(df, slider_format)
                    st.session_state[stats_key] = stats_df
                else:
                    stats_df = st.session_state[stats_key]
                
                # Mostrar gráfico de evolución temporal
                mostrar_grafico_evolucion_temporal(stats_df, slider_format)

                try:
                    mostrar_descomposicion_serie_temporal(stats_df, slider_format)
                except Exception as e:
                    st.error(f"Error al mostrar la descomposición de la serie temporal: {str(e)}")
                    st.info("No se puede mostrar la descomposición de la serie temporal. Se deben tener dos ciclos completos de datos. Intenta con un rango de fechas diferente o una granularidad distinta.")

                # Gráfico del histograma
                hist_key = f"histograma_{st.session_state.config['filtrar_outliers']}"
                
                # Aplicar filtro de outliers al histograma si está activado
                if hist_key not in st.session_state or st.session_state.need_refresh_hist:
                    df_for_hist = df.copy()
                    if st.session_state.config["filtrar_outliers"]:
                        df_for_hist = filtrar_outliers_para_visualizacion(df_for_hist)
                    
                    st.session_state[hist_key] = df_for_hist
                    st.session_state.need_refresh_hist = False
                
                # Mostrar histograma
                mostrar_histograma_no2(st.session_state[hist_key])
                
            except Exception as e:
                st.error(f"Error al generar gráficos: {str(e)}")

        # Pie de página
        mostrar_pie_pagina(df_original["fecha"].max())


def inicializar_session_state() -> None:
    """
    Inicializa todas las variables de session_state necesarias.
    Esta función debe ejecutarse al principio del script.
    """
    # Variables de control de estado
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    
    if "config" not in st.session_state:
        st.session_state.config = {}
    
    # Flags para controlar cuándo regenerar datos y visualizaciones
    if "need_reprocess" not in st.session_state:
        st.session_state.need_reprocess = False
        
    if "need_refresh_map" not in st.session_state:
        st.session_state.need_refresh_map = False
        
    if "need_refresh_hist" not in st.session_state:
        st.session_state.need_refresh_hist = False
    
    # Inicializar variables de datos para evitar AttributeError
    if "df_original" not in st.session_state and st.session_state.data_loaded:
        st.session_state.data_loaded = False  # Forzar recarga de datos
    
    # Variables para almacenar dataframes
    if "df_original_date_filtered" not in st.session_state:
        st.session_state.df_original_date_filtered = None


def mostrar_seccion_timelapse_con_callbacks(
    df: pd.DataFrame,
    time_groups: List[datetime],
    slider_format: str,
    global_min: float,
    global_max: float
) -> None:
    """
    Muestra la sección para generar y descargar timelapse con callbacks.
    """
    st.markdown('<div class="sub-header">🎬 Timelapse de evolución temporal</div>', unsafe_allow_html=True)
    col1, _ = st.columns([1, 2])
    
    with col1:
        if st.button("🎬 Generar Timelapse", key="generate_timelapse"):
            try:
                # Aplicar filtrado si corresponde
                df_timelapse = df.copy()
                if st.session_state.config["filtrar_outliers"]:
                    df_timelapse = filtrar_outliers_para_visualizacion(df_timelapse)
                
                timelapse_data = generar_timelapse(
                    df_timelapse, 
                    time_groups, 
                    slider_format, 
                    global_min, 
                    global_max, 
                    modo=st.session_state.config["timelapse_format"], 
                    fps=st.session_state.config["fps"], 
                    nivel_contaminacion=st.session_state.config["nivel"]
                )
                
                if timelapse_data:
                    st.session_state.timelapse = timelapse_data
                    st.session_state.timelapse_format = st.session_state.config["timelapse_format"]
                    st.success("✅ Timelapse generado correctamente!")
                else:
                    st.error("❌ No se pudo generar el timelapse.")
            except Exception as e:
                st.error(f"Error al generar timelapse: {str(e)}")
                st.info("Asegúrate de tener instaladas las dependencias necesarias (Selenium o imgkit).")

        # Botón de descarga si hay timelapse generado
        if 'timelapse' in st.session_state and st.session_state.timelapse:
            filename = f"timelapse_no2_{st.session_state.config['fecha_inicio']}_{st.session_state.config['fecha_fin']}.{st.session_state.timelapse_format}"
            st.download_button(
                label=f"⬇️ Descargar {st.session_state.timelapse_format.upper()}",
                data=st.session_state.timelapse,
                file_name=filename,
                mime="image/gif" if st.session_state.timelapse_format == "gif" else "video/mp4"
            )
        else:
            st.info("Haz clic en 'Generar Timelapse' para crear una visualización animada de la evolución de NO₂ en el tiempo.")


def filtrar_outliers_para_visualizacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra valores extremos (outliers) para mejorar visualizaciones.
    
    Args:
        df: DataFrame con columna 'no2_value'
        
    Returns:
        pd.DataFrame: DataFrame sin outliers
    """
    # Calcular cuartiles Q1 y Q3
    q1 = df['no2_value'].quantile(0.01)
    q3 = df['no2_value'].quantile(0.99)
    
    # Calcular el rango intercuartil (IQR)
    iqr = q3 - q1
    
    # Definir límites para outliers usando el método del rango intercuartil
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filtrar y devolver el DataFrame
    filtered_df = df[(df['no2_value'] >= lower_bound) & (df['no2_value'] <= upper_bound)].copy()
    
    # Mostrar información sobre el filtrado
    st.info(f"Filtrando valores fuera del rango [{lower_bound:.2f}, {upper_bound:.2f}] μg/m³")
    
    return filtered_df



