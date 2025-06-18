import folium
import pandas as pd
import random
import streamlit as st
import numpy as np
import folium
import leafmap.foliumap as leafmap
import imageio.v3
import os
import tempfile
from datetime import datetime, timedelta
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px
import altair as alt
import gc



@st.cache_data(ttl=3600)
def cargar_datos():
    """Carga y almacena en caché el mapeo entre sensores NO2 y tráfico"""
    #return pd.read_csv('data/super_processed/4_no2_to_traffic_sensor_mapping.csv')
    df = pd.read_parquet('data/super_processed/7_5_no2_with_1traffic_id.parquet')

    #drop duplicates
    #df = df.drop_duplicates(subset = ['id_no2'])
    
    return df

def cargar_datos_traffic_mapping():
    return pd.read_excel('data/super_processed/1_all_traffic_sensors.xlsx')



# def limpiar_cache():
#     """Limpia todo el caché de streamlit"""
#     cargar_mapping_no2_traffic.clear()
#     cargar_datos_traffic.clear()
#     crear_mapa_sensores_asignados_a_cada_no2.clear()
#     crear_mapa_sensores_asignados_a_cada_no2_continuo.clear()
#     gc.collect()  # Forzar recolección de basura


def crear_mapa_sensores_asignados_a_cada_no2():
    
    df = cargar_datos()
    df = df.copy().drop_duplicates(subset = ['id_no2'])

    df_traffic_mapping = cargar_datos_traffic_mapping()

    df_traffic_mapping['id_trafico'] = df_traffic_mapping['id_trafico'].astype(str)
    df['id_trafico'] = df['id_trafico'].astype(str)

    df = df.merge(df_traffic_mapping, on = 'id_trafico', how = 'left')
    df = df.rename(columns = {'latitud':'latitud_trafico', 'longitud':'longitud_trafico'})

    # Create a map centered around the average latitude and longitude
    map_center = [df['latitud_no2'].mean(), df['longitud_no2'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Generate a list of colors for the NO2 sensors
    colors = ['blue', 'green', 'orange', 'purple', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue']

    # Dictionary to hold the color for each NO2 sensor
    no2_colors = {}

    # Add markers for NO2 sensors with unique colors
    for _, row in df.iterrows():
        if row['id_no2'] not in no2_colors:
            # Assign a random color from the list
            color = random.choice(colors)
            no2_colors[row['id_no2']] = color
        
        # Add a circle marker for NO2 sensors
        folium.CircleMarker(
            location=[row['latitud_no2'], row['longitud_no2']],
            radius=12,
            #color=no2_colors[row['id_no2']],
            fill=True,
            fill_opacity=0.6,
            popup=f"NO2 Sensor ID: {row['id_no2']}",
            tooltip=f"NO2 Sensor ID: {row['id_no2']}"
        ).add_to(m)

    # Add markers for Traffic data with the same color as the corresponding NO2 sensor
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitud_trafico'], row['longitud_trafico']],
            popup=f"Traffic ID: {row['id_trafico']}, NO2 ID: {row['id_no2']}",
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10pt">🚦</div>'  # Use a small icon
            )
        ).add_to(m)

    # Save the map to an HTML file or display it directly
    unique = df.id_trafico.unique()
    
    return m, unique



# Función para mostrar la continuidad de datos
def mostrar_continuidad(sensor):
    with st.spinner("Cargando datos de continuidad..."):
            
        df = cargar_datos()
            
        # Filtrar y procesar solo los datos necesarios
        df = df[df['id_trafico'] == str(sensor)].copy()

        # Asegurarse que la columna 'fecha' sea de tipo datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Generar un rango horario completo
        fecha_inicio = df['fecha'].min()
        fecha_fin = df['fecha'].max()
        
        if fecha_inicio is not pd.NaT and fecha_fin is not pd.NaT:
            rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='H')
            
            # Crear DataFrame más eficiente solo con las columnas necesarias
            df_full = pd.DataFrame({'fecha': rango_fechas})
            # Usar merge en lugar de .isin() para mejor rendimiento
            df_full['dato_presente'] = df_full['fecha'].isin(df['fecha']).astype(int)
            
            # Crear un gráfico con Altair
            grafico = alt.Chart(df_full).mark_line(point=True).encode(
                x=alt.X('fecha:T', title='Fecha y Hora'),
                y=alt.Y('dato_presente:Q', 
                        scale=alt.Scale(domain=[0, 1]),
                        title='Presencia de datos (1: sí, 0: no)'),
                tooltip=['fecha:T', 'dato_presente:Q']
            ).properties(
                title='Continuidad de datos en el tiempo'
            )
            
            st.altair_chart(grafico, use_container_width=True)
        else:
            st.warning("No hay datos disponibles para este sensor.")

