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


# Funciones de caché y limpieza
@st.cache_data(ttl=3600)
def cargar_datos_no2_locations():
    """Carga y almacena en caché las ubicaciones de sensores NO2"""
    return pd.read_csv('data/more_processed/no2_data_locations.csv')

@st.cache_data(ttl=3600)
def cargar_datos_traffic_locations():
    """Carga y almacena en caché las ubicaciones de sensores de tráfico"""
    return pd.read_csv('data/more_processed/traffic_data_locations_2024.csv')

# Function to create a simple example map
def crear_mapa_trafico_y_no2_inicial():

    no2_data_locations = cargar_datos_no2_locations()
    traffic_data_locations = cargar_datos_traffic_locations()
    
    map_center = [no2_data_locations["latitud"].mean(), no2_data_locations["longitud"].mean()]
    m = folium.Map(location=map_center, zoom_start=11)

    # Add CircleMarkers for NO2 sensors
    for _, row in no2_data_locations.iterrows():
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=6,  # Size of the circle
            color='blue',  # Color for NO2 sensors
            fill=True,
            fill_color='blue',
            fill_opacity=0.9,
            #popup=f'Sensor NO2: {row["longitud"]}, {row["latitud"]}'
        ).add_to(m)

    # Add CircleMarkers for traffic sensors
    for _, row in traffic_data_locations.iterrows():
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=1,  # Size of the circle
            color='red',  # Color for traffic sensors
            fill=True,
            fill_color='red',
            fill_opacity=0.1,
            #popup=f'Sensor Traffic: {row["id_trafico"]}, {row["latitud"]}, {row["longitud"]}'
        ).add_to(m)

    return m