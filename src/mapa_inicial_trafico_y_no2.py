import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from datetime import datetime


# Funciones de caché para cargar datos
@st.cache_data(ttl=3600)
def cargar_datos_no2_locations():
    return pd.read_csv('data/more_processed/no2_data_locations.csv')

@st.cache_data(ttl=3600)
def cargar_datos_traffic_locations():
    return pd.read_csv('data/more_processed/traffic_data_locations_2024.csv')

def crear_mapa_trafico_y_no2_simplificado():
    no2_data = cargar_datos_no2_locations()
    traffic_data = cargar_datos_traffic_locations()
    
    # Encontrar el centro óptimo para el mapa
    all_lats = pd.concat([no2_data["latitud"], traffic_data["latitud"]])
    all_lons = pd.concat([no2_data["longitud"], traffic_data["longitud"]])
    
    map_center = [all_lats.mean(), all_lons.mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")
    
    # Grupo de sensores NO2
    no2_group = folium.FeatureGroup(name="Sensores NO2")
    
    for _, row in no2_data.iterrows():
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=6,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=folium.Popup(
                f"""<div style='width: 200px'>
                    <b>Sensor NO2:</b> {row['id_no2']}<br>
                    <b>Ubicación:</b> {row['latitud']:.5f}, {row['longitud']:.5f}
                </div>""",
                max_width=300
            )
        ).add_to(no2_group)
    
    # Grupo de sensores de tráfico sin clusterización
    traffic_group = folium.FeatureGroup(name="Sensores Tráfico")
    
    for _, row in traffic_data.iterrows():
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=1,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.3,
            popup=folium.Popup(
                f"""<div style='width: 200px'>
                    <b>Sensor Tráfico:</b> {row['id_trafico']}<br>
                    <b>Ubicación:</b> {row['latitud']:.5f}, {row['longitud']:.5f}
                </div>""",
                max_width=300
            )
        ).add_to(traffic_group)
    
    # Agregar leyenda al mapa
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; 
                background-color:white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                ">
      <div style="margin-bottom: 5px;"><b>Leyenda</b></div>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="background-color: blue; width: 15px; height: 15px; 
                  border-radius: 50%; margin-right: 8px;"></div>
        <div>Sensores NO2</div>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: red; width: 15px; height: 15px; 
                  border-radius: 50%; margin-right: 8px;"></div>
        <div>Sensores Tráfico</div>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Agregar grupos y control de capas
    m.add_child(no2_group)
    m.add_child(traffic_group)
    m.add_child(folium.LayerControl())
    
    return m

def crear_mapa_trafico_y_no2_inicial():
    
    # Métricas básicas
    no2_data = cargar_datos_no2_locations()
    traffic_data = cargar_datos_traffic_locations()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sensores NO2", len(no2_data))
    
    with col2:
        st.metric("Sensores de Tráfico", len(traffic_data))
    
    with col3:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.metric("Última actualización", timestamp)
    
    # Mostrar el mapa
    st.subheader("Mapa de Sensores")
    mapa = crear_mapa_trafico_y_no2_simplificado()
    
    # Botón para actualizar el mapa
    if st.button("Actualizar Datos"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Mostrar el mapa
    folium_static(mapa, width=1200, height=600)
    

