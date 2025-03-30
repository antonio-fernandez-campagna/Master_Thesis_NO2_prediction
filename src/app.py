import streamlit as st
import sys
import os
import folium
from streamlit_folium import folium_static
import pandas as pd

# Configuración de rutas y formato de números
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.2f}".format

from mapa_asignaciones_trafico_y_no2 import (
    crear_mapa_sensores_asignados_a_cada_no2,
    crear_mapa_sensores_asignados_a_cada_no2_continuo,
    mostrar_continuidad,
    #limpiar_cache
)

from src.analsis_no2 import generar_analisis_no2
from mapa_inicial_trafico_y_no2 import crear_mapa_trafico_y_no2_inicial

from analisis_sensores_no2_y_trafico import analisis_sensores

def main() -> None:
    """
    Función principal de la app.
    """
    st.set_page_config(page_title="NO2 Sensors Map", layout="wide")
    st.title("NO2 Sensors Map")
    
    
    # Uso de pestañas para organizar la visualización de los mapas
    st.subheader("Visualización de Mapas")
    tab1, tab2, tab3, tab4 = st.tabs(["Mapa NO2 + Tráfico", "Mapa de asignaciones NO2 + traffic sensor", "Análisis de NO2", "Análisis sensores de trafico"])

    # Solo cargar y renderizar el contenido de la pestaña activa
    with tab1:
        if st.button("Cargar mapa de NO2 y Tráfico", key="load_map1"):
            with st.spinner("Cargando mapa..."):
                st.session_state["map_1"] = crear_mapa_trafico_y_no2_inicial()
        
        if "map_1" in st.session_state:
            st.write("### Mapa NO2 y Tráfico")
            folium_static(st.session_state["map_1"])
        else:
            st.info("Haz clic en el botón para cargar el mapa NO2 + Tráfico.")

    with tab2:
        col1, col2 = st.columns([1, 1])
          
        with col1:
            if st.button("Cargar mapa de asignaciones", key="load_map2"):
                with st.spinner("Cargando mapa..."):
                    st.session_state["map_2"], st.session_state["id_trafico_cercanos"] = crear_mapa_sensores_asignados_a_cada_no2()
            
            if "map_2" in st.session_state:
                st.write("### Mapa de asignaciones NO2 y sensores de tráfico")
                st.write("Estos sensores han sido filtrados por estar a una distancia máxima de 200m")
                folium_static(st.session_state["map_2"])
            else:
                st.info("Haz clic en el botón para cargar el mapa de asignaciones.")
            
        with col2:
            if st.button("Cargar mapa de sensores continuos", key="load_map3"):
                with st.spinner("Cargando mapa..."):
                    st.session_state["map_3"] = crear_mapa_sensores_asignados_a_cada_no2_continuo()
            
            if "map_3" in st.session_state:
                st.write("### Sensores de tráfico filtrados por tener la mayor continuidad")
                st.write("Todos los datos han sido previamente filtrados >= 2018.")
                folium_static(st.session_state["map_3"])
            else:
                st.info("Haz clic en el botón para cargar el mapa de sensores continuos.")

        # Solo mostrar el selector si los datos están disponibles
        if "id_trafico_cercanos" in st.session_state:
            st.write("### Todos los sensores cercanos al sensor NO2")
            sensor = st.selectbox(
                "Seleccione un sensor para visualizar la continuidad temporal",
                st.session_state["id_trafico_cercanos"]
            )
            if st.button("Mostrar continuidad", key="show_continuity"):
                mostrar_continuidad(sensor)

    with tab3:
        generar_analisis_no2()

    with tab4:
        analisis_sensores()

if __name__ == "__main__":
    main()
