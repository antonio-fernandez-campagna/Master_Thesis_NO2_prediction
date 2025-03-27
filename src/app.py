import streamlit as st
import sys
import os
import folium
from streamlit_folium import folium_static
import pandas as pd

# Configuración de rutas y formato de números
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.2f}".format

from src.crear_mapas import crear_mapa_trafico_y_no2_all, crear_mapa_sensores_asignados_a_cada_no2, crear_mapa_no2, crear_mapa_sensores_asignados_a_cada_no2_continuo, mostrar_continuidad

def main() -> None:
    """
    Función principal de la app.
    """
    st.set_page_config(page_title="NO2 Sensors Map", layout="wide")
    st.title("NO2 Sensors Map")

    st.session_state["map_1"] = crear_mapa_trafico_y_no2_all()
    st.session_state["map_2"], id_trafico_cercanos = crear_mapa_sensores_asignados_a_cada_no2()
    st.session_state["map_3"] = crear_mapa_sensores_asignados_a_cada_no2_continuo()


    # Uso de pestañas para organizar la visualización de los mapas
    st.subheader("Visualización de Mapas")
    tabs = st.tabs(["Mapa NO2 + Tráfico", "Mapa de asignaciones NO2 + traffic sensor", "Mapa de NO2"])

    with tabs[0]:
        if "map_1" in st.session_state:
            st.write("### Mapa NO2 y Tráfico")
            folium_static(st.session_state["map_1"])
        else:
            st.info("Aún no se ha cargado el mapa NO2 + Tráfico. Usa el botón en la barra lateral.")

    with tabs[1]:
        
        # crear dos columnas:
        col1, col2 = st.columns([1, 1])
          
        with col1:  
            if "map_2" in st.session_state:
                st.write("### Mapa de asignaciones NO2 y sensores de tráfico")
                st.write("Estos sensores han sido filtrados por estar a una distancia máxima de 200m")
                folium_static(st.session_state["map_2"])
            else:
                st.info("Aún no se ha cargado el segundo mapa. Usa el botón en la barra lateral.")  
            
        with col2:  
            if "map_3" in st.session_state:
                st.write("### Sensores de tráfico filtrados por tener la mayor cotinuidad")
                st.write("Todos los datos han sido previamente filtrados >= 2018.")
                folium_static(st.session_state["map_3"])
            else:
                st.info("Aún no se ha cargado el tercer mapa. Usa el botón en la barra lateral.")
        

            #mostrar un desplegable de los sensores id_trafico_cercanos
            st.write("### Todos los sensores cercanos al sensor NO2")
            
            # de la lista de sensores id_trafico_cercanos, id_trafico_continuo ha de salir resaltado.
            sensor = st.selectbox("Seleccione un sensor para visualizar la continuidad temporal", id_trafico_cercanos)
            
        
        mostrar_continuidad(sensor)
    
    with tabs[2]:
        crear_mapa_no2()

if __name__ == "__main__":
    main()
