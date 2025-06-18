import streamlit as st
import sys
import os
import folium
from streamlit_folium import folium_static
import pandas as pd

# Configuración de rutas y formato de números
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.2f}".format


from src.mapa_inicial_trafico_y_no2 import crear_mapa_trafico_y_no2_inicial

from src.mapa_asignaciones_trafico_y_no2 import (
    crear_mapa_sensores_asignados_a_cada_no2,
    mostrar_continuidad,
)

from src.analsis_no2 import generate_analisis_no2
from training import training_page as training_page_gam     
# from src.analisis_sensores_no2_y_trafico import analisis_sensores
# from train_xgboost_model import training_page as training_xgboost_page

def main() -> None:
    """
    Función principal de la app.
    """
    st.set_page_config(page_title="NO2 Sensors Map", layout="wide")
    st.title("Análisis del NO2 en relación al tráfico y meteorología")
    
    
    # Uso de pestañas para organizar la visualización de los mapas
    st.subheader("Visualización de Mapas")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Mapa NO2 + Tráfico",
        "Mapa de asignaciones NO2 + traffic sensor",
        "Análisis de NO2",
        "Análisis sensores de trafico + meteorología + NO2",
        "Entrenamiento del modelo GAM",
        "Entrenamiento del modelo XGBoost"  # Nueva pestaña
    ])

    # Solo cargar y renderizar el contenido de la pestaña activa
    with tab1:
        #if st.button("Cargar mapa de NO2 y Tráfico", key="load_map1"):
        with st.spinner("Cargando mapa..."):
            crear_mapa_trafico_y_no2_inicial()
        
        # if "map_1" in st.session_state:
        #     st.write("### Mapa NO2 y Tráfico")
        #     folium_static(st.session_state["map_1"])
        # else:
        #     st.info("Haz clic en el botón para cargar el mapa NO2 + Tráfico.")

    with tab2:
        col1, col2 = st.columns([1, 1])
          
        with col1:
            # if st.button("Cargar mapa de asignaciones", key="load_map2"):
            #     with st.spinner("Cargando mapa..."):
            st.session_state["map_2"], st.session_state["id_trafico_cercanos"] = crear_mapa_sensores_asignados_a_cada_no2()
            
            if "map_2" in st.session_state:
                st.write("### Mapa de asignaciones NO2 y sensores de tráfico con mayor número de datos")
                st.write("(Se han filtrado los sensores de aire que tengan al menos un sensor de tráfico a 200m) ")
                folium_static(st.session_state["map_2"])
            else:
                st.info("Haz clic en el botón para cargar el mapa de asignaciones.")

        with col2:
            # Solo mostrar el selector si los datos están disponibles
            if "id_trafico_cercanos" in st.session_state:
                st.write("### Todos los sensores cercanos al sensor NO2")
                sensor = st.selectbox(
                    "Seleccione un sensor para visualizar la continuidad temporal",
                    st.session_state["id_trafico_cercanos"]
                )
                if st.button("Mostrar continuidad", key="show_continuity"):
                    mostrar_continuidad(sensor)
            
        # with col2:
        #     if st.button("Cargar mapa de sensores continuos", key="load_map3"):
        #         with st.spinner("Cargando mapa..."):
        #             st.session_state["map_3"] = crear_mapa_sensores_asignados_a_cada_no2_continuo()
            
        #     if "map_3" in st.session_state:
        #         st.write("### Sensores de tráfico filtrados por tener la mayor continuidad")
        #         st.write("Todos los datos han sido previamente filtrados >= 2018.")
        #         folium_static(st.session_state["map_3"])
            # else:
            #     st.info("Haz clic en el botón para cargar el mapa de sensores continuos.")

    with tab3:
        generate_analisis_no2()

    # with tab4:
    #     analisis_sensores()

    with tab5:
        training_page_gam()

    # with tab6:
    #     training_xgboost_page()

if __name__ == "__main__":
    main()
