import folium
import pandas as pd
import random

from streamlit_folium import folium_static

import streamlit as st
import pandas as pd
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

# Function to create a simple example map
def crear_mapa_trafico_y_no2_all():
    
    no2_data_locations = pd.read_csv('data/more_processed/no2_data_locations.csv')
    traffic_data_locations = pd.read_csv('data/more_processed/traffic_data_locations_2024.csv')
    
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


def crear_mapa_sensores_asignados_a_cada_no2():
    
    
    df = pd.read_csv('data/more_processed/mapping_no2_y_traffic_filtered_by_proximity.csv')

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


def crear_mapa_sensores_asignados_a_cada_no2_continuo():
    
    df = pd.read_csv('data/more_processed/mapping_no2_y_traffic_filtered_by_proximity.csv')
    
    list_id_trafico_continuo = [5547, 5783, 5465, 5414, 5084, 4555, 4129, 3915, 3911]
    
    df = df[df['id_trafico'].isin(list_id_trafico_continuo)]

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
    return m


def mostrar_continuidad(sensor):
    
    import altair as alt
    
    df = pd.read_parquet('data/more_processed/traffic_data.parquet')
    
    df = df[df['id_trafico'] == str(sensor)]

    # Asegurarse que la columna 'fecha' sea de tipo datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Generar un rango horario completo (cada hora) desde la fecha mínima a la máxima
    fecha_inicio = df['fecha'].min()
    fecha_fin = df['fecha'].max()
    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='H')
    
    # Crear un DataFrame completo con el rango de fechas y marcar la presencia de datos
    df_full = pd.DataFrame({'fecha': rango_fechas})
    # True si la fecha existe en el dataset original, False en caso contrario
    df_full['dato_presente'] = df_full['fecha'].isin(df['fecha'])
    # Convertir a entero para graficar (1: presente, 0: faltante)
    df_full['dato_presente'] = df_full['dato_presente'].astype(int)
    
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

# def crear_mapa_no2():
#     """
#     Nuevo tab para analizar el valor de NO2.
#     Permite seleccionar un rango de fechas, la granularidad de análisis y navegar en el tiempo.
#     """
#     st.header("Análisis de valores NO₂")
    
#     # Cargar el dataset (asegúrate de que la ruta sea correcta)
#     df = pd.read_csv('data/more_processed/air_data.csv')

#     # Convert 'fecha' column to datetime
#     df['fecha'] = pd.to_datetime(df['fecha'])
    
#     # Selección de rango de fechas
#     col1, col2 = st.columns(2)
#     with col1:
#         fecha_inicio = st.date_input("Fecha de inicio", df["fecha"].min().date())
#     with col2:
#         fecha_fin = st.date_input("Fecha de fin", df["fecha"].max().date())
    
#     # Filtrar datos por el rango seleccionado
#     df = df[(df["fecha"].dt.date >= fecha_inicio) & (df["fecha"].dt.date <= fecha_fin)]
    
#     # Selección de granularidad
#     granularity = st.selectbox("Selecciona la granularidad", ["Horaria", "Mensual", "Anual"])
#     if granularity == "Horaria":
#         df["time_group"] = df["fecha"].dt.floor("H")
#         slider_format = "YYYY-MM-DD HH:mm"
#     elif granularity == "Mensual":
#         df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
#         slider_format = "YYYY-MM-DD"
#     elif granularity == "Anual":
#         df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
#         slider_format = "YYYY"
    
#     # Verificar si hay datos en el rango seleccionado
#     if df.empty:
#         st.error("No hay datos en el rango seleccionado.")
#         return
    
#     # Obtener intervalos únicos y ordenados
#     time_groups = sorted(df["time_group"].unique())
    
#     time_groups = [t.to_pydatetime() for t in time_groups]  # Convertir a datetime.datetime
    
    
#     print("TEST")
#     print("time_groups_str ", time_groups[0])
#     print("time_groups_str ", time_groups[-1])
#     print("time_groups_str ", time_groups[0])
    
#     # Slider para navegar en el tiempo
#     selected_time = st.slider(
#         "Selecciona el tiempo",
#         min_value=time_groups[0],
#         max_value=time_groups[-1],
#         value=time_groups[0],
#         format=slider_format
#     )
    
#     selected_time = pd.to_datetime(selected_time)
    
#     # Filtrar datos para el intervalo seleccionado
#     df_selected = df[df["time_group"] == selected_time]
#     st.write(f"Mostrando datos para: {selected_time}")
    
#     # Visualización en mapa
#     if not df_selected.empty:
#         map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]
#         m = folium.Map(location=map_center, zoom_start=12)
        
#         for _, row in df_selected.iterrows():
#             folium.CircleMarker(
#                 location=[row["latitud"], row["longitud"]],
#                 radius=6,
#                 color="blue",
#                 fill=True,
#                 fill_opacity=0.6,
#                 popup=f"Fecha: {row['fecha']}<br>NO₂: {row['no2_value']}"
#             ).add_to(m)
#         folium_static(m)
#     else:
#         st.info("No hay datos para el tiempo seleccionado.")

# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import folium_static
# import time
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import branca.colormap as bcm

# from folium.plugins import HeatMap
# import leafmap.foliumap as leafmap

# # Librerías adicionales
# import imageio
# import os

# def crear_mapa_con_heatmap(df_selected, global_min, global_max, slider_format):
#     """
#     Crea un mapa Folium (usando leafmap) centrado en la media de las coordenadas.
#     """
#     map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]

#     m = leafmap.Map(center=[map_center[0], map_center[1]], zoom=12)

#     m.add_heatmap(
#         data=df_selected,
#         latitude="latitud",
#         longitude="longitud",
#         value="no2_value",  
#         name="NO2 Heatmap",
#         radius=25,
#         blur=15
#     )
#     return m

# def generar_gif_o_video(df, time_groups, slider_format, global_min, global_max, modo="gif"):
#     """
#     Genera un GIF o video (mp4) a partir de mapas Folium en diferentes momentos de tiempo.
#     1. Itera sobre cada fecha/hora en time_groups.
#     2. Crea un mapa Folium con la capa de calor.
#     3. Usa m._to_png() para convertirlo en imagen (requiere Selenium).
#     4. Combina todas las imágenes en un GIF o MP4.
#     """
#     # Contendrá los frames (imágenes) en formato numpy array
#     frames = []

#     for t in time_groups:
        
#         df_t = df[df["time_group"] == t]
        
#         if df_t.empty:
#             continue
        
#         # Creamos el mapa
#         m = crear_mapa_con_heatmap(df_t, global_min, global_max, slider_format)
        
#         # Convertimos el mapa a PNG (requiere Selenium y un driver)
#         # Ajusta el factor de escala si quieres más resolución
#         png_data = m._to_png(5)  # <-- método "privado" de Folium
#         # Lo leemos como array de imagen con imageio
#         img = imageio.v3.imread(png_data)
#         frames.append(img)
        
#     if not frames:
#         st.warning("No se generaron frames (quizá no hay datos en el rango).")
#         return None

#     # Guardamos en disco
#     if modo == "gif":
#         output_path = "timelapse.gif"
#         imageio.mimsave(output_path, frames, fps=1)  # Ajusta fps
#         return output_path
#     else:
#         # Para mp4 se requiere ffmpeg instalado
#         output_path = "timelapse.mp4"
#         imageio.mimsave(output_path, frames, fps=1, codec="libx264")
#         return output_path

# def mostrar_timelapse_como_video(df, time_groups, slider_format, global_min, global_max):
#     """
#     En lugar de mostrar frame a frame en la web,
#     creamos un GIF o MP4 y luego lo mostramos.
#     """
#     st.subheader("Generar Timelapse como Video/GIF")

#     col1, _ = st.columns(2)
#     with col1:
#         modo = st.selectbox("Formato de salida", ["gif", "mp4"], index=0)
#         if st.button("Generar Timelapse"):
#             st.info("Creando frames, por favor espera...")

#             # Generar el GIF o MP4
#             video_path = generar_gif_o_video(df, time_groups, slider_format, global_min, global_max, modo=modo)

#             if video_path and os.path.exists(video_path):
#                 st.success(f"Timelapse generado: {video_path}")
#                 if modo == "gif":
#                     st.image(video_path)
#                 else:
#                     st.video(video_path)
#             else:
#                 st.error("No se pudo generar el video.")

# def crear_mapa_no2():
#     st.header("Análisis de niveles de NO₂ en Madrid")
    
#     # Cargar el dataset
#     df_original = pd.read_csv('data/more_processed/air_data.csv')
#     df_original['fecha'] = pd.to_datetime(df_original['fecha'])
    
#     global_min = df_original["no2_value"].min()
#     global_max = df_original["no2_value"].max()
    
#     col1, col2 = st.columns(2)
#     with col1:
#         fecha_inicio = st.date_input("Fecha de inicio", df_original["fecha"].min().date())
#     with col2:
#         fecha_fin = st.date_input("Fecha de fin", df_original["fecha"].max().date())
    
#     df = df_original[
#         (df_original["fecha"].dt.date >= fecha_inicio) & 
#         (df_original["fecha"].dt.date <= fecha_fin)
#     ].copy()
    

#     granularity = st.selectbox("Selecciona la granularidad", ["Horaria", "Mensual", "Anual"])
#     if granularity == "Horaria":
#         df["time_group"] = df["fecha"].dt.floor("H")
#         slider_format = "%Y-%m-%d %H:%M"
#     elif granularity == "Mensual":
#         df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
#         slider_format = "%Y-%m-%d"
#     else:  # Anual
#         df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
#         slider_format = "%Y"

        
#     if granularity in ["Mensual", "Anual"]:
#         df = df.groupby(["time_group", "latitud", "longitud"]).agg(
#             no2_value=("no2_value", "mean"),
#             fecha=("fecha", "min")
#         ).reset_index()
        
    
#     if df.empty:
#         st.error("No hay datos en el rango seleccionado.")
#         return
    
#     time_groups = sorted(df["time_group"].unique())
#     #time_groups = [t.to_pydatetime() for t in time_groups]
#     time_groups = [t.to_pydatetime().date() if granularity in ["Mensual", "Anual"] else t.to_pydatetime() for t in time_groups]

#     print("min ", time_groups[0])
#     print("max ", time_groups[-1])

#     # Slider para ver un instante específico
#     selected_time = st.slider(
#         "Selecciona el tiempo",
#         min_value=time_groups[0],
#         max_value=time_groups[-1],
#         value=time_groups[0],
#         format=slider_format
#     )
#     st.write(f"Mostrando datos para: {selected_time.strftime(slider_format)}")
    
#     #df_selected = df[df["time_group"] == selected_time]
#     if granularity == "Mensual":
#         df_selected = df[df["time_group"].dt.to_period("M") == pd.Period(selected_time, "M")]
#     elif granularity == "Anual":
#         df_selected = df[df["time_group"].dt.to_period("Y") == pd.Period(selected_time, "Y")]
#     else:
#         df_selected = df[df["time_group"] == selected_time]
    
#     if not df_selected.empty:
#         m = crear_mapa_con_heatmap(df_selected, global_min, global_max, slider_format)
#         folium_static(m)
#     else:
#         st.info("No hay datos para el tiempo seleccionado.")
    
#     # Llamamos a la función que genera el video/gif
#     mostrar_timelapse_como_video(df, time_groups, slider_format, global_min, global_max)

# # Si es tu script principal, descomenta:
# # if __name__ == "__main__":
# #     crear_mapa_no2()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import folium
# import leafmap.foliumap as leafmap
# import imageio.v3
# import os
# import tempfile
# from datetime import datetime, timedelta
# from folium.plugins import HeatMap
# from streamlit_folium import folium_static
# import plotly.express as px
# import altair as alt



# # Aplicar CSS personalizado para mejorar la apariencia
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1E88E5;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #424242;
#         margin-bottom: 1rem;
#     }
#     .info-box {
#         background-color: #E3F2FD;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .stSlider > div > div > div {
#         background-color: #1E88E5;
#     }
#     .stButton > button {
#         background-color: #1E88E5;
#         color: white;
#         border-radius: 0.5rem;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#     }
#     .css-12w0qpk {
#         background-color: #F5F5F5;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_data(ttl=3600, show_spinner=False)
# def cargar_datos():
#     """Carga y preprocesa los datos con caché para mejorar el rendimiento."""
#     df = pd.read_csv('data/more_processed/air_data.csv')
#     df['fecha'] = pd.to_datetime(df['fecha'])
#     return df

# def crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_contaminacion=None):
#     """
#     Crea un mapa Folium (usando leafmap) centrado en la media de las coordenadas
#     con opciones mejoradas de visualización.
    
#     Args:
#         df_selected: DataFrame con los datos seleccionados
#         global_min: Valor mínimo global de NO2 para normalizar la escala
#         global_max: Valor máximo global de NO2 para normalizar la escala
#         nivel_contaminacion: Filtro opcional para mostrar solo áreas con cierto nivel
#     """
#     if df_selected.empty:
#         return None
    
#     # Centro del mapa en el centro de Madrid o en la media de las coordenadas
#     if len(df_selected) > 0:
#         map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]
#     else:
#         map_center = [40.4168, -3.7038]  # Centro de Madrid por defecto
        
#     # Crear mapa base con opciones de basemaps
#     m = leafmap.Map(
#         center=map_center,
#         zoom=12,
#         tiles="CartoDB positron",
#         draw_control=False,
#         measure_control=False,
#         fullscreen_control=True
#     )
    
#     # Añadir controles de capas base
#     #m.add_basemap("CartoDB dark_matter", name="CartoDB Dark")
#     #m.add_basemap("OpenStreetMap", name="OpenStreetMap")
#     #m.add_basemap("HYBRID", name="Google Hybrid")
    
#     # Personalizar datos según nivel de contaminación si se especifica
#     if nivel_contaminacion:
#         if nivel_contaminacion == "Bajo":
#             df_selected = df_selected[df_selected['no2_value'] <= 40]
#         elif nivel_contaminacion == "Medio":
#             df_selected = df_selected[(df_selected['no2_value'] > 40) & (df_selected['no2_value'] <= 100)]
#         elif nivel_contaminacion == "Alto":
#             df_selected = df_selected[df_selected['no2_value'] > 100]
    
#     # Crear mapa de calor con parámetros optimizados
#     if not df_selected.empty:
#         # Configurar parámetros de heatmap según cantidad de datos
#         radius = 15 if len(df_selected) > 100 else 25
#         blur = 10 if len(df_selected) > 100 else 15
        
#         # Configuración de colores para el heatmap (de verde a rojo)
#         gradient = {
#             0.0: 'green',
#             0.3: 'yellow',
#             0.6: 'orange',
#             1.0: 'red'
#         }
        
#         # Normalizar los valores para mejor visualización
#         heat_data = [[row['latitud'], row['longitud'], 
#                       max(0.1, min(1, (row['no2_value'] - global_min) / (global_max - global_min) * 0.8 + 0.2))] 
#                      for _, row in df_selected.iterrows()]
        
#         # Añadir el heatmap al mapa
#         HeatMap(
#             heat_data,
#             name="NO₂ Heatmap",
#             radius=radius,
#             blur=blur,
#             gradient=gradient,
#             show=True,
#             overlay=True,
#             control=True,
#             min_opacity=0.5
#         ).add_to(m)
        
#         # Añadir leyenda
#         m.add_legend(title="Niveles de NO₂", legend_dict={
#             "Bajo (<40 μg/m³)": "green",
#             "Medio (40-100 μg/m³)": "yellow",
#             "Alto (>100 μg/m³)": "red"
#         })
        
#     return m

# def generar_timelapse(df, time_groups, slider_format, global_min, global_max, 
#                       modo="gif", fps=2, nivel_contaminacion=None):
#     """
#     Genera un GIF o video (mp4) a partir de mapas con datos de diferentes momentos.
    
#     Args:
#         df: DataFrame con todos los datos
#         time_groups: Lista de grupos temporales únicos
#         slider_format: Formato para mostrar las fechas
#         global_min, global_max: Valores globales mín/máx para normalizar
#         modo: 'gif' o 'mp4'
#         fps: Frames por segundo
#         nivel_contaminacion: Filtro para nivel de contaminación
#     """
#     with st.spinner('Generando timelapse... Este proceso puede tardar unos minutos.'):
#         frames = []
#         temp_frames = []
#         temp_dir = tempfile.mkdtemp()
        
#         total_frames = len(time_groups)
#         progress_bar = st.progress(0)
        
#         for i, t in enumerate(time_groups):
#             # Actualizar barra de progreso
#             progress_bar.progress((i + 1) / total_frames)
            
#             # Filtrar datos para este momento
#             if isinstance(t, datetime):
#                 if isinstance(df["time_group"].iloc[0], datetime):
#                     # Si ambos son datetime, comparar directamente
#                     df_t = df[df["time_group"] == t]
#                 else:
#                     # Convertir la columna time_group a datetime para comparar
#                     df_t = df[pd.to_datetime(df["time_group"]) == t]
#             else:
#                 # Para comparaciones con fechas o periodos
#                 time_format = "%Y-%m" if slider_format == "%Y-%m" else "%Y-%m-%d" if "-%d" in slider_format else "%Y"
#                 t_str = t.strftime(time_format) if hasattr(t, 'strftime') else str(t)
                
#                 if "time_group_str" not in df.columns:
#                     df["time_group_str"] = df["time_group"].dt.strftime(time_format)
                
#                 df_t = df[df["time_group_str"] == t_str]
            
#             if df_t.empty:
#                 continue
            
#             # Crear mapa para este momento
#             m = crear_mapa_con_heatmap(df_t, global_min, global_max, nivel_contaminacion)
            
#             if m is None:
#                 continue
                
#             # Añadir título con timestamp
#             title_html = f'''
#                 <h3 style="position:absolute;z-index:1000;left:50px;top:10px;background-color:white;
#                 padding:5px;border-radius:5px;box-shadow:0 0 5px rgba(0,0,0,0.3);">
#                 {t.strftime(slider_format) if hasattr(t, 'strftime') else str(t)}
#                 </h3>
#             '''
#             m.get_root().html.add_child(folium.Element(title_html))
            
#             # Guardar cada mapa como imagen temporal
#             temp_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
#             temp_frames.append(temp_file)
            
#             try:
#                 # Convertir mapa a imagen
#                 m.save(temp_file.replace('.png', '.html'))
#                 # Usar un método seguro para convertir HTML a imagen
#                 from selenium import webdriver
#                 from selenium.webdriver.chrome.options import Options
#                 from selenium.webdriver.chrome.service import Service
#                 from webdriver_manager.chrome import ChromeDriverManager
                
#                 options = Options()
#                 options.add_argument("--headless")
#                 options.add_argument("--no-sandbox")
#                 options.add_argument("--disable-dev-shm-usage")
                
#                 driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#                 driver.set_window_size(1280, 1024)
#                 driver.get('file://' + temp_file.replace('.png', '.html'))
#                 driver.save_screenshot(temp_file)
#                 driver.quit()
                
#                 img = imageio.v3.imread(temp_file)
#                 frames.append(img)
#             except Exception as e:
#                 st.error(f"Error al generar frame {i}: {str(e)}")
#                 continue
        
#         progress_bar.empty()
        
#         if not frames:
#             st.warning("No se pudieron generar frames para el timelapse.")
#             return None
            
#         # Guardar archivo final
#         output_file = os.path.join(temp_dir, f"timelapse.{modo}")
        
#         try:
#             if modo == "gif":
#                 imageio.v3.imwrite(output_file, frames, loop=0, fps=fps)
#             else:  # mp4
#                 imageio.v3.imwrite(output_file, frames, fps=fps, codec='h264', output_params=['-pix_fmt', 'yuv420p'])
                
#             # Leer el archivo para devolverlo
#             with open(output_file, 'rb') as file:
#                 return file.read()
#         except Exception as e:
#             st.error(f"Error al guardar el timelapse: {str(e)}")
#             return None
#         finally:
#             # Limpiar archivos temporales
#             for f in temp_frames:
#                 if os.path.exists(f):
#                     try:
#                         os.remove(f)
#                         os.remove(f.replace('.png', '.html'))
#                     except:
#                         pass

# def crear_mapa_no2():
#     """Función principal para la visualización de datos de NO₂ en Madrid."""
#     # Encabezado con diseño mejorado
#     st.markdown('<div class="main-header">🌍 Análisis de niveles de NO₂ en Madrid</div>', unsafe_allow_html=True)
    
#     # Información contextual
#     with st.expander("ℹ️ Acerca de este dashboard", expanded=False):
#         st.markdown("""
#         <div class="info-box">
#         <p>Este dashboard permite analizar la evolución temporal de los niveles de NO₂ (dióxido de nitrógeno) en Madrid a distintas escalas temporales.</p>
#         <p><strong>Cómo usar:</strong></p>
#         <ul>
#             <li>Selecciona un rango de fechas para filtrar los datos</li>
#             <li>Elige la granularidad temporal (horaria, diaria, semanal, mensual o anual)</li>
#             <li>Utiliza el slider para visualizar la concentración de NO₂ en momentos específicos</li>
#             <li>Puedes generar timelapses para observar la evolución en el tiempo</li>
#             <li>Explora las estadísticas complementarias en los gráficos inferiores</li>
#         </ul>
#         <p><strong>Nota:</strong> El NO₂ es un contaminante asociado principalmente al tráfico rodado. La OMS recomienda no superar los 40 μg/m³ de media anual.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Cargar datos
#     with st.spinner('Cargando datos...'):
#         df_original = cargar_datos()
    
#     # Obtener los valores mínimos y máximos globales para la normalización
#     global_min = df_original["no2_value"].min()
#     global_max = df_original["no2_value"].max()
    
#     # Sidebar para controles de filtrado
#     with st.sidebar:
#         st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        
#         # Filtro de fechas
#         fecha_min = df_original["fecha"].min().date()
#         fecha_max = df_original["fecha"].max().date()
        
#         st.markdown("#### 📅 Rango de fechas")
#         fecha_inicio = st.date_input("Fecha inicial", 
#                                     fecha_min,
#                                     min_value=fecha_min,
#                                     max_value=fecha_max)
#         fecha_fin = st.date_input("Fecha final", 
#                                 fecha_max,
#                                 min_value=fecha_min,
#                                 max_value=fecha_max)
        
#         # Verificar que el rango sea válido
#         if fecha_inicio > fecha_fin:
#             st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
#             fecha_fin = fecha_inicio + timedelta(days=7)
        
#         # Granularidad temporal con más opciones
#         st.markdown("#### ⏱️ Agregación temporal")
#         granularity = st.radio(
#             "Selecciona la granularidad",
#             ["Horaria", "Diaria", "Semanal", "Mensual", "Anual"],
#             horizontal=True
#         )
        
#         # Filtro por nivel de contaminación
#         st.markdown("#### 🔍 Filtro de visualización")
#         nivel_contaminacion = st.selectbox(
#             "Mostrar áreas con nivel de contaminación:",
#             ["Todos", "Bajo", "Medio", "Alto"]
#         )
#         nivel_seleccionado = None if nivel_contaminacion == "Todos" else nivel_contaminacion
        
#         # Configuración de visualización del mapa
#         st.markdown("#### 🔆 Opciones de visualización")
#         show_stats = st.checkbox("Mostrar estadísticas", value=True)
        
#         # Opciones de timelapse
#         st.markdown("#### 🎬 Configuración de timelapse")
#         timelapse_format = st.radio("Formato de salida", ["gif", "mp4"], horizontal=True)
#         fps = st.slider("Velocidad (fps)", min_value=1, max_value=10, value=2)
    
#     # Filtrar datos según el rango de fechas seleccionado
#     df = df_original[
#         (df_original["fecha"].dt.date >= fecha_inicio) & 
#         (df_original["fecha"].dt.date <= fecha_fin)
#     ].copy()
    
#     if df.empty:
#         st.error("⚠️ No hay datos disponibles para el rango de fechas seleccionado.")
#         return
        
#     # Configurar la granularidad temporal
#     if granularity == "Horaria":
#         df["time_group"] = df["fecha"].dt.floor("H")
#         slider_format = "%Y-%m-%d %H:%M"
#     elif granularity == "Diaria":
#         df["time_group"] = df["fecha"].dt.floor("D")
#         slider_format = "%Y-%m-%d"
#     elif granularity == "Semanal":
#         df["time_group"] = df["fecha"].dt.to_period("W").dt.to_timestamp()
#         slider_format = "%Y-%m-%d"
#     elif granularity == "Mensual":
#         df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
#         slider_format = "%Y-%m"
#     else:  # Anual
#         df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
#         slider_format = "%Y"
        
#     # Agregar datos según granularidad seleccionada
#     if granularity != "Horaria":
#         df = df.groupby(["time_group", "latitud", "longitud"]).agg(
#             no2_value=("no2_value", "mean"),
#             fecha=("fecha", "min")
#         ).reset_index()
        
#     # Obtener los grupos de tiempo únicos
#     time_groups = sorted(df["time_group"].unique())
    
#     if len(time_groups) == 0:
#         st.error("⚠️ No hay suficientes datos para la granularidad seleccionada.")
#         return
    
#     # Organizar el layout
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown('<div class="sub-header">🗺️ Mapa de concentraciones de NO₂</div>', unsafe_allow_html=True)
        
#         # Slider para seleccionar el tiempo
#         # Convertir los time_groups a formato datetime para el slider
#         slider_values = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in time_groups]
        
#         selected_time = st.select_slider(
#             "Selecciona el momento temporal",
#             options=slider_values,
#             format_func=lambda x: x.strftime(slider_format) if hasattr(x, 'strftime') else str(x),
#             value=slider_values[0]
#         )
        
#         # Mostrar la fecha seleccionada
#         st.markdown(f"📅 **Mostrando datos para:** {selected_time.strftime(slider_format) if hasattr(selected_time, 'strftime') else selected_time}")
        
#         # Filtrar DataFrame para el tiempo seleccionado
#         if granularity == "Mensual":
#             df_selected = df[df["time_group"].dt.to_period("M") == pd.Period(selected_time, "M")]
#         elif granularity == "Anual":
#             df_selected = df[df["time_group"].dt.to_period("Y") == pd.Period(selected_time, "Y")]
#         else:
#             # Para horaria, diaria o semanal
#             df_selected = df[df["time_group"] == selected_time]
        
#         # Crear y mostrar el mapa
#         if not df_selected.empty:
#             m = crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_seleccionado)
#             if m:
#                 folium_static(m, width=700, height=500)
                
#                 # Mostrar estadísticas básicas
#                 avg_no2 = df_selected["no2_value"].mean()
#                 max_no2 = df_selected["no2_value"].max()
                
#                 # Evaluar el nivel según límites recomendados
#                 if avg_no2 <= 40:
#                     nivel = "Bajo"
#                     color = "green"
#                 elif avg_no2 <= 100:
#                     nivel = "Medio"
#                     color = "orange"
#                 else:
#                     nivel = "Alto"
#                     color = "red"
                
#                 st.markdown(f"""
#                 <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
#                         <div style="font-size: 1.5rem; color: {color};">{avg_no2:.1f} μg/m³</div>
#                     </div>
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Máximo NO₂</div>
#                         <div style="font-size: 1.5rem;">{max_no2:.1f} μg/m³</div>
#                     </div>
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Nivel</div>
#                         <div style="font-size: 1.5rem; color: {color};">{nivel}</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.info("ℹ️ No hay datos disponibles para el momento seleccionado.")
    
#     with col2:
#         st.markdown('<div class="sub-header">📊 Estadísticas</div>', unsafe_allow_html=True)
        
#         if show_stats and not df.empty:
#             # Preparar datos para gráficos
#             stats_df = df.groupby("time_group").agg(
#                 no2_promedio=("no2_value", "mean"),
#                 no2_max=("no2_value", "max"),
#                 num_readings=("no2_value", "count")
#             ).reset_index()
            
#             # Añadir formato de fecha para mostrar
#             stats_df["fecha_str"] = stats_df["time_group"].dt.strftime(slider_format)
            
#             # Gráfico de evolución temporal
#             st.write("**Evolución temporal de NO₂**")
#             line_chart = alt.Chart(stats_df).mark_line(point=True).encode(
#                 x=alt.X('time_group:T', title='Fecha'),
#                 y=alt.Y('no2_promedio:Q', title='NO₂ promedio (μg/m³)'),
#                 tooltip=['fecha_str', 'no2_promedio', 'no2_max', 'num_readings']
#             ).properties(height=200)
            
#             # Añadir línea horizontal para el límite recomendado
#             limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(
#                 color='red', strokeDash=[3, 3]
#             ).encode(y='y')
            
#             st.altair_chart(line_chart + limit_line, use_container_width=True)
            
#             # Histograma de valores
#             st.write("**Distribución de valores de NO₂**")
#             hist = alt.Chart(df).mark_bar().encode(
#                 x=alt.X('no2_value:Q', bin=alt.Bin(maxbins=20), title='NO₂ (μg/m³)'),
#                 y=alt.Y('count()', title='Frecuencia')
#             ).properties(height=150)
            
#             st.altair_chart(hist, use_container_width=True)
    
#     # Sección de generación de timelapse
#     st.markdown('<div class="sub-header">🎬 Timelapse de evolución temporal</div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         if st.button("🎬 Generar Timelapse", key="generate_timelapse"):
#             # Generar el timelapse
#             timelapse_data = generar_timelapse(
#                 df, time_groups, slider_format, global_min, global_max, 
#                 modo=timelapse_format, fps=fps, nivel_contaminacion=nivel_seleccionado
#             )
            
#             if timelapse_data:
#                 # Guardar el timelapse en la sesión
#                 st.session_state.timelapse = timelapse_data
#                 st.session_state.timelapse_format = timelapse_format
#                 st.success("✅ Timelapse generado correctamente!")
#             else:
#                 st.error("❌ No se pudo generar el timelapse.")
    
#     with col2:
#         # Mostrar el timelapse si existe en la sesión
#         if 'timelapse' in st.session_state and st.session_state.timelapse:
#             if st.session_state.timelapse_format == "gif":
#                 st.image(st.session_state.timelapse, caption="Timelapse de evolución de NO₂")
#             else:
#                 st.video(st.session_state.timelapse)
            
#             # Botón para descargar
#             filename = f"timelapse_no2_{fecha_inicio}_{fecha_fin}.{st.session_state.timelapse_format}"
#             st.download_button(
#                 label=f"⬇️ Descargar {st.session_state.timelapse_format.upper()}",
#                 data=st.session_state.timelapse,
#                 file_name=filename,
#                 mime=f"image/{st.session_state.timelapse_format}" if st.session_state.timelapse_format == "gif" else "video/mp4"
#             )
#         else:
#             st.info("Haz clic en 'Generar Timelapse' para crear una visualización animada de la evolución de NO₂ en el tiempo.")
            
#     # Pie de página
#     st.markdown("""
#     <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
#         Datos proporcionados por el Ayuntamiento de Madrid. Última actualización: {}.
#     </div>
#     """.format(df_original["fecha"].max().strftime("%d/%m/%Y")), unsafe_allow_html=True)

# def crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_contaminacion=None):
#     """
#     Crea un mapa Folium (usando leafmap) centrado en la media de las coordenadas
#     con opciones mejoradas de visualización.
    
#     Args:
#         df_selected: DataFrame con los datos seleccionados
#         global_min: Valor mínimo global de NO2 para normalizar la escala
#         global_max: Valor máximo global de NO2 para normalizar la escala
#         nivel_contaminacion: Filtro opcional para mostrar solo áreas con cierto nivel
#     """
#     if df_selected.empty:
#         return None
    
#     # Centro del mapa en el centro de Madrid o en la media de las coordenadas
#     if len(df_selected) > 0:
#         map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]
#     else:
#         map_center = [40.4168, -3.7038]  # Centro de Madrid por defecto
        
#     # Crear mapa base
#     m = leafmap.Map(
#         center=map_center,
#         zoom=12,
#         tiles="CartoDB positron",
#         draw_control=False,
#         measure_control=False,
#         fullscreen_control=True
#     )
    
#     # Añadir controles de capas base (solo usando los disponibles en leafmap)
#     #m.add_basemap("OpenStreetMap", name="OpenStreetMap")
#     #m.add_basemap("CartoDB dark_matter", name="CartoDB Dark")
    
#     # Personalizar datos según nivel de contaminación si se especifica
#     if nivel_contaminacion:
#         if nivel_contaminacion == "Bajo":
#             df_selected = df_selected[df_selected['no2_value'] <= 40]
#         elif nivel_contaminacion == "Medio":
#             df_selected = df_selected[(df_selected['no2_value'] > 40) & (df_selected['no2_value'] <= 100)]
#         elif nivel_contaminacion == "Alto":
#             df_selected = df_selected[df_selected['no2_value'] > 100]
    
#     # Crear mapa de calor con parámetros optimizados
#     if not df_selected.empty:
#         # Configurar parámetros de heatmap según cantidad de datos
#         radius = 15 if len(df_selected) > 100 else 25
#         blur = 10 if len(df_selected) > 100 else 15
        
#         # Configuración de colores para el heatmap (de verde a rojo)
#         gradient = {
#             0.0: 'green',
#             0.3: 'yellow',
#             0.6: 'orange',
#             1.0: 'red'
#         }
        
#         # Normalizar los valores para mejor visualización
#         heat_data = [[row['latitud'], row['longitud'], 
#                       max(0.1, min(1, (row['no2_value'] - global_min) / (global_max - global_min) * 0.8 + 0.2))] 
#                      for _, row in df_selected.iterrows()]
        
#         # Añadir el heatmap al mapa
#         HeatMap(
#             heat_data,
#             name="NO₂ Heatmap",
#             radius=radius,
#             blur=blur,
#             gradient=gradient,
#             show=True,
#             overlay=True,
#             control=True,
#             min_opacity=0.5
#         ).add_to(m)
        
#         # Añadir leyenda
#         m.add_legend(title="Niveles de NO₂", legend_dict={
#             "Bajo (<40 μg/m³)": "green",
#             "Medio (40-100 μg/m³)": "yellow",
#             "Alto (>100 μg/m³)": "red"
#         })
        
#     return m


# # Aplicar CSS personalizado para mejorar la apariencia
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1E88E5;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #424242;
#         margin-bottom: 1rem;
#     }
#     .info-box {
#         background-color: #E3F2FD;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .stSlider > div > div > div {
#         background-color: #1E88E5;
#     }
#     .stButton > button {
#         background-color: #1E88E5;
#         color: white;
#         border-radius: 0.5rem;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#     }
#     .css-12w0qpk {
#         background-color: #F5F5F5;
#     }
# </style>
# """, unsafe_allow_html=True)




@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos():
    """Carga y preprocesa los datos con caché para mejorar el rendimiento."""
    df = pd.read_csv('data/more_processed/air_data.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_contaminacion=None):
    """
    Crea un mapa Folium (usando leafmap) centrado en la media de las coordenadas.
    """
    map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]

    m = leafmap.Map(center=[map_center[0], map_center[1]], zoom=12)
    
    m = leafmap.Map(
        center=map_center,
        zoom=12,
        tiles="CartoDB positron",
        draw_control=False,
        measure_control=False,
        fullscreen_control=True
    )
    
    # Personalizar datos según nivel de contaminación si se especifica
    if nivel_contaminacion:
        if nivel_contaminacion == "Bajo":
            df_selected = df_selected[df_selected['no2_value'] <= 40]
        elif nivel_contaminacion == "Medio":
            df_selected = df_selected[(df_selected['no2_value'] > 40) & (df_selected['no2_value'] <= 100)]
        elif nivel_contaminacion == "Alto":
            df_selected = df_selected[df_selected['no2_value'] > 100]

    #Crear mapa de calor con parámetros optimizados
    if not df_selected.empty:
        # Configurar parámetros de heatmap según cantidad de datos
        radius = 15 if len(df_selected) > 100 else 25
        blur = 10 if len(df_selected) > 100 else 15
    
        
        # Normalizar los valores para mejor visualización
        heat_data = [[row['latitud'], row['longitud'], 
                      max(0.1, min(1, (row['no2_value'] - global_min) / (global_max - global_min) * 0.8 + 0.2))] 
                     for _, row in df_selected.iterrows()]
        
        m.add_heatmap(
            data=heat_data,
            latitude="latitud",
            longitude="longitud",
            value="no2_value",  
            name="NO2 Heatmap",
            radius=radius,
            blur=blur,
        )
        
        # # Añadir leyenda
        # m.add_legend(title="Niveles de NO₂", legend_dict={
        #     "Bajo (<40 μg/m³)": "green",
        #     "Medio (40-100 μg/m³)": "yellow",
        #     "Alto (>100 μg/m³)": "red"
        # })
    
    return m

def generar_timelapse(df, time_groups, slider_format, global_min, global_max, 
                      modo="gif", fps=2, nivel_contaminacion=None):
    """
    Genera un GIF o video (mp4) a partir de mapas con datos de diferentes momentos.
    
    Args:
        df: DataFrame con todos los datos
        time_groups: Lista de grupos temporales únicos
        slider_format: Formato para mostrar las fechas
        global_min, global_max: Valores globales mín/máx para normalizar
        modo: 'gif' o 'mp4'
        fps: Frames por segundo
        nivel_contaminacion: Filtro para nivel de contaminación
    """
    with st.spinner('Generando timelapse... Este proceso puede tardar unos minutos.'):
        frames = []
        temp_frames = []
        temp_dir = tempfile.mkdtemp()
        
        total_frames = len(time_groups)
        progress_bar = st.progress(0)
        
        for i, t in enumerate(time_groups):
            # Actualizar barra de progreso
            progress_bar.progress((i + 1) / total_frames)
            
            # Filtrar datos para este momento
            if isinstance(t, datetime):
                if isinstance(df["time_group"].iloc[0], datetime):
                    # Si ambos son datetime, comparar directamente
                    df_t = df[df["time_group"] == t]
                else:
                    # Convertir la columna time_group a datetime para comparar
                    df_t = df[pd.to_datetime(df["time_group"]) == t]
            else:
                # Para comparaciones con fechas o periodos
                time_format = "%Y-%m" if slider_format == "%Y-%m" else "%Y-%m-%d" if "-%d" in slider_format else "%Y"
                t_str = t.strftime(time_format) if hasattr(t, 'strftime') else str(t)
                
                if "time_group_str" not in df.columns:
                    df["time_group_str"] = df["time_group"].dt.strftime(time_format)
                
                df_t = df[df["time_group_str"] == t_str]
            
            if df_t.empty:
                continue
            
            # Crear mapa para este momento
            m = crear_mapa_con_heatmap(df_t, global_min, global_max, nivel_contaminacion)
            
            if m is None:
                continue
                
            # Añadir título con timestamp
            title_html = f'''
                <h3 style="position:absolute;z-index:1000;left:50px;top:10px;background-color:white;
                padding:5px;border-radius:5px;box-shadow:0 0 5px rgba(0,0,0,0.3);">
                {t.strftime(slider_format) if hasattr(t, 'strftime') else str(t)}
                </h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Guardar cada mapa como imagen temporal
            temp_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
            temp_frames.append(temp_file)
            
            try:
                # Usar método más seguro para generar imágenes
                m.save(temp_file.replace('.png', '.html'))
                
                # Intentar usar método alternativo para convertir HTML a PNG
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
                    import imgkit
                    imgkit.from_file(temp_file.replace('.png', '.html'), temp_file)
                
                img = imageio.v3.imread(temp_file)
                frames.append(img)
            except Exception as e:
                st.error(f"Error al generar frame {i}: {str(e)}")
                continue
        
        progress_bar.empty()
        
        if not frames:
            st.warning("No se pudieron generar frames para el timelapse.")
            return None
            
        # Guardar archivo final
        output_file = os.path.join(temp_dir, f"timelapse.{modo}")
        
        try:
            if modo == "gif":
                imageio.v3.imwrite(output_file, frames, loop=0, fps=fps)
            else:  # mp4
                imageio.v3.imwrite(output_file, frames, fps=fps, codec='h264', output_params=['-pix_fmt', 'yuv420p'])
                
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

# def crear_mapa_no2():
#     """Función principal para la visualización de datos de NO₂ en Madrid."""
#     # Encabezado con diseño mejorado
#     #st.markdown('<div class="main-header">🌍 Análisis de niveles de NO₂ en Madrid</div>', unsafe_allow_html=True)
    
#     # Información contextual
#     with st.expander("ℹ️ Acerca de este dashboard", expanded=False):
#         st.markdown("""
#         <div class="info-box">
#         <p>Este dashboard permite analizar la evolución temporal de los niveles de NO₂ (dióxido de nitrógeno) en Madrid a distintas escalas temporales.</p>
#         <p><strong>Cómo usar:</strong></p>
#         <ul>
#             <li>Selecciona un rango de fechas para filtrar los datos</li>
#             <li>Elige la granularidad temporal (horaria, diaria, semanal, mensual o anual)</li>
#             <li>Utiliza el slider para visualizar la concentración de NO₂ en momentos específicos</li>
#             <li>Puedes generar timelapses para observar la evolución en el tiempo</li>
#             <li>Explora las estadísticas complementarias en los gráficos inferiores</li>
#         </ul>
#         <p><strong>Nota:</strong> El NO₂ es un contaminante asociado principalmente al tráfico rodado. La OMS recomienda no superar los 40 μg/m³ de media anual.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Cargar datos
#     with st.spinner('Cargando datos...'):
#         try:
#             df_original = cargar_datos()
#         except Exception as e:
#             st.error(f"Error al cargar los datos: {str(e)}")
#             st.info("Asegúrate de que el archivo 'data/more_processed/air_data.csv' existe y es accesible.")
#             return
    
#     # Obtener los valores mínimos y máximos globales para la normalización
#     global_min = df_original["no2_value"].min()
#     global_max = df_original["no2_value"].max()
    
#     # Sidebar para controles de filtrado
#     with st.sidebar:
#         st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        
#         # Filtro de fechas
#         fecha_min = df_original["fecha"].min().date()
#         fecha_max = df_original["fecha"].max().date()
        
#         st.markdown("#### 📅 Rango de fechas")
#         fecha_inicio = st.date_input("Fecha inicial", 
#                                     fecha_min,
#                                     min_value=fecha_min,
#                                     max_value=fecha_max)
#         fecha_fin = st.date_input("Fecha final", 
#                                 fecha_max,
#                                 min_value=fecha_min,
#                                 max_value=fecha_max)
        
#         # Verificar que el rango sea válido
#         if fecha_inicio > fecha_fin:
#             st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
#             fecha_fin = fecha_inicio + timedelta(days=7)
        
#         # Granularidad temporal con más opciones
#         st.markdown("#### ⏱️ Agregación temporal")
#         granularity = st.radio(
#             "Selecciona la granularidad",
#             ["Horaria", "Diaria", "Semanal", "Mensual", "Anual"],
#             horizontal=True
#         )
        
#         # Filtro por nivel de contaminación
#         st.markdown("#### 🔍 Filtro de visualización")
#         nivel_contaminacion = st.selectbox(
#             "Mostrar áreas con nivel de contaminación:",
#             ["Todos", "Bajo", "Medio", "Alto"]
#         )
#         nivel_seleccionado = None if nivel_contaminacion == "Todos" else nivel_contaminacion
        
#         # Configuración de visualización del mapa
#         st.markdown("#### 🔆 Opciones de visualización")
#         show_stats = st.checkbox("Mostrar estadísticas", value=True)
        
#         # Opciones de timelapse
#         st.markdown("#### 🎬 Configuración de timelapse")
#         timelapse_format = st.radio("Formato de salida", ["gif", "mp4"], horizontal=True)
#         fps = st.slider("Velocidad (fps)", min_value=1, max_value=10, value=2)
    
#     # Filtrar datos según el rango de fechas seleccionado
#     df = df_original[
#         (df_original["fecha"].dt.date >= fecha_inicio) & 
#         (df_original["fecha"].dt.date <= fecha_fin)
#     ].copy()
    
#     if df.empty:
#         st.error("⚠️ No hay datos disponibles para el rango de fechas seleccionado.")
#         return
        
#     # Configurar la granularidad temporal
#     if granularity == "Horaria":
#         df["time_group"] = df["fecha"].dt.floor("H")
#         slider_format = "%Y-%m-%d %H:%M"
#     elif granularity == "Diaria":
#         df["time_group"] = df["fecha"].dt.floor("D")
#         slider_format = "%Y-%m-%d"
#     elif granularity == "Semanal":
#         df["time_group"] = df["fecha"].dt.to_period("W").dt.to_timestamp()
#         slider_format = "%Y-%m-%d"
#     elif granularity == "Mensual":
#         df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
#         slider_format = "%Y-%m"
#     else:  # Anual
#         df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
#         slider_format = "%Y"
        
#     # Agregar datos según granularidad seleccionada
#     if granularity != "Horaria":
#         df = df.groupby(["time_group", "latitud", "longitud"]).agg(
#             no2_value=("no2_value", "mean"),
#             fecha=("fecha", "min")
#         ).reset_index()
        
#     # Obtener los grupos de tiempo únicos
#     time_groups = sorted(df["time_group"].unique())
    
#     if len(time_groups) == 0:
#         st.error("⚠️ No hay suficientes datos para la granularidad seleccionada.")
#         return
    
#     # Organizar el layout
#     #col1, col2 = st.columns([2, 0])
    
#     #with col1:
#     st.markdown('<div class="sub-header">🗺️ Mapa de concentraciones de NO₂</div>', unsafe_allow_html=True)
    
#     # Slider para seleccionar el tiempo
#     # Convertir los time_groups a formato datetime para el slider
#     slider_values = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in time_groups]
    
#     selected_time = st.select_slider(
#         "Selecciona el momento temporal",
#         options=slider_values,
#         format_func=lambda x: x.strftime(slider_format) if hasattr(x, 'strftime') else str(x),
#         value=slider_values[0]
#     )
    
#     # Mostrar la fecha seleccionada
#     st.markdown(f"📅 **Mostrando datos para:** {selected_time.strftime(slider_format) if hasattr(selected_time, 'strftime') else selected_time}")
    
#     # Filtrar DataFrame para el tiempo seleccionado
#     try:
#         if granularity == "Mensual":
#             df_selected = df[df["time_group"].dt.to_period("M") == pd.Period(selected_time, "M")]
#         elif granularity == "Anual":
#             df_selected = df[df["time_group"].dt.to_period("Y") == pd.Period(selected_time, "Y")]
#         elif granularity == "Semanal":
#             df_selected = df[df["time_group"].dt.to_period("W") == pd.Period(selected_time, "W")]
#         else:
#             # Para horaria o diaria
#             df_selected = df[df["time_group"] == selected_time]
#     except Exception as e:
#         st.error(f"Error al filtrar datos: {str(e)}")
#         df_selected = pd.DataFrame()
    
#     # Crear y mostrar el mapa
#     if not df_selected.empty:
#         try:
            
#             m = crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_seleccionado)
#             if m:
#                 folium_static(m, width=700, height=500)
                
#                 # Mostrar estadísticas básicas
#                 avg_no2 = df_selected["no2_value"].mean()
#                 max_no2 = df_selected["no2_value"].max()
                
#                 # Evaluar el nivel según límites recomendados
#                 if avg_no2 <= 40:
#                     nivel = "Bajo"
#                     color = "green"
#                 elif avg_no2 <= 100:
#                     nivel = "Medio"
#                     color = "orange"
#                 else:
#                     nivel = "Alto"
#                     color = "red"
                
#                 st.markdown(f"""
#                 <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Media NO₂</div>
#                         <div style="font-size: 1.5rem; color: {color};">{avg_no2:.1f} μg/m³</div>
#                     </div>
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Máximo NO₂</div>
#                         <div style="font-size: 1.5rem; color: red">{max_no2:.1f} μg/m³</div>
#                     </div>
#                     <div style="text-align: center; padding: 0.5rem; background-color: #f0f0f0; border-radius: 0.5rem; width: 30%;">
#                         <div style="font-size: 0.8rem; color: #666;">Nivel</div>
#                         <div style="font-size: 1.5rem; color: {color};">{nivel}</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error al crear el mapa: {str(e)}")
#             st.info("Intenta con un rango de fechas diferente o una granularidad distinta.")
#     else:
#         st.info("ℹ️ No hay datos disponibles para el momento seleccionado.")
        
#     st.markdown('<div class="sub-header">📊 Estadísticas</div>', unsafe_allow_html=True)

#     if show_stats and not df.empty:
#         try:
#             # Preparar datos para gráficos
#             stats_df = df.groupby("time_group").agg(
#                 no2_promedio=("no2_value", "mean"),
#                 no2_max=("no2_value", "max"),
#                 num_readings=("no2_value", "count")
#             ).reset_index()
            
#             # Añadir formato de fecha para mostrar en tooltips
#             stats_df["fecha_str"] = stats_df["time_group"].dt.strftime(slider_format)
            
#             # Convertir time_group a datetime (por si acaso no lo estuviera)
#             stats_df["time_group"] = pd.to_datetime(stats_df["time_group"])
            
#             # Gráfico de evolución temporal de NO₂ usando campo temporal
#             st.write("**Evolución temporal de NO₂**")
            
#             line_chart = alt.Chart(stats_df).mark_line(point=True).encode(
#                 x=alt.X('time_group:T', title='Fecha', axis=alt.Axis(format=slider_format)),
#                 y=alt.Y('no2_promedio:Q', title='NO₂ promedio (μg/m³)'),
#                 tooltip=[
#                     alt.Tooltip('fecha_str:N', title='Fecha'),
#                     alt.Tooltip('no2_promedio:Q', title='NO₂ promedio', format='.1f'),
#                     alt.Tooltip('no2_max:Q', title='NO₂ máximo', format='.1f'),
#                     alt.Tooltip('num_readings:Q', title='Nº de mediciones')
#                 ]
#             ).properties(height=200)
            
#             # Filtrar valores extremos y negativos
#             #hist_data = stats_df[(stats_df["no2_promedio"] >= 0) & (stats_df["no2_promedio"] <= 500)]  # Ajusta el límite si es necesario
#             hist_data = df[(df["no2_value"] >= 0) & (df["no2_value"] <= 500)]  # Ajusta el límite si es necesario

#             # Línea de referencia para límite OMS
#             limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(
#                 color='red', strokeDash=[3, 3]
#             ).encode(y='y:Q')
            
#             st.altair_chart(line_chart + limit_line, use_container_width=True)
            
#             st.write("La OMS recomienda que los niveles medios anuales de NO₂ no superen los 40 μg/m³ para proteger la salud pública. (linea roja horizontal)")
#             #TODO: añadir leyenda.
            
#             # Histograma de valores de NO₂
#             st.write("**Distribución de valores de NO₂**")
#             hist = alt.Chart(hist_data).mark_bar().encode(
#                 x=alt.X('no2_value:Q', bin=alt.Bin(maxbins=30), title='NO₂ (μg/m³)'),
#                 y=alt.Y('count()', title='Frecuencia')
#             ).properties(height=150)
            
#             st.altair_chart(hist, use_container_width=True)
            
#             # Información sobre límites recomendados
#             # st.markdown("""
#             # <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 0.9rem;">
#             # <strong>Nota:</strong> La OMS recomienda que los niveles medios anuales de NO₂ no superen los 
#             # <span style="color: red;">40 μg/m³</span> para proteger la salud pública.
#             # </div>
#             # """, unsafe_allow_html=True)
            
#         except Exception as e:
#             st.error(f"Error al generar gráficos: {str(e)}")

#     # Sección de generación de timelapse
#     st.markdown('<div class="sub-header">🎬 Timelapse de evolución temporal</div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         # Advertencia sobre posibles limitaciones
#         #st.info("⚠️ La generación del timelapse puede requerir dependencias adicionales como Selenium o imgkit.")
        
#         if st.button("🎬 Generar Timelapse", key="generate_timelapse"):
#             try:
#                 # Generar el timelapse
#                 timelapse_data = generar_timelapse(
#                     df, time_groups, slider_format, global_min, global_max, 
#                     modo=timelapse_format, fps=fps, nivel_contaminacion=nivel_seleccionado
#                 )
                
#                 if timelapse_data:
#                     # Guardar el timelapse en la sesión
#                     st.session_state.timelapse = timelapse_data
#                     st.session_state.timelapse_format = timelapse_format
#                     st.success("✅ Timelapse generado correctamente!")
#                 else:
#                     st.error("❌ No se pudo generar el timelapse.")
#             except Exception as e:
#                 st.error(f"Error al generar timelapse: {str(e)}")
#                 st.info("Para generar timelapse, asegúrate de tener instalado Selenium o imgkit.")
    
#     with col2:
#         # Mostrar el timelapse si existe en la sesión
#         if 'timelapse' in st.session_state and st.session_state.timelapse:
#             # if st.session_state.timelapse_format == "gif":
#             #     #st.image(st.session_state.timelapse, caption="Timelapse de evolución de NO₂")
#             #     print("st.session_state.timelapse ", st.session_state.timelapse)
#             #     st.markdown(f'<img src="{st.session_state.timelapse}" alt="Timelapse de NO₂" style="width: 100%;">', unsafe_allow_html=True)
#             # else:
#             #     st.video(st.session_state.timelapse)
            
#             # Botón para descargar
#             filename = f"timelapse_no2_{fecha_inicio}_{fecha_fin}.{st.session_state.timelapse_format}"
#             st.download_button(
#                 label=f"⬇️ Descargar {st.session_state.timelapse_format.upper()}",
#                 data=st.session_state.timelapse,
#                 file_name=filename,
#                 mime=f"image/{st.session_state.timelapse_format}" if st.session_state.timelapse_format == "gif" else "video/mp4"
#             )
#         else:
#             st.info("Haz clic en 'Generar Timelapse' para crear una visualización animada de la evolución de NO₂ en el tiempo.")
            
#     # Pie de página
#     st.markdown("""
#     <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
#         Datos proporcionados por el Ayuntamiento de Madrid. Última actualización: {}.
#     </div>
#     """.format(df_original["fecha"].max().strftime("%d/%m/%Y")), unsafe_allow_html=True)

def crear_mapa_no2():
    """Función principal para la visualización de datos de NO₂ en Madrid."""

    st.markdown('<div class="sub-header">🌍 Análisis de niveles de NO₂ en Madrid</div>', unsafe_allow_html=True)

    # Información contextual
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

    # Cargar datos
    with st.spinner('Cargando datos...'):
        try:
            df_original = cargar_datos()
        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
            st.info("Asegúrate de que el archivo 'data/more_processed/air_data.csv' existe y es accesible.")
            return

    # Obtener los valores mínimos y máximos globales para la normalización
    global_min = df_original["no2_value"].min()
    global_max = df_original["no2_value"].max()

    # --- Controles de configuración (en lugar de st.sidebar, se incluyen en la página) ---
    st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2_main = st.columns([1, 3])
        with col1:
            # Filtro de fechas
            st.markdown("#### 📅 Rango de fechas")
            fecha_min = df_original["fecha"].min().date()
            fecha_max = df_original["fecha"].max().date()
            fecha_inicio = st.date_input("Fecha inicial", fecha_min, min_value=fecha_min, max_value=fecha_max)
            fecha_fin = st.date_input("Fecha final", fecha_max, min_value=fecha_min, max_value=fecha_max)
            if fecha_inicio > fecha_fin:
                st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
                fecha_fin = fecha_inicio + timedelta(days=7)
                
             # Granularidad temporal y filtro de nivel de contaminación
            st.markdown("#### ⏱️ Agregación y filtro")
            granularity = st.radio("Granularidad", ["Horaria", "Diaria", "Semanal", "Mensual", "Anual"], horizontal=True)
            nivel_contaminacion = st.selectbox("Nivel de contaminación", ["Todos", "Bajo", "Medio", "Alto"])
            nivel_seleccionado = None if nivel_contaminacion == "Todos" else nivel_contaminacion
            
            # Opciones de visualización adicionales en otra fila
            st.markdown("#### 🔆 Opciones de visualización y timelapse")
            # with st.container():
            #     col1, col2 = st.columns(2)
            #     with col1:
            show_stats = st.checkbox("Mostrar estadísticas", value=True)
                # with col2:
            timelapse_format = st.radio("Formato de salida", ["gif", "mp4"], horizontal=True)
            fps = st.slider("Velocidad (fps)", min_value=1, max_value=10, value=2)
            
        with col2_main:
            
            # --- Fin de los controles integrados ---

            # Filtrar datos según el rango de fechas seleccionado
            df = df_original[
                (df_original["fecha"].dt.date >= fecha_inicio) & 
                (df_original["fecha"].dt.date <= fecha_fin)
            ].copy()
            if df.empty:
                st.error("⚠️ No hay datos disponibles para el rango de fechas seleccionado.")
                return

            # Configurar la granularidad temporal
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

            if granularity != "Horaria":
                df = df.groupby(["time_group", "latitud", "longitud"]).agg(
                    no2_value=("no2_value", "mean"),
                    fecha=("fecha", "min")
                ).reset_index()

            # Obtener los grupos de tiempo únicos
            time_groups = sorted(df["time_group"].unique())
            if len(time_groups) == 0:
                st.error("⚠️ No hay suficientes datos para la granularidad seleccionada.")
                return

            st.markdown('<div class="sub-header">🗺️ Mapa de concentraciones de NO₂</div>', unsafe_allow_html=True)

            # Slider para seleccionar el momento temporal
            slider_values = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in time_groups]
            selected_time = st.select_slider(
                "Selecciona el momento temporal",
                options=slider_values,
                format_func=lambda x: x.strftime(slider_format) if hasattr(x, 'strftime') else str(x),
                value=slider_values[0]
            )
            st.markdown(f"📅 **Mostrando datos para:** {selected_time.strftime(slider_format) if hasattr(selected_time, 'strftime') else selected_time}")

            try:
                if granularity == "Mensual":
                    df_selected = df[df["time_group"].dt.to_period("M") == pd.Period(selected_time, "M")]
                elif granularity == "Anual":
                    df_selected = df[df["time_group"].dt.to_period("Y") == pd.Period(selected_time, "Y")]
                elif granularity == "Semanal":
                    df_selected = df[df["time_group"].dt.to_period("W") == pd.Period(selected_time, "W")]
                else:
                    df_selected = df[df["time_group"] == selected_time]
            except Exception as e:
                st.error(f"Error al filtrar datos: {str(e)}")
                df_selected = pd.DataFrame()

            if not df_selected.empty:
                
                import streamlit.components.v1 as components

                try:
                    m = crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_seleccionado)
                    if m:
                        with st.container():
                            col1, col2 = st.columns([4, 1])

                            with col1:
                                # Renderizar el mapa como un iframe para forzar que ocupe el ancho completo
                                map_html = f"""
                                <div style="width: 100%; height: 100%;">
                                    {m._repr_html_()}
                                </div>
                                """
                                components.html(map_html, height=600)
                            
                            with col2:
                                avg_no2 = df_selected["no2_value"].mean()
                                max_no2 = df_selected["no2_value"].max()
                                if avg_no2 <= 40:
                                    nivel = "Bajo"
                                    color = "green"
                                elif avg_no2 <= 100:
                                    nivel = "Medio"
                                    color = "orange"
                                else:
                                    nivel = "Alto"
                                    color = "red"
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


                except Exception as e:
                    st.error(f"Error al crear el mapa: {str(e)}")
                    st.info("Intenta con un rango de fechas diferente o una granularidad distinta.")
            else:
                st.info("ℹ️ No hay datos disponibles para el momento seleccionado.")


    st.markdown("## 📊 Estadísticas")
    
    print("cols ", df.columns)
    
    # sensor_seleccionado = st.selectbox(
    #     "Selecciona un sensor de NO₂",
    #     df["id_no2"].unique(),
    #     index=0  # Por defecto "Todos"
    # )
    
    if show_stats and not df.empty:
        try:
            stats_df = df.groupby("time_group").agg(
                no2_promedio=("no2_value", "mean"),
                no2_max=("no2_value", "max"),
                num_readings=("no2_value", "count")
            ).reset_index()
            stats_df["fecha_str"] = stats_df["time_group"].dt.strftime(slider_format)
            stats_df["time_group"] = pd.to_datetime(stats_df["time_group"])
            
            st.write("**Evolución temporal de NO₂**")
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
            
            limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(color='red', strokeDash=[3, 3]).encode(y='y:Q')
            st.altair_chart(line_chart + limit_line, use_container_width=True)
            st.write("La OMS recomienda que los niveles medios anuales de NO₂ no superen los 40 μg/m³ (línea roja).")
            st.write("**Distribución de valores de NO₂**")
            
            hist = alt.Chart(df[(df["no2_value"] >= 0) & (df["no2_value"] <= 500)]).mark_bar().encode(
                x=alt.X('no2_value:Q', bin=alt.Bin(maxbins=30), title='NO₂ (μg/m³)'),
                y=alt.Y('count()', title='Frecuencia')
            ).properties(height=150)
            st.altair_chart(hist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al generar gráficos: {str(e)}")

    st.markdown('<div class="sub-header">🎬 Timelapse de evolución temporal</div>', unsafe_allow_html=True)
    col1, _ = st.columns([1, 2])
    with col1:
        if st.button("🎬 Generar Timelapse", key="generate_timelapse"):
            try:
                timelapse_data = generar_timelapse(
                    df, time_groups, slider_format, global_min, global_max, 
                    modo=timelapse_format, fps=fps, nivel_contaminacion=nivel_seleccionado
                )
                if timelapse_data:
                    st.session_state.timelapse = timelapse_data
                    st.session_state.timelapse_format = timelapse_format
                    st.success("✅ Timelapse generado correctamente!")
                else:
                    st.error("❌ No se pudo generar el timelapse.")
            except Exception as e:
                st.error(f"Error al generar timelapse: {str(e)}")
                st.info("Asegúrate de tener instaladas las dependencias necesarias (Selenium o imgkit).")

        if 'timelapse' in st.session_state and st.session_state.timelapse:
            filename = f"timelapse_no2_{fecha_inicio}_{fecha_fin}.{st.session_state.timelapse_format}"
            st.download_button(
                label=f"⬇️ Descargar {st.session_state.timelapse_format.upper()}",
                data=st.session_state.timelapse,
                file_name=filename,
                mime="image/gif" if st.session_state.timelapse_format == "gif" else "video/mp4"
            )
        else:
            st.info("Haz clic en 'Generar Timelapse' para crear una visualización animada de la evolución de NO₂ en el tiempo.")

    st.markdown("""
    <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
        Datos proporcionados por el Ayuntamiento de Madrid. Última actualización: {}.
    </div>
    """.format(df_original["fecha"].max().strftime("%d/%m/%Y")), unsafe_allow_html=True)
