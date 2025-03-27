import folium
import pandas as pd
import random

from streamlit_folium import folium_static

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

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca.colormap as bcm

from folium.plugins import HeatMap
import leafmap.foliumap as leafmap

# Librerías adicionales
import imageio
import os

def crear_mapa_con_heatmap(df_selected, global_min, global_max, slider_format):
    """
    Crea un mapa Folium (usando leafmap) centrado en la media de las coordenadas.
    """
    map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]

    m = leafmap.Map(center=[map_center[0], map_center[1]], zoom=12)

    m.add_heatmap(
        data=df_selected,
        latitude="latitud",
        longitude="longitud",
        value="no2_value",  
        name="NO2 Heatmap",
        radius=25,
        blur=15
    )
    return m

def generar_gif_o_video(df, time_groups, slider_format, global_min, global_max, modo="gif"):
    """
    Genera un GIF o video (mp4) a partir de mapas Folium en diferentes momentos de tiempo.
    1. Itera sobre cada fecha/hora en time_groups.
    2. Crea un mapa Folium con la capa de calor.
    3. Usa m._to_png() para convertirlo en imagen (requiere Selenium).
    4. Combina todas las imágenes en un GIF o MP4.
    """
    # Contendrá los frames (imágenes) en formato numpy array
    frames = []

    for t in time_groups:
        
        print("t: ", t)

        df_t = df[df["time_group"] == t]
        
        print("df_t head", df_t.head())
        if df_t.empty:
            continue
        
        # Creamos el mapa
        m = crear_mapa_con_heatmap(df_t, global_min, global_max, slider_format)
        
        # Convertimos el mapa a PNG (requiere Selenium y un driver)
        # Ajusta el factor de escala si quieres más resolución
        png_data = m._to_png(5)  # <-- método "privado" de Folium
        # Lo leemos como array de imagen con imageio
        img = imageio.v3.imread(png_data)
        frames.append(img)
        
    if not frames:
        st.warning("No se generaron frames (quizá no hay datos en el rango).")
        return None

    # Guardamos en disco
    if modo == "gif":
        output_path = "timelapse.gif"
        imageio.mimsave(output_path, frames, fps=1)  # Ajusta fps
        return output_path
    else:
        # Para mp4 se requiere ffmpeg instalado
        output_path = "timelapse.mp4"
        imageio.mimsave(output_path, frames, fps=1, codec="libx264")
        return output_path

def mostrar_timelapse_como_video(df, time_groups, slider_format, global_min, global_max):
    """
    En lugar de mostrar frame a frame en la web,
    creamos un GIF o MP4 y luego lo mostramos.
    """
    st.subheader("Generar Timelapse como Video/GIF")

    col1, col2 = st.columns(2)
    with col1:
        modo = st.selectbox("Formato de salida", ["gif", "mp4"], index=0)
    with col2:
        if st.button("Generar Timelapse"):
            st.info("Creando frames, por favor espera...")

            # Generar el GIF o MP4
            print("llega 1")
            video_path = generar_gif_o_video(df, time_groups, slider_format, global_min, global_max, modo=modo)
            print("llega 2")

            if video_path and os.path.exists(video_path):
                st.success(f"Timelapse generado: {video_path}")
                if modo == "gif":
                    st.image(video_path)
                else:
                    st.video(video_path)
            else:
                st.error("No se pudo generar el video.")

def crear_mapa_no2():
    st.header("Análisis de niveles de NO₂ en Madrid")
    
    # Cargar el dataset
    df_original = pd.read_csv('data/more_processed/air_data.csv')
    df_original['fecha'] = pd.to_datetime(df_original['fecha'])
    
    global_min = df_original["no2_value"].min()
    global_max = df_original["no2_value"].max()
    
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Fecha de inicio", df_original["fecha"].min().date())
    with col2:
        fecha_fin = st.date_input("Fecha de fin", df_original["fecha"].max().date())
    
    df = df_original[
        (df_original["fecha"].dt.date >= fecha_inicio) & 
        (df_original["fecha"].dt.date <= fecha_fin)
    ].copy()
    
    granularity = st.selectbox("Selecciona la granularidad", ["Horaria", "Mensual", "Anual"])
    if granularity == "Horaria":
        df["time_group"] = df["fecha"].dt.floor("H")
        slider_format = "%Y-%m-%d %H:%M"
    elif granularity == "Mensual":
        df["time_group"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
        slider_format = "%Y-%m-%d"
    else:  # Anual
        df["time_group"] = df["fecha"].dt.to_period("Y").dt.to_timestamp()
        slider_format = "%Y"
    
    if granularity in ["Mensual", "Anual"]:
        df = df.groupby(["time_group", "latitud", "longitud"]).agg(
            no2_value=("no2_value", "mean"),
            fecha=("fecha", "min")
        ).reset_index()
        
    print(df.head(30))
    print(len(df))
    
    if df.empty:
        st.error("No hay datos en el rango seleccionado.")
        return
    
    time_groups = sorted(df["time_group"].unique())
    time_groups = [t.to_pydatetime() for t in time_groups]
    
    # Slider para ver un instante específico
    selected_time = st.slider(
        "Selecciona el tiempo",
        min_value=time_groups[0],
        max_value=time_groups[-1],
        value=time_groups[0],
        format=slider_format
    )
    st.write(f"Mostrando datos para: {selected_time.strftime(slider_format)}")
    
    df_selected = df[df["time_group"] == selected_time]
    
    if not df_selected.empty:
        m = crear_mapa_con_heatmap(df_selected, global_min, global_max, slider_format)
        folium_static(m)
    else:
        st.info("No hay datos para el tiempo seleccionado.")
    
    # Llamamos a la función que genera el video/gif
    mostrar_timelapse_como_video(df, time_groups, slider_format, global_min, global_max)

# Si es tu script principal, descomenta:
# if __name__ == "__main__":
#     crear_mapa_no2()
