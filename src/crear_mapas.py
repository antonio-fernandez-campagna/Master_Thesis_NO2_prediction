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

@st.cache_data(ttl=3600)
def cargar_mapping_no2_traffic():
    """Carga y almacena en caché el mapeo entre sensores NO2 y tráfico"""
    return pd.read_csv('data/more_processed/mapping_no2_y_traffic_filtered_by_proximity.csv')

@st.cache_data(ttl=3600)
def cargar_datos_traffic():
    """Carga datos de tráfico con caché"""
    return pd.read_parquet('data/more_processed/traffic_data.parquet')

@st.cache_data(ttl=3600)
def cargar_datos_air():
    """Carga y preprocesa los datos de aire con caché"""
    df = pd.read_parquet('data/more_processed/air_data.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def limpiar_cache():
    """Limpia todo el caché de streamlit"""
    cargar_datos_no2_locations.clear()
    cargar_datos_traffic_locations.clear()
    cargar_mapping_no2_traffic.clear()
    cargar_datos_traffic.clear()
    cargar_datos_air.clear()
    crear_mapa_trafico_y_no2_all.clear()
    crear_mapa_sensores_asignados_a_cada_no2.clear()
    crear_mapa_sensores_asignados_a_cada_no2_continuo.clear()
    gc.collect()  # Forzar recolección de basura

# Function to create a simple example map
def crear_mapa_trafico_y_no2_all():

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


def crear_mapa_sensores_asignados_a_cada_no2():
    
    df = cargar_mapping_no2_traffic()

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


def crear_mapa_sensores_asignados_a_cada_no2_continuo():
    
    df = cargar_mapping_no2_traffic()

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
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitud_trafico'], row['longitud_trafico']],
            popup=f"Traffic ID: {row['id_trafico']}, NO2 ID: {row['id_no2']}",
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10pt">🚦</div>'  
            )
        ).add_to(m)

    # Save the map to an HTML file or display it directly
    return m



# Función para mostrar la continuidad de datos
def mostrar_continuidad(sensor):
    with st.spinner("Cargando datos de continuidad..."):
        df = cargar_datos_traffic()
        
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


# Función para crear mapa de calor optimizado
def crear_mapa_con_heatmap(df_selected, global_min, global_max, nivel_contaminacion=None):
    """Crea un mapa Folium optimizado para reducir memoria"""
    if df_selected.empty:
        return None
        
    map_center = [df_selected["latitud"].mean(), df_selected["longitud"].mean()]

    m = leafmap.Map(
        center=map_center,
        zoom=12,
        tiles="CartoDB positron",
        draw_control=False,
        measure_control=False,
        fullscreen_control=True
    )
    
    # Personalizar datos según nivel de contaminación
    if nivel_contaminacion:
        if nivel_contaminacion == "Bajo":
            df_selected = df_selected[df_selected['no2_value'] <= 40]
        elif nivel_contaminacion == "Medio":
            df_selected = df_selected[(df_selected['no2_value'] > 40) & (df_selected['no2_value'] <= 100)]
        elif nivel_contaminacion == "Alto":
            df_selected = df_selected[df_selected['no2_value'] > 100]

    # Si hay demasiados puntos, muestrear para mejor rendimiento
    max_points = 2000  # Limitar número de puntos para el heatmap
    if len(df_selected) > max_points:
        df_selected = df_selected.sample(max_points)

    if not df_selected.empty:
        # Configurar parámetros de heatmap según cantidad de datos
        radius = 15 if len(df_selected) > 100 else 25
        blur = 10 if len(df_selected) > 100 else 15
    
        # Normalizar valores una sola vez y guardar como lista para evitar operaciones repetidas
        heat_data = []
        for _, row in df_selected.iterrows():
            normalized_value = max(0.1, min(1, (row['no2_value'] - global_min) / (global_max - global_min) * 0.8 + 0.2))
            heat_data.append([row['latitud'], row['longitud'], normalized_value])
        
        m.add_heatmap(
            data=heat_data,
            name="NO2 Heatmap",
            radius=radius,
            blur=blur,
        )
    
    return m



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

    if st.button("Cargar datos de no2"):

        # Cargar datos
        with st.spinner('Cargando datos...'):
            try:
                df_original = cargar_datos_air()
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
                            
                # Obtener lista de sensores y agregar "Todos" al inicio
                sensores = sorted(df_original["id_no2"].unique())
                sensores = ["Todos"] + list(sensores)
                
                sensor_seleccionado = st.selectbox(
                    "Selecciona un sensor de NO₂",
                    sensores,
                    index=0  # Por defecto "Todos"
                )
                
                if sensor_seleccionado != "Todos":
                    df_original = df_original[df_original["id_no2"] == sensor_seleccionado]
                
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
                st.write("La OMS recomienda que los niveles medios anuales de NO₂ no superen los 40 μg/m³ (línea roja).")
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
