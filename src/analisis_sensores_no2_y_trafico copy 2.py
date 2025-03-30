import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta
import seaborn as sns
import altair as alt
import calplot
from matplotlib.colors import LinearSegmentedColormap

@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """Carga y preprocesa los datos con caché"""
    df = pd.read_parquet('data/more_processed/no2_with_traffic_and_meteo_one_station.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def analisis_sensores():
    """Función para el análisis comparativo de sensores de tráfico y NO2."""
    
    st.markdown('<div class="sub-header">🚗 Análisis de sensores de tráfico y su relación con NO₂</div>', unsafe_allow_html=True)

    # Información contextual
    with st.expander("ℹ️ Acerca de este análisis", expanded=False):
        st.markdown("""
        <div class="info-box">
        <p>Este panel permite analizar la relación entre los datos de tráfico y los niveles de NO₂ en Madrid.</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Manejo del estado ----
    if "df_master" not in st.session_state:
        st.session_state.df_master = None

    if st.button("Cargar datos") or st.session_state.df_master is not None:
        if st.session_state.df_master is None:
            with st.spinner('Cargando datos...'):
                try:
                    st.session_state.df_master = cargar_datos_trafico_y_meteo()
                except Exception as e:
                    st.error(f"Error al cargar los datos: {str(e)}")
                    return

        df = st.session_state.df_master

        # --- Controles de configuración ---
        st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sensores = sorted(df["id_no2"].unique())
            sensor_seleccionado = st.selectbox("Selecciona un sensor de NO₂", sensores, index=0)
            
            # Obtener los sensores de tráfico relacionados
            sensores_trafico = sorted(df[df["id_no2"] == sensor_seleccionado]["id_trafico"].unique())
            sensor_trafico = st.selectbox("Selecciona un sensor de tráfico", sensores_trafico, index=0) if sensores_trafico else None
            
            # Filtro de fechas
            fecha_min = df["fecha"].min().date()
            fecha_max = df["fecha"].max().date()
            fecha_inicio = st.date_input("Fecha inicial", fecha_min, min_value=fecha_min, max_value=fecha_max, key="sensor_fecha_inicio")
            fecha_fin = st.date_input("Fecha final", fecha_max, min_value=fecha_min, max_value=fecha_max, key="sensor_fecha_fin")
            
            if fecha_inicio > fecha_fin:
                st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
                fecha_fin = fecha_inicio + timedelta(days=7)
            
            # Granularidad temporal
            granularity = st.radio("Granularidad", ["Horaria", "Diaria", "Semanal", "Mensual"], horizontal=True)
            
            # Variables a analizar
            variables_disponibles = ['intensidad', 'carga', 'ocupacion', 'd2m', 't2m', 'sst', 'ssrd', 'u10', 'v10', 'sp', 'tp']
            variables_disponibles = [var for var in variables_disponibles if var in df.columns]
            variables_seleccionadas = st.multiselect("Selecciona variables a comparar con NO₂", variables_disponibles, default=variables_disponibles)
            
            # Opciones de visualización
            mostrar_correlacion = st.checkbox("Mostrar matriz de correlación", value=True)
            mostrar_calplot = st.checkbox("Mostrar calendario de disponibilidad", value=True)
            
            # Filtrado de datos
            df_filtrado = df[(df["id_no2"] == sensor_seleccionado) & 
                            (df["fecha"].dt.date >= fecha_inicio) & 
                            (df["fecha"].dt.date <= fecha_fin)].copy()
            
            if sensor_trafico:
                df_filtrado = df_filtrado[df_filtrado["id_trafico"] == sensor_trafico]

        with col2:

            if df_filtrado.empty:
                st.error("⚠️ No hay datos disponibles para el sensor y rango de fechas seleccionados.")
                return
            
            # Configurar granularidad
            if granularity == "Horaria":
                df_filtrado["time_group"] = df_filtrado["fecha"].dt.floor("H")
            elif granularity == "Diaria":
                df_filtrado["time_group"] = df_filtrado["fecha"].dt.floor("D")
            elif granularity == "Semanal":
                df_filtrado["time_group"] = df_filtrado["fecha"].dt.to_period("W").dt.to_timestamp()
            else:  # Mensual
                df_filtrado["time_group"] = df_filtrado["fecha"].dt.to_period("M").dt.to_timestamp()
            
            if granularity != "Horaria":
                variables_agg = ["no2_value"] + variables_seleccionadas
                df_agregado = df_filtrado.groupby(["time_group"]).agg({var: "mean" for var in variables_agg}).reset_index()
            else:
                df_agregado = df_filtrado

            # --- Sección de visualizaciones ---
            st.markdown('<div class="sub-header">📊 Análisis comparativo de variables</div>', unsafe_allow_html=True)

            if mostrar_correlacion and len(variables_seleccionadas) >= 1:
                st.markdown("### Matriz de correlación")
                corr_data = df_agregado[["no2_value"] + variables_seleccionadas].corr()

                fig, ax = plt.subplots(figsize=(20, 8))
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
                plt.title('Correlación entre NO₂ y variables seleccionadas')
                st.pyplot(fig)
            
            # Nueva sección: Visualización de disponibilidad de datos (calplot)
            if mostrar_calplot:
                st.markdown("### Calendario de disponibilidad de datos")
                
                # Crear un DataFrame con el índice temporal completo
                fecha_min_filtro = df_filtrado["fecha"].min()
                fecha_max_filtro = df_filtrado["fecha"].max()
                
                # Crear un índice con todas las horas que deberían existir en el rango
                todas_horas = pd.date_range(start=fecha_min_filtro.replace(hour=0, minute=0, second=0), 
                                         end=fecha_max_filtro.replace(hour=23, minute=59, second=59), 
                                         freq='H')
                
                # Crear DataFrame con todas las horas
                df_completo = pd.DataFrame(index=todas_horas)
                df_completo.index.name = 'hora'
                
                # Marcar las horas que existen en los datos originales
                df_filtrado_hora = df_filtrado.set_index("fecha")
                df_completo['tiene_datos'] = df_completo.index.isin(df_filtrado_hora.index).astype(int)
                
                # Agregar columnas de fecha y hora del día
                df_completo['fecha'] = df_completo.index.date
                df_completo['hora_dia'] = df_completo.index.hour
                
                # Preparar datos para el calendario
                df_completo['fecha'] = pd.to_datetime(df_completo['fecha'])
                datos_diarios = df_completo.groupby('fecha')['tiene_datos'].sum()
                datos_diarios = datos_diarios / 24 * 100  # Convertir a porcentaje de completitud

                

                print(datos_diarios.head())
                print(type(datos_diarios))
                
                # Crear visualización de calendario
                fig, ax = plt.subplots(figsize=(16, 10))
                calplot.calplot(datos_diarios, cmap='YlGn', 
                               fillcolor='lightgrey',
                               vmin=0, vmax=100, 
                               suptitle=f'Disponibilidad diaria de datos (%) - Sensor NO₂: {sensor_seleccionado}' +
                               (f', Sensor Tráfico: {sensor_trafico}' if sensor_trafico else ''))
                

                plt.tight_layout()
                st.pyplot(fig)

                fig.savefig("temp.png")
                st.image("temp.png")

                
                # Mostrar estadísticas de completitud
                total_horas = len(todas_horas)
                horas_con_datos = df_completo['tiene_datos'].sum()
                porcentaje_completitud = (horas_con_datos / total_horas) * 100
                
                
                # Mostrar resumen de disponibilidad
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Total de horas en el periodo", f"{total_horas}", f"{horas_con_datos} con datos")
                with col_stats2:
                    st.metric("Porcentaje de completitud", f"{porcentaje_completitud:.2f}%", 
                             f"{total_horas - horas_con_datos} horas sin datos")
                
                # Heatmap de disponibilidad por hora del día y día de la semana
                st.markdown("### Disponibilidad por hora del día y día de la semana")
                
                # Agregar día de la semana
                df_completo['dia_semana'] = df_completo.index.dayofweek
                
                # Crear heatmap de disponibilidad
                pivot_data = df_completo.pivot_table(
                    values='tiene_datos', 
                    index='hora_dia',
                    columns='dia_semana', 
                    aggfunc='mean'
                ) * 100  # Convertir a porcentaje
                
                # Definir etiquetas para días y horas
                dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot_data, annot=False, cmap='YlGnBu', vmin=0, vmax=100, 
                           xticklabels=dias_semana, yticklabels=range(24),
                           cbar_kws={'label': 'Disponibilidad (%)'})
                plt.title('Disponibilidad de datos por hora del día y día de la semana (%)')
                plt.xlabel('Día de la semana')
                plt.ylabel('Hora del día')
                st.pyplot(fig)