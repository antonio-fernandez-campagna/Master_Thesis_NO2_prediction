import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta
import seaborn as sns
import altair as alt

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

            df_filtrado = df[(df["id_no2"] == sensor_seleccionado) & (df["fecha"].dt.date >= fecha_inicio) & (df["fecha"].dt.date <= fecha_fin)].copy()

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

                #col_1, col_2 = st.columns([3, 1])
                #with col_1:
                fig, ax = plt.subplots(figsize=(20, 8))
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
                plt.title('Correlación entre NO₂ y variables seleccionadas')
                st.pyplot(fig)


        # TDOO: mostrar aqui el codigo nuevo para la nueva visualizacion de datos.

            
        # Lista de colores predefinidos para las variables
        colores_variables = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:cyan']

        if st.button("Mostrar gráficos con una columna: "):

            if len(variables_seleccionadas) > 0:
                for variable in variables_seleccionadas:
                    st.markdown(f"### Evolución temporal comparativa de {variable.capitalize()}")
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.set_xlabel('Fecha')
                    ax1.set_ylabel('NO₂ (μg/m³)', color='tab:blue')
                    ax1.plot(df_agregado['time_group'], df_agregado['no2_value'], color='tab:blue', marker='o', label='NO₂')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    ax1.axhline(y=40, color='r', linestyle='--', alpha=0.7)
                    
                    ax2 = ax1.twinx()
                    ax2.set_ylabel(f'{variable.capitalize()}', color='tab:red')
                    ax2.plot(df_agregado['time_group'], df_agregado[variable], color='tab:red', marker='x', label=variable)
                    ax2.tick_params(axis='y', labelcolor='tab:red')

                    fig.autofmt_xdate()
                    plt.title(f'Comparativa NO₂ vs {variable.capitalize()}')
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    st.markdown(f"### 📊 Gráfico de dispersión de {variable.capitalize()}")
                    # Gráfico de dispersión
                    scatter_chart = alt.Chart(df_agregado).mark_circle(size=60).encode(
                        x=alt.X(variables_seleccionadas[0], title=variables_seleccionadas[0].capitalize()),
                        y=alt.Y('no2_value', title='NO₂ (μg/m³)'),
                        tooltip=[variables_seleccionadas[0], 'no2_value']
                    ).properties(width=600, height=400)
                    
                    st.altair_chart(scatter_chart, use_container_width=True)



        if st.button("Mostrar gráficos con dos columna: "):

            if len(variables_seleccionadas) > 0:
                st.markdown("### 📈 Evolución temporal comparativa")
                
                cols = st.columns(2)  # Crear dos columnas

                for i, variable in enumerate(variables_seleccionadas):
                    col_index = i % 2  # Alternar entre las dos columnas
                    
                    # Asignar un color distinto a cada variable seleccionada
                    color_variable = colores_variables[i % len(colores_variables)]

                    with cols[col_index]:  # Insertar en la columna correspondiente
                        fig, ax1 = plt.subplots(figsize=(12, 4))
                        ax1.set_xlabel('Fecha')
                        ax1.set_ylabel('NO₂ (μg/m³)', color='tab:blue')
                        ax1.plot(df_agregado['time_group'], df_agregado['no2_value'], color='tab:blue', marker='o', label='NO₂')
                        ax1.tick_params(axis='y', labelcolor='tab:blue')
                        ax1.axhline(y=40, color='r', linestyle='--', alpha=0.7)

                        ax2 = ax1.twinx()
                        ax2.set_ylabel(f'{variable.capitalize()}', color=color_variable)
                        ax2.plot(df_agregado['time_group'], df_agregado[variable], color=color_variable, marker='x', label=variable)
                        ax2.tick_params(axis='y', labelcolor=color_variable)

                        fig.autofmt_xdate()
                        plt.title(f'Comparativa NO₂ vs {variable.capitalize()}')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        

                # Separador
                st.markdown("---")

                # Gráfico de dispersión para la primera variable seleccionada
                st.markdown("### 📊 Relación NO₂ vs Variables Seleccionadas")

                scatter_cols = st.columns(2)  # Otra estructura de dos columnas

                for i, variable in enumerate(variables_seleccionadas):

                    col_index = i % 2  # Alternar entre las dos columnas
                    # Asignar color a la variable
                    color_variable = colores_variables[i % len(colores_variables)]
                    

                    with scatter_cols[col_index]:  # Insertar en la columna correspondiente
                        # Crear figura matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Crear scatter plot
                        sns.scatterplot(x=df_agregado[variable], y=df_agregado['no2_value'], color=color_variable, ax=ax)
                        
                        # Agregar línea de regresión sin repetir puntos
                        sns.regplot(x=df_agregado[variable], y=df_agregado['no2_value'], 
                                    scatter=False, color=color_variable, ax=ax)
                        
                        # Configurar títulos y etiquetas
                        plt.title(f'{variable.capitalize()} vs NO₂')
                        plt.xlabel(variable.capitalize())
                        plt.ylabel('NO₂ (μg/m³)')
                        
                        # Ajustar para mejor visualización
                        plt.tight_layout()
                                
                        # Mostrar gráfico en Streamlit
                        st.pyplot(fig)
                        
                        # Agregar un pequeño espacio entre gráficos
                        st.write("")

                st.markdown("---")

                # Gráfico de dispersión para la primera variable seleccionada
                st.markdown("### 📊 Relación NO₂ vs Variables Seleccionadas")

                scatter_cols = st.columns(2)  # Otra estructura de dos columnas

                for i, variable in enumerate(variables_seleccionadas):
                    col_index = i % 2  # Alternar entre columnas

                    # Asignar el mismo color a la variable para la gráfica de dispersión
                    color_variable = colores_variables[i % len(colores_variables)]

                    with scatter_cols[col_index]:  # Insertar en la columna correcta
                        scatter_chart = alt.Chart(df_agregado).mark_circle(size=60).encode(
                            x=alt.X(variable, title=variable.capitalize()),
                            y=alt.Y('no2_value', title='NO₂ (μg/m³)'),
                            tooltip=[variable, 'no2_value']
                        ).properties(width=400, height=300)

                        regression = scatter_chart.transform_regression(
                            variable, 'no2_value'
                        ).mark_line(color=color_variable, strokeDash=[4, 2])

                        st.altair_chart(scatter_chart + regression, use_container_width=True)


                    