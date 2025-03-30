import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta
import seaborn as sns
import altair as alt
import calplot



# ---------- Función de carga de datos ----------
@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """Carga y preprocesa los datos con caché"""
    df = pd.read_parquet('data/more_processed/no2_with_traffic_and_meteo_one_station.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

# ---------- Funciones para visualizaciones ----------
def create_time_series_plot(df_agregado, variable, color_variable):
    """Genera un gráfico de series temporales con dos ejes."""
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('NO₂ (μg/m³)', color='tab:blue')
    ax1.plot(df_agregado['time_group'], df_agregado['no2_value'],
             color='tab:blue', marker='o', label='NO₂')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=40, color='r', linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(variable.capitalize(), color=color_variable)
    ax2.plot(df_agregado['time_group'], df_agregado[variable],
             color=color_variable, marker='x', label=variable)
    ax2.tick_params(axis='y', labelcolor=color_variable)
    
    fig.autofmt_xdate()
    plt.title(f'Comparativa NO₂ vs {variable.capitalize()}')
    plt.grid(True, alpha=0.3)
    return fig

def create_scatter_plot(df_agregado, variable, color_variable):
    """Genera un scatter plot con línea de regresión."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_agregado[variable], y=df_agregado['no2_value'], color=color_variable, ax=ax)
    sns.regplot(x=df_agregado[variable], y=df_agregado['no2_value'], 
                scatter=False, color=color_variable, ax=ax)
    plt.title(f'{variable.capitalize()} vs NO₂')
    plt.xlabel(variable.capitalize())
    plt.ylabel('NO₂ (μg/m³)')
    plt.tight_layout()
    return fig

def create_altair_scatter(df_agregado, variable, color_variable):
    """Genera un gráfico interactivo de dispersión con Altair."""
    scatter_chart = alt.Chart(df_agregado).mark_circle(size=60).encode(
        x=alt.X(variable, title=variable.capitalize()),
        y=alt.Y('no2_value', title='NO₂ (μg/m³)'),
        tooltip=[variable, 'no2_value']
    ).properties(width=400, height=300)
    
    regression = scatter_chart.transform_regression(
        variable, 'no2_value'
    ).mark_line(color=color_variable, strokeDash=[4, 2])
    
    return scatter_chart + regression

def mostrar_matriz_correlacion(df_agregado, variables):
    """Muestra una matriz de correlación entre las variables seleccionadas."""
    cols = ["no2_value"] + variables
    corr_data = df_agregado[cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                square=True, linewidths=.5, ax=ax)
    plt.title('Correlación entre NO₂ y variables seleccionadas')
    
    return fig

def mostrar_disponibilidad_datos(df_filtrado):
    """Analiza y muestra la disponibilidad de datos."""
    fecha_min_filtro = df_filtrado["fecha"].min()
    fecha_max_filtro = df_filtrado["fecha"].max()
    todas_horas = pd.date_range(
        start=fecha_min_filtro.replace(hour=0, minute=0, second=0),
        end=fecha_max_filtro.replace(hour=23, minute=59, second=59),
        freq='H'
    )
    df_completo = pd.DataFrame(index=todas_horas)
    df_completo.index.name = 'hora'
    df_filtrado_hora = df_filtrado.set_index("fecha")
    df_completo['tiene_datos'] = df_completo.index.isin(df_filtrado_hora.index).astype(int)
    
    total_horas = len(todas_horas)
    horas_con_datos = df_completo['tiene_datos'].sum()
    porcentaje_completitud = (horas_con_datos / total_horas) * 100
    
    # Disponibilidad por hora y día de la semana
    df_completo['dia_semana'] = df_completo.index.dayofweek
    pivot_data = df_completo.pivot_table(
        values='tiene_datos', 
        index=df_completo.index.hour,
        columns='dia_semana', 
        aggfunc='mean'
    ) * 100
    
    return {
        'total_horas': total_horas,
        'horas_con_datos': horas_con_datos,
        'porcentaje_completitud': porcentaje_completitud,
        'pivot_data': pivot_data
    }

def renderizar_disponibilidad(disponibilidad):
    """Renderiza las métricas y gráficos de disponibilidad."""
    col1, col2 = st.columns(2)
    
    # Métricas de disponibilidad
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total de horas", 
            f"{disponibilidad['total_horas']}", 
            f"{disponibilidad['horas_con_datos']} con datos"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Completitud (%)", 
            f"{disponibilidad['porcentaje_completitud']:.2f}%", 
            f"{disponibilidad['total_horas'] - disponibilidad['horas_con_datos']} sin datos"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Heatmap de disponibilidad
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            disponibilidad['pivot_data'], 
            annot=False, 
            cmap='YlGnBu', 
            vmin=0, 
            vmax=100, 
            xticklabels=dias_semana, 
            yticklabels=range(24),
            cbar_kws={'label': 'Disponibilidad (%)'}
        )
        plt.title('Disponibilidad por hora y día de la semana (%)')
        plt.xlabel('Día de la semana')
        plt.ylabel('Hora del día')
        
        st.pyplot(fig)

# ---------- Función principal ----------
def analisis_sensores():
    # Título principal
    st.markdown('<div class="main-header">🚗 Análisis de Sensores: Tráfico y NO₂</div>', unsafe_allow_html=True)
    
    # Información contextual
    with st.expander("ℹ️ Acerca de este análisis", expanded=False):
        st.markdown(
            """
            <div class="info-container">
            <p>Este panel permite analizar la relación entre los datos de tráfico y los niveles de NO₂ en Madrid.
            Puedes seleccionar diferentes sensores, periodos de tiempo y variables para visualizar cómo se 
            relacionan con la contaminación por dióxido de nitrógeno.</p>
            </div>
            """, unsafe_allow_html=True
        )
    
    # Carga inicial de datos
    if "df_master" not in st.session_state:
        with st.spinner('Cargando datos iniciales...'):
            try:
                st.session_state.df_master = cargar_datos_trafico_y_meteo()
            except Exception as e:
                st.error(f"Error al cargar los datos: {str(e)}")
                return

    df = st.session_state.df_master
    
    # ---------- Panel de configuración ----------
    st.markdown('<div class="section-header">⚙️ Configuración del análisis</div>', unsafe_allow_html=True)
    
    with st.container():
        # Primera fila de configuración
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            sensores = sorted(df["id_no2"].unique())
            sensor_seleccionado = st.selectbox("Sensor de NO₂", sensores, index=0)
            
            sensores_trafico = sorted(df[df["id_no2"] == sensor_seleccionado]["id_trafico"].unique())
            sensor_trafico = st.selectbox(
                "Sensor de tráfico asociado", 
                sensores_trafico, 
                index=0, 
                disabled=True
            ) if sensores_trafico else None
        
        with col2:
            fecha_min = df["fecha"].min().date()
            fecha_max = df["fecha"].max().date()
            fecha_inicio = st.date_input(
                "Fecha inicial", 
                fecha_min, 
                min_value=fecha_min, 
                max_value=fecha_max
            )
        
        with col3:
            fecha_fin = st.date_input(
                "Fecha final", 
                fecha_max, 
                min_value=fecha_min, 
                max_value=fecha_max
            )
            if fecha_inicio > fecha_fin:
                st.error("La fecha inicial debe ser anterior a la fecha final.")
                fecha_fin = fecha_inicio + timedelta(days=7)
        
        # Segunda fila de configuración
        col1, col2 = st.columns([1, 2])
        
        with col1:
            granularity = st.radio(
                "Granularidad temporal", 
                ["Horaria", "Diaria", "Semanal", "Mensual"], 
                horizontal=True
            )
        
        with col2:
            variables_disponibles = [
                'intensidad', 'carga', 'ocupacion', 
                'd2m', 't2m', 'sst', 'ssrd', 
                'u10', 'v10', 'sp', 'tp'
            ]
            variables_disponibles = [var for var in variables_disponibles if var in df.columns]
            
            variables_seleccionadas = st.multiselect(
                "Variables a comparar con NO₂", 
                variables_disponibles, 
                default=variables_disponibles
            )
    
    # ---------- Filtrado y procesamiento de datos ----------
    df_filtrado = df[
        (df["id_no2"] == sensor_seleccionado) & 
        (df["fecha"].dt.date >= fecha_inicio) & 
        (df["fecha"].dt.date <= fecha_fin)
    ].copy()
    
    if sensor_trafico:
        df_filtrado = df_filtrado[df_filtrado["id_trafico"] == sensor_trafico]
    
    # Procesar según granularidad
    granularity_map = {
        "Horaria": "H", 
        "Diaria": "D", 
        "Semanal": "W", 
        "Mensual": "M"
    }
    
    if granularity == "Horaria":
        df_filtrado["time_group"] = df_filtrado["fecha"].dt.floor(granularity_map[granularity])
        df_agregado = df_filtrado.copy()
    else:
        if granularity in ["Semanal", "Mensual"]:
            df_filtrado["time_group"] = df_filtrado["fecha"].dt.to_period(granularity_map[granularity]).dt.to_timestamp()
        else:
            df_filtrado["time_group"] = df_filtrado["fecha"].dt.floor(granularity_map[granularity])
            
        agg_vars = ["no2_value"] + variables_seleccionadas
        df_agregado = df_filtrado.groupby("time_group").agg({var: "mean" for var in agg_vars}).reset_index()
    
    # ---------- Visualizaciones ----------
    if df_filtrado.empty:
        st.warning("No hay datos disponibles para los filtros seleccionados. Por favor, ajusta los criterios de selección.")
        return
    
    # Mostrar tabs con diferentes visualizaciones
    st.markdown('<div class="section-header">📊 Visualizaciones y análisis</div>', unsafe_allow_html=True)
    
    # Tab principal
    tab1, tab2, tab3 = st.tabs([
        "📈 Análisis temporal", 
        "🔄 Correlaciones", 
        "📋 Disponibilidad de datos"
    ])
    
    # ---- Tab 1: Análisis temporal ----
    with tab1:
        st.subheader("Series temporales de NO₂ y variables seleccionadas")
        
        # Colores para las diferentes variables
        colores_variables = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']
        
        # Mostrar gráficos de series temporales
        if variables_seleccionadas:
            # Organizar en dos columnas
            for i in range(0, len(variables_seleccionadas), 2):
                col1, col2 = st.columns(2)
                
                # Primera columna
                with col1:
                    variable = variables_seleccionadas[i]
                    color_variable = colores_variables[i % len(colores_variables)]
                    st.pyplot(create_time_series_plot(df_agregado, variable, color_variable))
                
                # Segunda columna (si hay suficientes variables)
                if i + 1 < len(variables_seleccionadas):
                    with col2:
                        variable = variables_seleccionadas[i + 1]
                        color_variable = colores_variables[(i + 1) % len(colores_variables)]
                        st.pyplot(create_time_series_plot(df_agregado, variable, color_variable))
        else:
            st.info("Selecciona al menos una variable para visualizar series temporales.")
    
    # ---- Tab 2: Correlaciones ----
    with tab2:
        if variables_seleccionadas:
            # Matriz de correlación

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Matriz de correlación")
                st.pyplot(mostrar_matriz_correlacion(df_agregado, variables_seleccionadas))
            
            
            # Subtabs para diferentes tipos de gráficos de correlación
            scatter_tab1, scatter_tab2 = st.tabs(["Gráficos estáticos", "Gráficos interactivos"])
            
            with scatter_tab1:
                st.subheader("Gráficos de dispersión con líneas de regresión")
                for i in range(0, len(variables_seleccionadas), 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        variable = variables_seleccionadas[i]
                        color_variable = colores_variables[i % len(colores_variables)]
                        st.pyplot(create_scatter_plot(df_agregado, variable, color_variable))
                    
                    if i + 1 < len(variables_seleccionadas):
                        with col2:
                            variable = variables_seleccionadas[i + 1]
                            color_variable = colores_variables[(i + 1) % len(colores_variables)]
                            st.pyplot(create_scatter_plot(df_agregado, variable, color_variable))
            
            with scatter_tab2:
                st.subheader("Gráficos interactivos de dispersión")
                for i in range(0, len(variables_seleccionadas), 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        variable = variables_seleccionadas[i]
                        color_variable = colores_variables[i % len(colores_variables)]
                        st.altair_chart(
                            create_altair_scatter(df_agregado, variable, color_variable),
                            use_container_width=True
                        )
                    
                    if i + 1 < len(variables_seleccionadas):
                        with col2:
                            variable = variables_seleccionadas[i + 1]
                            color_variable = colores_variables[(i + 1) % len(colores_variables)]
                            st.altair_chart(
                                create_altair_scatter(df_agregado, variable, color_variable),
                                use_container_width=True
                            )
        else:
            st.info("Selecciona al menos una variable para analizar correlaciones.")
    
    # ---- Tab 3: Disponibilidad de datos ----
    with tab3:
        
        st.subheader("Análisis de disponibilidad de datos")
        disponibilidad = mostrar_disponibilidad_datos(df_filtrado)
        renderizar_disponibilidad(disponibilidad)
