import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtrado' not in st.session_state:
    st.session_state.df_filtrado = None
if 'df_agregado' not in st.session_state:
    st.session_state.df_agregado = None
if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = False

@st.cache_data(ttl=3600)
def cargar_datos_trafico_y_meteo():
    """Carga y preprocesa los datos de aire con caché"""
    df = pd.read_parquet('data/more_processed/test.parquet')
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def cargar_datos_callback():
    """Callback para cargar datos cuando se presiona el botón"""
    try:
        st.session_state.df = cargar_datos_trafico_y_meteo()
        st.session_state.datos_cargados = True
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.info("Asegúrate de que el archivo de datos existe y es accesible.")

def filtrar_datos():
    """Función para filtrar y agregar los datos según las selecciones del usuario"""
    # Solo ejecutar si los datos han sido cargados
    if not st.session_state.datos_cargados:
        return
    
    # Filtrar datos según el sensor seleccionado y el rango de fechas
    st.session_state.df_filtrado = st.session_state.df[
        (st.session_state.df["id_no2"] == st.session_state.sensor_seleccionado) & 
        (st.session_state.df["fecha"].dt.date >= st.session_state.fecha_inicio) & 
        (st.session_state.df["fecha"].dt.date <= st.session_state.fecha_fin)
    ].copy()
    
    # Configurar la granularidad temporal
    if st.session_state.granularity == "Horaria":
        st.session_state.df_filtrado["time_group"] = st.session_state.df_filtrado["fecha"].dt.floor("H")
        st.session_state.format_str = "%Y-%m-%d %H:%M"
    elif st.session_state.granularity == "Diaria":
        st.session_state.df_filtrado["time_group"] = st.session_state.df_filtrado["fecha"].dt.floor("D")
        st.session_state.format_str = "%Y-%m-%d"
    elif st.session_state.granularity == "Semanal":
        st.session_state.df_filtrado["time_group"] = st.session_state.df_filtrado["fecha"].dt.to_period("W").dt.to_timestamp()
        st.session_state.format_str = "%Y-%m-%d"
    else:  # Mensual
        st.session_state.df_filtrado["time_group"] = st.session_state.df_filtrado["fecha"].dt.to_period("M").dt.to_timestamp()
        st.session_state.format_str = "%Y-%m"
    
    # Agregar datos según granularidad
    if st.session_state.granularity != "Horaria":
        variables_agg = ["no2_value"] + st.session_state.variables_seleccionadas
        st.session_state.df_agregado = st.session_state.df_filtrado.groupby(["time_group"]).agg(
            {var: "mean" for var in variables_agg}
        ).reset_index()
    else:
        st.session_state.df_agregado = st.session_state.df_filtrado

def update_sensor_callback():
    """Callback para actualizar cuando cambia el sensor"""
    filtrar_datos()

def update_fecha_callback():
    """Callback para actualizar cuando cambian las fechas"""
    # Verificar que la fecha inicial sea anterior a la fecha final
    if st.session_state.fecha_inicio > st.session_state.fecha_fin:
        st.session_state.fecha_fin = st.session_state.fecha_inicio + timedelta(days=7)
    filtrar_datos()

def update_granularity_callback():
    """Callback para actualizar cuando cambia la granularidad"""
    filtrar_datos()

def update_variables_callback():
    """Callback para actualizar cuando cambian las variables seleccionadas"""
    filtrar_datos()

def analisis_sensores():
    """Función para el análisis comparativo de sensores de tráfico y NO2."""
    
    st.markdown('<div class="sub-header">🚗 Análisis de sensores de tráfico y su relación con NO₂</div>', unsafe_allow_html=True)
    
    # Información contextual
    with st.expander("ℹ️ Acerca de este análisis", expanded=False):
        st.markdown("""
        <div class="info-box">
        <p>Este panel permite analizar la relación entre los datos de tráfico y los niveles de NO₂ en Madrid.</p>
        <p><strong>Cómo usar:</strong></p>
        <ul>
            <li>Selecciona un sensor específico para analizar</li>
            <li>Elige un rango de fechas para filtrar los datos</li>
            <li>Selecciona la granularidad temporal para visualizar las tendencias</li>
            <li>Explora los gráficos comparativos entre NO₂ y distintas variables de tráfico/meteorológicas</li>
        </ul>
        <p><strong>Nota:</strong> Las variables analizadas incluyen intensidad de tráfico, carga, ocupación y variables meteorológicas.</p>
        </div>
        """, unsafe_allow_html=True)

    # Botón para cargar datos (solo se ejecuta una vez)
    if not st.session_state.datos_cargados:
        if st.button("Cargar datos"):
            with st.spinner('Cargando datos...'):
                cargar_datos_callback()
    
    # Si los datos están cargados, mostrar los controles y visualizaciones
    if st.session_state.datos_cargados:
        
        # --- Controles de configuración ---
        st.markdown('<div class="sub-header">⚙️ Configuración</div>', unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Selector de sensor (obligatorio para este análisis)
                sensores = sorted(st.session_state.df["id_no2"].unique())
                
                # Inicializar el valor en session state si no existe
                if 'sensor_seleccionado' not in st.session_state:
                    st.session_state.sensor_seleccionado = sensores[0]
                
                # Usar el widget con session_state y on_change callback
                st.selectbox(
                    "Selecciona un sensor de NO₂",
                    sensores,
                    index=sensores.index(st.session_state.sensor_seleccionado),
                    key='sensor_seleccionado',
                    on_change=update_sensor_callback
                )
                
                # Filtro de fechas
                st.markdown("#### 📅 Rango de fechas")
                fecha_min = st.session_state.df["fecha"].min().date()
                fecha_max = st.session_state.df["fecha"].max().date()
                
                # Inicializar fechas en session state si no existen
                if 'fecha_inicio' not in st.session_state:
                    st.session_state.fecha_inicio = fecha_min
                if 'fecha_fin' not in st.session_state:
                    st.session_state.fecha_fin = fecha_max
                
                # Usar date_input con session_state y on_change callback
                st.date_input(
                    "Fecha inicial", 
                    st.session_state.fecha_inicio, 
                    min_value=fecha_min, 
                    max_value=fecha_max, 
                    key="fecha_inicio",
                    on_change=update_fecha_callback
                )
                
                st.date_input(
                    "Fecha final", 
                    st.session_state.fecha_fin, 
                    min_value=fecha_min, 
                    max_value=fecha_max, 
                    key="fecha_fin",
                    on_change=update_fecha_callback
                )
                
                if st.session_state.fecha_inicio > st.session_state.fecha_fin:
                    st.error("⚠️ La fecha inicial debe ser anterior a la fecha final")
                
                # Granularidad temporal
                st.markdown("#### ⏱️ Agregación temporal")
                
                # Inicializar granularidad en session state si no existe
                if 'granularity' not in st.session_state:
                    st.session_state.granularity = "Horaria"
                
                # Usar radio con session_state y on_change callback
                st.radio(
                    "Granularidad", 
                    ["Horaria", "Diaria", "Semanal", "Mensual"], 
                    index=["Horaria", "Diaria", "Semanal", "Mensual"].index(st.session_state.granularity),
                    key="granularity",
                    on_change=update_granularity_callback,
                    horizontal=True
                )
                
                # Variables a analizar
                st.markdown("#### 📊 Variables a analizar")
                variables_disponibles = ['intensidad', 'carga', 'ocupacion', 'd2m', 't2m', 'sst', 'ssrd', 'u10', 'v10', 'sp', 'tp']
            
                # Filtrar variables que existen en el DataFrame
                variables_disponibles = [var for var in variables_disponibles if var in st.session_state.df.columns]
                
                # Inicializar variables seleccionadas en session state si no existen
                if 'variables_seleccionadas' not in st.session_state:
                    st.session_state.variables_seleccionadas = variables_disponibles[:3]  # Primeras 3 variables por defecto
                
                # Usar multiselect con session_state y on_change callback
                st.multiselect(
                    "Selecciona variables a comparar con NO₂",
                    variables_disponibles,
                    default=st.session_state.variables_seleccionadas,
                    key="variables_seleccionadas",
                    on_change=update_variables_callback
                )
                
                # Opciones de visualización
                st.markdown("#### 🔆 Opciones de visualización")
                
                # Inicializar opciones de visualización en session state si no existen
                if 'mostrar_correlacion' not in st.session_state:
                    st.session_state.mostrar_correlacion = True
                if 'tipo_grafico' not in st.session_state:
                    st.session_state.tipo_grafico = "Líneas"
                
                # Usar checkbox y radio con session_state
                st.checkbox(
                    "Mostrar matriz de correlación", 
                    value=st.session_state.mostrar_correlacion,
                    key="mostrar_correlacion"
                )
                
                st.radio(
                    "Tipo de gráfico", 
                    ["Líneas", "Dispersión"], 
                    index=["Líneas", "Dispersión"].index(st.session_state.tipo_grafico),
                    key="tipo_grafico",
                    horizontal=True
                )
                
                # Primera vez que se carga, filtrar los datos
                if st.session_state.df_filtrado is None:
                    filtrar_datos()
                
                # Mostrar información básica del sensor seleccionado si hay datos filtrados
                if st.session_state.df_filtrado is not None:
                    st.markdown('<div class="sub-header">📍 Información del sensor seleccionado</div>', unsafe_allow_html=True)
                    st.markdown(f"**ID del sensor:** {st.session_state.sensor_seleccionado}")
                    st.markdown(f"**Datos disponibles:** {st.session_state.df_filtrado.shape[0]} registros")
                    st.markdown(f"**Periodo analizado:** {st.session_state.fecha_inicio} a {st.session_state.fecha_fin}")
            
            with col2:
                if st.session_state.df_filtrado is None or st.session_state.df_filtrado.empty:
                    st.error("⚠️ No hay datos disponibles para el sensor y rango de fechas seleccionados.")
                    return
                
                # --- Sección de visualizaciones ---
                st.markdown('<div class="sub-header">📊 Análisis comparativo de variables</div>', unsafe_allow_html=True)

                # Para la matriz de correlación
                if st.session_state.mostrar_correlacion and len(st.session_state.variables_seleccionadas) >= 1:
                    st.markdown("### Matriz de correlación")
                    variables_corr = ["no2_value"] + st.session_state.variables_seleccionadas
                    corr_data = st.session_state.df_agregado[variables_corr].corr()

                    fig, ax = plt.subplots(figsize=(8, 4))
                    mask = np.triu(np.ones_like(corr_data, dtype=bool))
                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                    sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                                annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
                    plt.title('Correlación entre NO₂ y variables seleccionadas')
                    # Usar st.pyplot para mostrar el gráfico en Streamlit
                    st.pyplot(fig)
                    
                    # Mostrar interpretación de las correlaciones más fuertes
                    max_corr_var = corr_data['no2_value'].drop('no2_value').abs().idxmax()
                    max_corr_val = corr_data.loc['no2_value', max_corr_var]
                    
                    st.markdown(f"""
                    **Interpretación:**
                    - La variable con mayor correlación con los niveles de NO₂ es **{max_corr_var}** (r = {max_corr_val:.2f})
                    - {'Existe una correlación positiva, lo que sugiere que a mayor ' if max_corr_val > 0 else 'Existe una correlación negativa, lo que sugiere que a menor '} 
                    {max_corr_var}, {'mayores' if max_corr_val > 0 else 'menores'} son los niveles de NO₂.
                    """)
        
        if len(st.session_state.variables_seleccionadas) > 0:
            # Crear gráficos comparativos
            try:
                # Gráficos de series temporales para cada variable seleccionada
                st.markdown("### Evolución temporal comparativa")

                for variable in st.session_state.variables_seleccionadas:
                    # Opción 1: Usar Matplotlib con st.pyplot
                    if st.session_state.tipo_grafico == "Líneas":
                        fig, ax1 = plt.subplots(figsize=(12, 6))
                        
                        # Configurar eje y para NO2
                        color = 'tab:blue'
                        ax1.set_xlabel('Fecha')
                        ax1.set_ylabel('NO₂ (μg/m³)', color=color)
                        ax1.plot(st.session_state.df_agregado['time_group'], st.session_state.df_agregado['no2_value'], color=color, marker='o', label='NO₂')
                        ax1.tick_params(axis='y', labelcolor=color)
                        
                        # Agregar línea de referencia de la OMS (40 μg/m³)
                        ax1.axhline(y=40, color='r', linestyle='--', alpha=0.7)
                        ax1.text(st.session_state.df_agregado['time_group'].iloc[0], 41, 'Límite OMS', color='r', fontsize=10)
                        
                        # Configurar eje y secundario para la variable seleccionada
                        ax2 = ax1.twinx()
                        color = 'tab:red'
                        ax2.set_ylabel(f'{variable.capitalize()}', color=color)
                        ax2.plot(st.session_state.df_agregado['time_group'], st.session_state.df_agregado[variable], color=color, marker='x', label=variable)
                        ax2.tick_params(axis='y', labelcolor=color)
                        
                        # Formatear fechas en el eje x
                        fig.autofmt_xdate()
                        plt.title(f'Comparativa NO₂ vs {variable.capitalize()}')
                        plt.grid(True, alpha=0.3)
                        
                        # Crear leyenda combinada
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        
                        plt.tight_layout()
                        # Usar st.pyplot para mostrar en Streamlit
                        st.pyplot(fig)
                    
                    # Opción 2: Usar Altair para gráficos interactivos
                    else:  # Tipo "Dispersión" - usamos Altair que es más interactivo en Streamlit
                        # Preparar datos para Altair
                        chart_data = st.session_state.df_agregado[['time_group', 'no2_value', variable]].copy()
                        
                        # Asegurarnos que el formato de fecha es correcto para Altair
                        chart_data['time_group'] = pd.to_datetime(chart_data['time_group'])
                        
                        # Crear gráfico base
                        base = alt.Chart(chart_data).encode(
                            x=alt.X('time_group:T', title='Fecha')
                        )
                        
                        # Gráfico para NO2
                        no2_chart = base.mark_circle(color='blue', opacity=0.7).encode(
                            y=alt.Y('no2_value:Q', title='NO₂ (μg/m³)', scale=alt.Scale(zero=False)),
                            tooltip=['time_group:T', alt.Tooltip('no2_value:Q', title='NO₂ (μg/m³)', format='.1f')]
                        )
                        
                        # Gráfico para la variable seleccionada
                        var_chart = base.mark_circle(color='red', opacity=0.7).encode(
                            y=alt.Y(f'{variable}:Q', title=variable.capitalize(), scale=alt.Scale(zero=False)),
                            tooltip=['time_group:T', alt.Tooltip(f'{variable}:Q', title=variable.capitalize(), format='.1f')]
                        )
                        
                        # Línea de referencia de la OMS
                        limit_line = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(color='red', strokeDash=[3, 3]).encode(y='y:Q')
                        
                        # Combinar gráficos con escalas independientes
                        chart = alt.layer(no2_chart, limit_line).resolve_scale(y='independent') | var_chart
                        
                        # Configurar propiedades del gráfico
                        chart = chart.properties(
                            title=f'Comparativa NO₂ vs {variable.capitalize()}',
                            width=400,
                            height=300
                        ).interactive()
                        
                        # Mostrar en Streamlit
                        st.altair_chart(chart, use_container_width=True)
                    
                    # Análisis de relación entre la variable y NO2
                    corr = st.session_state.df_agregado[['no2_value', variable]].corr().iloc[0, 1]
                    
                    st.markdown(f"""
                    **Análisis de la relación NO₂ - {variable.capitalize()}:**
                    - Coeficiente de correlación: {corr:.2f}
                    - Interpretación: {'Correlación fuerte' if abs(corr) > 0.7 else 'Correlación moderada' if abs(corr) > 0.4 else 'Correlación débil'}
                    """)
                    
                    # Gráfico de dispersión
                    scatter_data = st.session_state.df_agregado[[variable, 'no2_value']].copy()
                    scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
                        x=alt.X(f'{variable}:Q', title=variable.capitalize()),
                        y=alt.Y('no2_value:Q', title='NO₂ (μg/m³)'),
                        tooltip=[
                            alt.Tooltip(f'{variable}:Q', title=variable.capitalize(), format='.2f'),
                            alt.Tooltip('no2_value:Q', title='NO₂ (μg/m³)', format='.2f')
                        ]
                    ).properties(
                        width=600,
                        height=400,
                        title=f'Dispersión NO₂ vs {variable.capitalize()} (r={corr:.2f})'
                    ).interactive()
                    
                    # Agregar línea de regresión
                    regression = scatter_chart.transform_regression(
                        variable, 'no2_value'
                    ).mark_line(color='red', strokeDash=[4, 2])
                    
                    # Mostrar en Streamlit
                    st.altair_chart(scatter_chart + regression, use_container_width=True)
                    
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error al generar gráficos: {str(e)}")
                import traceback
                st.info(traceback.format_exc())
        else:
            st.warning("⚠️ Por favor, selecciona al menos una variable para comparar con NO₂.")

        st.markdown("""
        <div style="margin-top: 2rem; text-align: center; color: #666; font-size: 0.8rem;">
            Este análisis permite identificar correlaciones entre variables de tráfico/meteorológicas y concentraciones de NO₂.
        </div>
        """, unsafe_allow_html=True)