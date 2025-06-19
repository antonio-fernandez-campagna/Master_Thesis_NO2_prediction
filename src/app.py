"""
Aplicación principal refactorizada para análisis de contaminación y tráfico en Madrid.

Esta aplicación proporciona múltiples módulos de análisis a través de un sistema de tabs,
cada uno con su propio sidebar y configuración específica.
"""

import streamlit as st
from typing import Dict, Callable
import sys
import os
import pandas as pd

# Configuración de rutas y formato de números
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pd.options.display.float_format = "{:.2f}".format

# Importar configuración centralizada
from src.config import PAGE_CONFIG, TAB_CONFIG

# Importar módulos refactorizados
from src.welcome_page import welcome_page
from src.no2_analysis import generar_analisis_no2
from src.sensor_mapping import generar_mapa_asignaciones
from src.correlations_analysis import analisis_sensores
from src.gam_training import training_page
from src.xgboost_unified import xgboost_unified_page
from src.bayesian_nowcasting import bayesian_nowcasting_page

# ==================== CONFIGURACIÓN DE TABS CON FUNCIONES ====================

TAB_FUNCTIONS = {
    "🏠 Inicio": welcome_page,
    "Análisis NO₂": generar_analisis_no2,
    "Mapeo Sensores": generar_mapa_asignaciones,
    "Correlaciones": analisis_sensores,
    "Entrenamiento GAM": training_page,
    "XGBoost Unificado": xgboost_unified_page,
    "Nowcasting Bayesiano": bayesian_nowcasting_page
}

# ==================== CLASE PRINCIPAL ====================

class DashboardApp:
    """Clase principal para manejar la aplicación dashboard."""
    
    def __init__(self):
        self._configure_page()
        self._initialize_session_state()
    
    def _configure_page(self):
        """Configura la página de Streamlit."""
        st.set_page_config(**PAGE_CONFIG)
    
    def _initialize_session_state(self):
        """Inicializa variables globales de session_state."""
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = list(TAB_CONFIG.keys())[0]
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
    
    def _clear_sidebar_for_tab(self, tab_name: str):
        """Limpia el sidebar cuando cambia de tab."""
        # Solo limpiar si realmente cambió de tab
        if st.session_state.get('previous_tab') != tab_name:
            # Limpiar variables específicas del sidebar del tab anterior
            keys_to_remove = [
                key for key in st.session_state.keys() 
                if key.startswith(('sidebar_', 'filter_', 'config_'))
            ]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.session_state.previous_tab = tab_name
    
    def _show_header(self):
        """Muestra el header principal de la aplicación."""
        st.title("🌍 Dashboard Madrid - Calidad del Aire y Tráfico")
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <p style="font-size: 1.1rem; color: #666;">
                Análisis integrado de datos de contaminación atmosférica y tráfico en la ciudad de Madrid
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_tab_description(self, tab_name: str):
        """Muestra la descripción del tab actual."""
        config = TAB_CONFIG.get(tab_name, {})
        if config.get('description'):
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #1f4e79;">
                    {config.get('icon', '')} {tab_name}
                </h4>
                <p style="margin: 0.5rem 0 0 0; color: #666;">
                    {config['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _create_tabs(self) -> Dict:
        """Crea y configura los tabs de la aplicación."""
        tab_names = list(TAB_CONFIG.keys())
        tab_labels = [f"{TAB_CONFIG[name]['icon']} {name}" for name in tab_names]
        
        tabs = st.tabs(tab_labels)
        return dict(zip(tab_names, tabs))
    
    def _execute_tab_function(self, tab_name: str, tab_container):
        """Ejecuta la función asociada a un tab específico."""
        config = TAB_CONFIG.get(tab_name)
        
        if not config:
            st.error(f"Configuración no encontrada para el tab: {tab_name}")
            return
        
        function = TAB_FUNCTIONS.get(tab_name)
        
        if not function:
            st.error(f"Función no definida para el tab: {tab_name}")
            return
        
        try:
            with tab_container:
                # Mostrar descripción del tab
                self._show_tab_description(tab_name)
                
                # Limpiar sidebar si es necesario
                self._clear_sidebar_for_tab(tab_name)
                
                # Ejecutar función del tab
                function()
                
        except Exception as e:
            st.error(f"Error al ejecutar {tab_name}: {str(e)}")
            st.exception(e)
    
    def _show_footer(self):
        """Muestra el footer de la aplicación."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
            <p>
                <strong>Dashboard Madrid - Calidad del Aire y Tráfico</strong><br>
                Datos proporcionados por el Ayuntamiento de Madrid<br>
                Desarrollado para análisis de investigación científica
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Método principal para ejecutar la aplicación."""
        try:
            # Mostrar header
            self._show_header()
            
            # Crear tabs
            tabs = self._create_tabs()
            
            # Ejecutar funciones de cada tab
            for tab_name, tab_container in tabs.items():
                self._execute_tab_function(tab_name, tab_container)
            
            # Mostrar footer
            self._show_footer()
            
        except Exception as e:
            st.error("Error crítico en la aplicación")
            st.exception(e)


# ==================== FUNCIONES AUXILIARES ====================

def show_sidebar_info():
    """Muestra información general en el sidebar."""
    with st.sidebar:
        st.markdown("## ℹ️ Información")
        st.markdown("""
        **Navegación:**
        - Usa las pestañas superiores para cambiar entre módulos
        - Cada módulo tiene controles específicos en este panel lateral
        - Los datos se cargan automáticamente al acceder a cada sección
        """)
        
        st.markdown("---")
        st.markdown("**Estado de la aplicación:**")
        
        # Mostrar estado de carga de datos
        if hasattr(st.session_state, 'data_loaded'):
            if st.session_state.data_loaded:
                st.success("✅ Datos de NO₂ cargados")
            else:
                st.info("⏳ Datos de NO₂ pendientes")
        
        if hasattr(st.session_state, 'mapping_data_loaded'):
            if st.session_state.mapping_data_loaded:
                st.success("✅ Datos de mapeo cargados")
            else:
                st.info("⏳ Datos de mapeo pendientes")
        
        if hasattr(st.session_state, 'sensor_data_loaded'):
            if st.session_state.sensor_data_loaded:
                st.success("✅ Datos de sensores cargados")
            else:
                st.info("⏳ Datos de sensores pendientes")
        
        if hasattr(st.session_state, 'training_data_loaded'):
            if st.session_state.training_data_loaded:
                st.success("✅ Datos de entrenamiento cargados")
            else:
                st.info("⏳ Datos de entrenamiento pendientes")


def handle_navigation():
    """Maneja la navegación entre tabs."""
    # Esta función puede expandirse para manejar navegación más compleja
    # Por ejemplo, deep linking, estado persistente entre tabs, etc.
    pass


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal de la aplicación."""
    # Crear y ejecutar la aplicación
    app = DashboardApp()
    
    # Mostrar información general en sidebar
    show_sidebar_info()
    
    # Manejar navegación
    handle_navigation()
    
    # Ejecutar aplicación principal
    app.run()


if __name__ == "__main__":
    main() 