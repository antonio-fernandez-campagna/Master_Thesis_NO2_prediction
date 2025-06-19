"""
Clase base para entrenadores de modelos de predicción de NO2.

Esta clase base contiene toda la funcionalidad común que se comparte entre
los diferentes algoritmos de machine learning (XGBoost, GAM, Bayesian).
"""

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

from ..config import FILE_PATHS, get_model_config, get_session_state_key
from ..utils.data_processing import (
    load_master_data, prepare_data_for_training, prepare_matrices_for_training,
    scale_features, scale_target
)
from ..utils.session_management import (
    initialize_common_session_state, initialize_unified_session_state,
    store_model_results, get_model_results, show_variable_selection_interface,
    show_individual_configuration, show_global_configuration, show_configuration_summary
)
from ..utils.visualization import show_data_overview

warnings.filterwarnings('ignore')


class BaseTrainer(ABC):
    """
    Clase base abstracta para todos los entrenadores de modelos.
    
    Contiene toda la funcionalidad común y define la interfaz que deben
    implementar las clases específicas de cada algoritmo.
    """
    
    def __init__(self, model_type: str):
        """
        Inicializa el entrenador base.
        
        Args:
            model_type: Tipo de modelo ('xgboost', 'gam', 'bayesian')
        """
        self.model_type = model_type
        self.df_master = None
        self.model = None
        self.scaler_dict = {}
        self.scaler_target = None
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de la sesión para este tipo de modelo."""
        initialize_common_session_state(self.model_type)
    
    # ==================== CARGA Y GESTIÓN DE DATOS ====================
    
    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos principales utilizando cache.
        
        Returns:
            DataFrame con los datos cargados
        """
        return load_master_data()
    
    def show_data_overview(self):
        """Muestra overview del dataset cargado."""
        if self.df_master is not None and not self.df_master.empty:
            show_data_overview(self.df_master, f"Overview del Dataset - {self.model_type.upper()}")
        else:
            st.warning("No hay datos cargados para mostrar overview.")
    
    # ==================== INTERFAZ DE CONFIGURACIÓN ====================
    
    def show_variable_selection(self, key_prefix: str = "") -> List[str]:
        """
        Muestra interfaz de selección de variables.
        
        Args:
            key_prefix: Prefijo para las claves de session_state
            
        Returns:
            Lista de variables seleccionadas
        """
        if self.df_master is None or self.df_master.empty:
            st.error("No hay datos cargados para seleccionar variables.")
            return []
        
        return show_variable_selection_interface(self.df_master, self.model_type, key_prefix)
    
    def show_individual_config_panel(self, key_prefix: str = "") -> Dict:
        """
        Muestra panel de configuración para modelos individuales.
        
        Args:
            key_prefix: Prefijo para las claves de session_state
            
        Returns:
            Diccionario con la configuración
        """
        if self.df_master is None or self.df_master.empty:
            st.error("No hay datos cargados para configurar modelo.")
            return {}
        
        return show_individual_configuration(self.df_master, self.model_type, key_prefix)
    
    def show_global_config_panel(self, key_prefix: str = "") -> Dict:
        """
        Muestra panel de configuración para modelos globales.
        
        Args:
            key_prefix: Prefijo para las claves de session_state
            
        Returns:
            Diccionario con la configuración
        """
        if self.df_master is None or self.df_master.empty:
            st.error("No hay datos cargados para configurar modelo.")
            return {}
        
        return show_global_configuration(self.df_master, self.model_type, key_prefix)
    
    def show_config_summary(self, config: Dict, selected_features: List[str], mode: str = None):
        """
        Muestra resumen de configuración.
        
        Args:
            config: Configuración del modelo
            selected_features: Variables seleccionadas
            mode: Modo específico ('individual', 'global') si aplica
        """
        show_configuration_summary(config, selected_features, self.model_type, mode)
    
    # ==================== PREPARACIÓN DE DATOS ====================
    
    def prepare_training_data(self, df: pd.DataFrame, selected_features: List[str],
                            split_date: pd.Timestamp, outlier_method: str = 'none',
                            preprocessing: str = 'none') -> Dict:
        """
        Prepara los datos para entrenamiento utilizando el pipeline común.
        
        Args:
            df: DataFrame con los datos
            selected_features: Lista de variables seleccionadas
            split_date: Fecha de división
            outlier_method: Método de filtrado de outliers
            preprocessing: Tipo de preprocesamiento
            
        Returns:
            Diccionario con datos preparados
        """
        return prepare_data_for_training(df, selected_features, split_date, 
                                       outlier_method, preprocessing)
    
    def prepare_training_matrices(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                selected_features: List[str], target_col: str = 'no2_value') -> Dict:
        """
        Prepara matrices X, y para entrenamiento.
        
        Args:
            train_df: DataFrame de entrenamiento
            test_df: DataFrame de prueba
            selected_features: Lista de variables seleccionadas
            target_col: Nombre de la columna objetivo
            
        Returns:
            Diccionario con matrices preparadas
        """
        return prepare_matrices_for_training(train_df, test_df, selected_features, target_col)
    
    def scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, selected_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict, Any]:
        """
        Escala datos de entrenamiento y prueba.
        
        Args:
            X_train: Variables predictoras de entrenamiento
            X_test: Variables predictoras de prueba
            y_train: Variable objetivo de entrenamiento
            selected_features: Lista de variables a escalar
            
        Returns:
            Tupla con (X_train_scaled, X_test_scaled, y_train_scaled, scaler_dict, scaler_target)
        """
        X_train_scaled, X_test_scaled, scaler_dict = scale_features(X_train, X_test, selected_features)
        y_train_scaled, scaler_target = scale_target(y_train)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, scaler_dict, scaler_target
    
    # ==================== EVALUACIÓN COMÚN ====================
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evalúa las predicciones calculando métricas estándar.
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'y_pred': y_pred
        }
        
        # Métricas adicionales
        residuals = y_true - y_pred
        metrics['bias'] = np.mean(residuals)
        metrics['std_residuals'] = np.std(residuals)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return metrics
    
    # ==================== GUARDADO Y CARGA DE MODELOS ====================
    
    def save_model(self, model: Any, feature_names: List[str], scaler_dict: Dict,
                   scaler_target: Any, sensor_id: str, outlier_method: str,
                   preprocessing: str, additional_info: Dict = None) -> str:
        """
        Guarda el modelo y sus metadatos.
        
        Args:
            model: Modelo entrenado
            feature_names: Lista de nombres de características
            scaler_dict: Diccionario de escaladores para características
            scaler_target: Escalador para variable objetivo
            sensor_id: ID del sensor (para modelos individuales)
            outlier_method: Método de outliers utilizado
            preprocessing: Método de preprocesamiento utilizado
            additional_info: Información adicional específica del modelo
            
        Returns:
            Ruta del archivo guardado
        """
        model_info = {
            'model': model,
            'feature_names': feature_names,
            'scaler_dict': scaler_dict,
            'scaler_target': scaler_target,
            'sensor_id': sensor_id,
            'outlier_method': outlier_method,
            'preprocessing': preprocessing,
            'model_type': self.model_type
        }
        
        if additional_info:
            model_info.update(additional_info)
        
        model_dir = FILE_PATHS['models_dir']
        os.makedirs(model_dir, exist_ok=True)
        filename = f'{model_dir}/{self.model_type}_model_{sensor_id}_{outlier_method}_{preprocessing}.pkl'
        
        joblib.dump(model_info, filename)
        return filename
    
    def load_model(self, filepath: str) -> Optional[Dict]:
        """
        Carga un modelo guardado.
        
        Args:
            filepath: Ruta del archivo del modelo
            
        Returns:
            Diccionario con información del modelo o None si hay error
        """
        try:
            return joblib.load(filepath)
        except Exception as e:
            st.error(f"Error al cargar modelo: {str(e)}")
            return None
    
    # ==================== GESTIÓN DE RESULTADOS ====================
    
    def store_results(self, results: Dict, config_key: str, mode: str = None):
        """
        Almacena resultados en session state.
        
        Args:
            results: Diccionario con resultados
            config_key: Clave única de configuración
            mode: Modo específico si aplica
        """
        store_model_results(self.model_type, results, config_key, mode)
    
    def get_stored_results(self, config_key: str, mode: str = None) -> Optional[Dict]:
        """
        Obtiene resultados almacenados.
        
        Args:
            config_key: Clave única de configuración
            mode: Modo específico si aplica
            
        Returns:
            Diccionario con resultados o None
        """
        return get_model_results(self.model_type, config_key, mode)
    
    def generate_config_key(self, **kwargs) -> str:
        """
        Genera clave única para configuración.
        
        Args:
            **kwargs: Parámetros de configuración
            
        Returns:
            Clave única como string
        """
        return get_session_state_key(self.model_type, **kwargs)
    
    # ==================== MÉTODOS ABSTRACTOS ====================
    
    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                   X_val: pd.DataFrame, y_val: np.ndarray, **kwargs) -> Any:
        """
        Entrena el modelo específico. Debe ser implementado por cada subclase.
        
        Args:
            X_train: Variables predictoras de entrenamiento
            y_train: Variable objetivo de entrenamiento
            X_val: Variables predictoras de validación
            y_val: Variable objetivo de validación
            **kwargs: Parámetros específicos del algoritmo
            
        Returns:
            Modelo entrenado
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo. Debe ser implementado por cada subclase.
        
        Args:
            model: Modelo entrenado
            X: Variables predictoras
            
        Returns:
            Array con predicciones
        """
        pass
    
    @abstractmethod
    def get_model_specific_metrics(self, model: Any, X_test: pd.DataFrame, 
                                 y_test: np.ndarray) -> Dict:
        """
        Calcula métricas específicas del modelo. Debe ser implementado por cada subclase.
        
        Args:
            model: Modelo entrenado
            X_test: Variables predictoras de prueba
            y_test: Variable objetivo de prueba
            
        Returns:
            Diccionario con métricas específicas
        """
        pass
    
    @abstractmethod
    def show_model_specific_analysis(self, results: Dict):
        """
        Muestra análisis específico del modelo. Debe ser implementado por cada subclase.
        
        Args:
            results: Diccionario con resultados del modelo
        """
        pass
    
    # ==================== PIPELINE COMPLETO ====================
    
    def full_training_pipeline(self, config: Dict, selected_features: List[str],
                             training_params: Dict = None) -> Optional[Dict]:
        """
        Pipeline completo de entrenamiento que puede ser reutilizado por las subclases.
        
        Args:
            config: Configuración del modelo
            selected_features: Variables seleccionadas
            training_params: Parámetros específicos de entrenamiento
            
        Returns:
            Diccionario con resultados completos o None si hay error
        """
        training_params = training_params or {}
        
        try:
            with st.spinner(f"Entrenando modelo {self.model_type.upper()}..."):
                # Preparar datos según el tipo de configuración
                if 'sensor' in config:
                    # Modelo individual
                    data_prep = self.prepare_training_data(
                        config['df_sensor'], selected_features,
                        pd.to_datetime(config['fecha_division']),
                        config['outlier_method'], config['preprocessing']
                    )
                else:
                    # Modelo global
                    df_train = self.df_master[self.df_master['id_no2'].isin(config['sensores_train'])]
                    df_test = self.df_master[self.df_master['id_no2'].isin(config['sensores_test'])]
                    
                    # Aplicar preprocesamiento
                    from ..utils.data_processing import create_cyclical_features, convert_units, remove_outliers
                    
                    if config['preprocessing'] == 'sin_cos':
                        df_train = create_cyclical_features(df_train)
                        df_test = create_cyclical_features(df_test)
                    
                    df_train = convert_units(df_train)
                    df_test = convert_units(df_test)
                    
                    # Eliminar outliers solo en entrenamiento
                    outliers_removed = 0
                    if config['outlier_method'] != 'none':
                        len_before = len(df_train)
                        df_train = remove_outliers(df_train, config['outlier_method'])
                        outliers_removed = len_before - len(df_train)
                    
                    data_prep = {
                        'train_df': df_train,
                        'test_df': df_test,
                        'outliers_removed': outliers_removed
                    }
                
                # Preparar matrices
                matrices = self.prepare_training_matrices(
                    data_prep['train_df'], data_prep['test_df'], selected_features
                )
                
                # Validar datos
                if matrices['train_samples_clean'] == 0 or matrices['test_samples_clean'] == 0:
                    st.error("❌ No hay datos suficientes después de la limpieza")
                    return None
                
                # Escalar datos
                X_train_scaled, X_test_scaled, y_train_scaled, scaler_dict, scaler_target = self.scale_data(
                    matrices['X_train'], matrices['X_test'], matrices['y_train'], selected_features
                )
                
                # Entrenar modelo específico
                model = self.train_model(
                    X_train_scaled, y_train_scaled, X_test_scaled, 
                    matrices['y_test'], **training_params
                )
                
                # Hacer predicciones
                y_pred_scaled = self.predict(model, X_test_scaled)
                y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # Evaluar modelo
                metrics = self.evaluate_predictions(matrices['y_test'], y_pred)
                
                # Agregar métricas específicas del modelo
                specific_metrics = self.get_model_specific_metrics(model, X_test_scaled, matrices['y_test'])
                metrics.update(specific_metrics)
                
                # Guardar modelo si es individual
                model_path = None
                if 'sensor' in config:
                    model_path = self.save_model(
                        model, selected_features, scaler_dict, scaler_target,
                        config['sensor'], config['outlier_method'], config['preprocessing']
                    )
                    st.success(f"✅ Modelo {self.model_type.upper()} entrenado y guardado en: {model_path}")
                else:
                    st.success(f"✅ Modelo {self.model_type.upper()} global entrenado exitosamente!")
                
                return {
                    'model': model,
                    'metrics': metrics,
                    'test_df': matrices['test_df'],
                    'scaler_dict': scaler_dict,
                    'scaler_target': scaler_target,
                    'selected_features': selected_features,
                    'config': config,
                    'model_path': model_path,
                    'data_prep': data_prep
                }
                
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
            return None


class UnifiedTrainerMixin:
    """
    Mixin que añade funcionalidad para manejo de modelos unificados (individual/global).
    
    Puede ser heredado junto con BaseTrainer para crear entrenadores unificados.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_unified_session_state()
    
    def _initialize_unified_session_state(self):
        """Inicializa estados específicos para modelos unificados."""
        initialize_unified_session_state(self.model_type)
    
    def get_training_mode(self) -> str:
        """
        Obtiene el modo de entrenamiento actual.
        
        Returns:
            'individual' o 'global'
        """
        return st.session_state.get(f'{self.model_type}_unified_mode', 'individual')
    
    def set_training_mode(self, mode: str):
        """
        Establece el modo de entrenamiento.
        
        Args:
            mode: 'individual' o 'global'
        """
        st.session_state[f'{self.model_type}_unified_mode'] = mode
    
    def show_mode_selector(self) -> str:
        """
        Muestra selector de modo de entrenamiento.
        
        Returns:
            Modo seleccionado
        """
        st.header("🎯 Selecciona Tipo de Modelo")
        
        mode = st.radio(
            "Tipo de entrenamiento:",
            ["🏠 Individual (por sensor)", "🌍 Global (multi-sensor)"],
            index=0 if self.get_training_mode() == 'individual' else 1,
            horizontal=True
        )
        
        new_mode = 'individual' if "Individual" in mode else 'global'
        self.set_training_mode(new_mode)
        
        return new_mode 