"""
Implementación específica del entrenador XGBoost utilizando la clase base refactorizada.

Este módulo contiene solo la funcionalidad específica de XGBoost,
reutilizando toda la funcionalidad común de BaseTrainer.
"""

import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from typing import Dict, List, Any, Optional
import warnings

from .base_trainer import BaseTrainer, UnifiedTrainerMixin
from ..config import get_training_config
from ..utils.visualization import (
    show_model_metrics, show_residual_analysis, show_temporal_predictions,
    show_residuals_over_time, show_prediction_scatter, show_hourly_patterns
)

warnings.filterwarnings('ignore')


class XGBoostTrainer(BaseTrainer):
    """
    Entrenador especializado para modelos XGBoost.
    
    Hereda toda la funcionalidad común de BaseTrainer e implementa
    los métodos específicos para XGBoost.
    """
    
    def __init__(self):
        super().__init__('xgboost')
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray, **kwargs) -> xgb.XGBRegressor:
        """
        Entrena el modelo XGBoost con configuración optimizada.
        
        Args:
            X_train: Variables predictoras de entrenamiento
            y_train: Variable objetivo de entrenamiento
            X_val: Variables predictoras de validación
            y_val: Variable objetivo de validación
            **kwargs: Parámetros adicionales de entrenamiento
            
        Returns:
            Modelo XGBoost entrenado
        """
        # Obtener configuración por defecto
        default_config = get_training_config('xgboost')
        
        # Combinar con parámetros personalizados
        model_params = {**default_config, **kwargs}
        
        # Filtrar solo variables numéricas
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = X_train[numeric_features]
        X_val_numeric = X_val[numeric_features]
        
        # Crear modelo con parámetros optimizados
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=model_params.get('n_estimators', 1000),
            learning_rate=model_params.get('learning_rate', 0.05),
            max_depth=model_params.get('max_depth', 7),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse',
            early_stopping_rounds=model_params.get('early_stopping_rounds', 50)
        )
        
        # Configurar progreso visual
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        eval_set = [(X_val_numeric, y_val)]
        
        status_text.text("Entrenando modelo XGBoost...")
        
        # Entrenar modelo
        model.fit(
            X_train_numeric, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        progress_bar.progress(100)
        status_text.success("Entrenamiento XGBoost completado.")
        
        return model
    
    def predict(self, model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo XGBoost.
        
        Args:
            model: Modelo XGBoost entrenado
            X: Variables predictoras
            
        Returns:
            Array con predicciones
        """
        # Filtrar solo variables numéricas
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_features]
        
        return model.predict(X_numeric)
    
    def get_model_specific_metrics(self, model: xgb.XGBRegressor, 
                                 X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Calcula métricas específicas de XGBoost.
        
        Args:
            model: Modelo XGBoost entrenado
            X_test: Variables predictoras de prueba
            y_test: Variable objetivo de prueba
            
        Returns:
            Diccionario con métricas específicas de XGBoost
        """
        metrics = {}
        
        # Información sobre el entrenamiento
        if hasattr(model, 'best_iteration'):
            metrics['best_iteration'] = model.best_iteration
        
        if hasattr(model, 'best_score'):
            metrics['best_score'] = model.best_score
        
        # Información sobre las características
        numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
        metrics['n_features_used'] = len(numeric_features)
        
        # Feature importance si está disponible
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            metrics['feature_importances'] = {
                feature: importance 
                for feature, importance in zip(numeric_features, importances)
            }
            metrics['top_features'] = sorted(
                metrics['feature_importances'].items(), 
                key=lambda x: x[1], reverse=True
            )[:5]
        
        return metrics
    
    def show_feature_importance(self, model: xgb.XGBRegressor, feature_names: List[str]):
        """
        Muestra la importancia de las variables para XGBoost.
        
        Args:
            model: Modelo XGBoost entrenado
            feature_names: Lista de nombres de características
        """
        if not hasattr(model, 'feature_importances_'):
            st.warning("El modelo no tiene información de importancia de variables.")
            return
        
        st.subheader("🎯 Importancia de Variables (XGBoost)")
        
        importances = model.feature_importances_
        
        # Crear DataFrame para mejor manejo
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Gráfico de barras
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Importancia de Variables (XGBoost)')
        ax.set_xlabel('Importancia Relativa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Tabla de importancias
        with st.expander("📋 Tabla de Importancias"):
            st.dataframe(
                importance_df.sort_values('importance', ascending=False), 
                use_container_width=True
            )
    
    def show_model_specific_analysis(self, results: Dict):
        """
        Muestra análisis específico de XGBoost.
        
        Args:
            results: Diccionario con resultados del modelo
        """
        # Mostrar importancia de variables
        self.show_feature_importance(results['model'], results['selected_features'])
        
        # Métricas específicas de XGBoost
        metrics = results['metrics']
        if 'best_iteration' in metrics:
            st.subheader("🚀 Información del Entrenamiento XGBoost")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mejor Iteración", metrics['best_iteration'])
            with col2:
                if 'best_score' in metrics:
                    st.metric("Mejor Score", f"{metrics['best_score']:.4f}")
            with col3:
                if 'n_features_used' in metrics:
                    st.metric("Features Utilizadas", metrics['n_features_used'])
        
        # Top features si están disponibles
        if 'top_features' in metrics:
            st.subheader("🏆 Top 5 Variables Más Importantes")
            for i, (feature, importance) in enumerate(metrics['top_features'], 1):
                st.write(f"{i}. **{feature}**: {importance:.4f}")


class XGBoostUnifiedTrainer(XGBoostTrainer, UnifiedTrainerMixin):
    """
    Entrenador XGBoost unificado que soporta modelos individuales y globales.
    
    Combina la funcionalidad específica de XGBoost con el mixin unificado
    para manejar entrenamientos individuales y globales.
    """
    
    def __init__(self):
        super().__init__()


# ==================== FUNCIONES DE CONVENIENCIA ====================

def create_xgboost_trainer() -> XGBoostTrainer:
    """
    Crea una instancia del entrenador XGBoost simple.
    
    Returns:
        Instancia de XGBoostTrainer
    """
    return XGBoostTrainer()


def create_xgboost_unified_trainer() -> XGBoostUnifiedTrainer:
    """
    Crea una instancia del entrenador XGBoost unificado.
    
    Returns:
        Instancia de XGBoostUnifiedTrainer
    """
    return XGBoostUnifiedTrainer()


def show_xgboost_info_panel():
    """Muestra panel de información sobre XGBoost."""
    with st.expander("ℹ️ Acerca del Entrenamiento XGBoost", expanded=False):
        st.markdown("""
        **🚀 Modelos XGBoost (eXtreme Gradient Boosting)**
        
        XGBoost es un algoritmo de machine learning que utiliza boosting para combinar 
        múltiples árboles de decisión débiles en un modelo robusto.
        
        **✨ Características:**
        - **Excelente rendimiento**: Superior a muchos algoritmos tradicionales
        - **Manejo automático de valores faltantes**: Gestión inteligente de NaNs
        - **Regularización incorporada**: Previene sobreajuste automáticamente
        - **Paralelización eficiente**: Entrenamiento rápido en múltiples cores
        - **Importancia de variables**: Ranking automático de predictores
        
        **🔄 Proceso de entrenamiento:**
        1. **Preprocesamiento**: Variables cíclicas y conversión de unidades
        2. **División temporal**: Separación por fechas (entrenamiento/evaluación)
        3. **Filtrado de outliers**: ⚠️ **Solo en datos de entrenamiento**
        4. **Escalado**: Normalización para mejor convergencia
        5. **Entrenamiento**: Optimización de árboles con boosting
        6. **Early Stopping**: Prevención automática de sobreajuste
        
        **⚖️ Ventajas vs otros algoritmos:**
        - ✅ **Captura interacciones**: Relaciones complejas entre variables
        - ✅ **Robusto a outliers**: Menos sensible a valores atípicos
        - ✅ **Escalabilidad**: Maneja grandes volúmenes de datos
        - ✅ **Flexibilidad**: Adapta automáticamente la complejidad
        
        **🎯 Aplicación específica:**
        Nowcasting de NO₂ basado en variables meteorológicas y de tráfico con 
        capacidad para detectar patrones complejos y no lineales.
        """) 