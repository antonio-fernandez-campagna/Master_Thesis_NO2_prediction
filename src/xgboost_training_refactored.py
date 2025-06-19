def show_temporal_predictions(test_df: pd.DataFrame, y_pred: np.ndarray, key_prefix: str = "refactored_default"):
    """Muestra gráficos temporales de predicciones vs valores reales."""
    df_plot = test_df[['fecha', 'no2_value']].copy()
    df_plot['Predicción'] = y_pred
    df_plot = df_plot.set_index('fecha')
    
    st.subheader("📈 Predicciones vs Valores Reales")
    
    # Controles para zoom temporal
    col1, col2 = st.columns(2)
    
    with col1:
        date_range = st.date_input(
            "Rango de fechas para visualizar:",
            value=(df_plot.index.min().date(), df_plot.index.max().date()),
            min_value=df_plot.index.min().date(),
            max_value=df_plot.index.max().date(),
            key=f"{key_prefix}_temporal_predictions_date_range"
        )
    
    with col2:
        granularity = st.selectbox(
            "Granularidad:",
            options=['Horaria', 'Media Diaria', 'Media Semanal'],
            index=0,
            key=f"{key_prefix}_temporal_predictions_granularity"
        )

if __name__ == "__main__":
    if st.session_state.xgboost_analysis_tab == 1:
        show_temporal_predictions(analysis_data['test_df'], analysis_data['metrics']['y_pred'], f"refactored_{config_key}")
        st.divider()
        show_residuals_over_time(analysis_data['test_df'], analysis_data['metrics']['y_pred'], f"refactored_{config_key}") 