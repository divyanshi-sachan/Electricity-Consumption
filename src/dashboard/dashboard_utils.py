"""
Dashboard utility functions for Streamlit.

This module provides functions for preparing data and creating visualizations for the dashboard.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from src.config import PROCESSED_DATA_DIR


def load_forecast_data(model_name: str = 'hybrid') -> pd.DataFrame:
    """
    Load forecast data from processed directory.
    
    Args:
        model_name: Name of the model (default: 'hybrid')
    
    Returns:
        DataFrame with forecast data
    """
    # Find most recent forecast file for this model
    pattern = f"{model_name}_forecasts_*.csv"
    forecast_files = list(PROCESSED_DATA_DIR.glob(pattern))
    
    if not forecast_files:
        raise FileNotFoundError(f"No forecast files found for {model_name}")
    
    latest_file = max(forecast_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    if 'period' in df.columns:
        df['period'] = pd.to_datetime(df['period'])
    
    return df


def get_forecast_for_state(
    forecast_df: pd.DataFrame,
    state_id: str
) -> Dict:
    """
    Get forecast data for a specific state.
    
    Args:
        forecast_df: DataFrame with forecasts
        state_id: State ID
    
    Returns:
        Dictionary with forecast data
    """
    state_fore = forecast_df[forecast_df['stateid'] == state_id].sort_values('period')
    
    return {
        'state_id': state_id,
        'forecasted': state_fore.to_dict('records') if len(state_fore) > 0 else []
    }


def prepare_dashboard_data(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine historical and forecast data for dashboard.
    
    Args:
        historical_df: Historical data
        forecast_df: Forecast data
    
    Returns:
        Combined DataFrame
    """
    # Prepare historical data
    hist_data = historical_df.copy()
    if 'period' in hist_data.columns:
        hist_data['period'] = pd.to_datetime(hist_data['period'])
    hist_data['type'] = 'Historical'
    if 'value' not in hist_data.columns and 'sales' in hist_data.columns:
        hist_data['value'] = hist_data['sales']
    
    # Prepare forecast data
    fore_data = forecast_df.copy()
    if 'period' in fore_data.columns:
        fore_data['period'] = pd.to_datetime(fore_data['period'])
    
    # Combine
    combined = pd.concat([hist_data, fore_data], ignore_index=True)
    combined = combined.sort_values(['stateid', 'period'])
    
    return combined

