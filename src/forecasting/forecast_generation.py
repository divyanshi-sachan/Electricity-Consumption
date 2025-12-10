"""
Forecast generation functions.

This module provides functions for generating future forecasts using trained models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from src.config import FORECAST_PERIODS, PROCESSED_DATA_DIR


def generate_future_features(
    df_model: pd.DataFrame,
    state_id: str,
    forecast_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Generate future features for forecasting.
    
    Args:
        df_model: Historical data
        state_id: State ID to generate features for
        forecast_dates: Dates to generate features for
    
    Returns:
        DataFrame with future features
    """
    # Get last known values for this state
    state_data = df_model[df_model['stateid'] == state_id].sort_values('period')
    if len(state_data) == 0:
        raise ValueError(f"No data found for state {state_id}")
    
    last_row = state_data.iloc[-1]
    
    # Generate future features
    future_features = []
    for date in forecast_dates:
        row = {
            'period': date,
            'stateid': state_id,
            'year': date.year,
            'month': date.month,
            'price': last_row.get('price', df_model['price'].median()),
            'customers': last_row.get('customers', df_model['customers'].median()),
            'revenue': last_row.get('revenue', df_model['revenue'].median()),
        }
        
        # Add derived features if they exist
        if 'revenue_per_customer' in df_model.columns:
            row['revenue_per_customer'] = (
                row['revenue'] / row['customers'] if row['customers'] > 0 else 0
            )
        if 'sales_per_customer' in df_model.columns:
            hist_avg = state_data['sales_per_customer'].mean()
            row['sales_per_customer'] = hist_avg if not np.isnan(hist_avg) else 0
        
        future_features.append(row)
    
    return pd.DataFrame(future_features)


def save_forecast_data(
    forecast_df: pd.DataFrame,
    model_name: str,
    filename: Optional[str] = None
) -> Path:
    """
    Save forecast data to file.
    
    Args:
        forecast_df: DataFrame with forecasts
        model_name: Name of the model (e.g., 'xgboost', 'hybrid')
        filename: Optional filename. If None, generates timestamped filename.
    
    Returns:
        Path to saved file
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_forecasts_{timestamp}.csv"
    
    file_path = PROCESSED_DATA_DIR / filename
    forecast_df.to_csv(file_path, index=False)
    print(f"✅ Saved forecast data to: {file_path}")
    
    return file_path

