"""
Regional insights and trend analysis functions.

This module provides functions to analyze forecast trends by state/region,
identify high-growth, falling demand, and volatile states.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path


def calculate_state_growth_metrics(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    forecast_start_year: int = 2025,
    forecast_end_year: int = 2035
) -> pd.DataFrame:
    """
    Calculate growth metrics for each state.
    
    Args:
        forecast_df: DataFrame with forecasts (columns: period, stateid, value)
        historical_df: DataFrame with historical data (columns: period, stateid, sales/value)
        forecast_start_year: Start year for forecast period
        forecast_end_year: End year for forecast period
    
    Returns:
        DataFrame with growth metrics per state
    """
    # Prepare data
    forecast_df = forecast_df.copy()
    historical_df = historical_df.copy()
    
    # Validate required columns
    if 'stateid' not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'stateid' column")
    if 'stateid' not in historical_df.columns:
        raise ValueError("historical_df must contain 'stateid' column")
    if 'period' not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'period' column")
    if 'period' not in historical_df.columns:
        raise ValueError("historical_df must contain 'period' column")
    
    if 'period' in forecast_df.columns:
        forecast_df['period'] = pd.to_datetime(forecast_df['period'])
    if 'period' in historical_df.columns:
        historical_df['period'] = pd.to_datetime(historical_df['period'])
    
    # Get value column name
    value_col = 'value' if 'value' in forecast_df.columns else 'sales'
    hist_value_col = 'value' if 'value' in historical_df.columns else 'sales'
    
    if value_col not in forecast_df.columns:
        raise ValueError(f"forecast_df must contain either 'value' or 'sales' column. Found columns: {list(forecast_df.columns)}")
    if hist_value_col not in historical_df.columns:
        raise ValueError(f"historical_df must contain either 'value' or 'sales' column. Found columns: {list(historical_df.columns)}")
    
    # Check if we have any data
    if len(forecast_df) == 0:
        raise ValueError("forecast_df is empty")
    if len(historical_df) == 0:
        raise ValueError("historical_df is empty")
    
    # Calculate metrics per state
    state_metrics = []
    
    unique_states = forecast_df['stateid'].unique()
    unique_hist_states = historical_df['stateid'].unique()
    
    if len(unique_states) == 0:
        raise ValueError("No unique state IDs found in forecast_df")
    
    # Debug: Check data overlap
    common_states = set(unique_states) & set(unique_hist_states)
    if len(common_states) == 0:
        import warnings
        warnings.warn(
            f"No common state IDs between forecast and historical data. "
            f"Forecast states: {len(unique_states)}, Historical states: {len(unique_hist_states)}. "
            f"Sample forecast states: {list(unique_states[:5])}, "
            f"Sample historical states: {list(unique_hist_states[:5])}"
        )
    
    # Check for zero values
    non_zero_forecasts = forecast_df[forecast_df[value_col] > 0]
    if len(non_zero_forecasts) == 0:
        import warnings
        warnings.warn(
            f"All forecast values are zero or negative. "
            f"Total forecast rows: {len(forecast_df)}, "
            f"Non-zero rows: {len(non_zero_forecasts)}"
        )
    
    # Filter forecast to only actual forecasts if 'type' column exists
    # (exclude component rows like 'sarima_component', 'xgb_residual_component')
    forecast_df_filtered = forecast_df.copy()
    if 'type' in forecast_df_filtered.columns:
        # Check what types exist
        valid_types = forecast_df_filtered['type'].unique()
        type_strs = [str(v).lower() for v in valid_types]
        
        # Filter to keep only the main forecast type (not components)
        # Common patterns: 'forecast', 'hybrid', 'final', etc.
        # Exclude component types like 'sarima_component', 'xgb_residual_component'
        # Check if any type contains 'forecast', 'hybrid', 'final', or 'prediction' (case-insensitive)
        forecast_keywords = ['forecast', 'hybrid', 'final', 'prediction']
        matching_types = [t for t in valid_types if any(kw in str(t).lower() for kw in forecast_keywords)]
        
        if len(matching_types) > 0:
            # Use types that match forecast keywords
            forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['type'].isin(matching_types)]
        elif any('component' not in str(t).lower() for t in valid_types):
            # If there are non-component types, use those
            non_component_types = [t for t in valid_types if 'component' not in str(t).lower()]
            if len(non_component_types) > 0:
                forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['type'].isin(non_component_types)]
        
        # If no clear forecast type, use all rows but log a warning
        if len(forecast_df_filtered) == 0:
            import warnings
            warnings.warn(f"After filtering by 'type', no forecast rows remain. Using all rows. Types found: {valid_types}")
            forecast_df_filtered = forecast_df.copy()
    
    # Update unique_states to use filtered dataframe
    unique_states_filtered = forecast_df_filtered['stateid'].unique()
    if len(unique_states_filtered) == 0:
        import warnings
        warnings.warn("No states found in filtered forecast data")
        unique_states_filtered = unique_states
    
    skipped_states = {
        'no_forecast': 0,
        'no_historical': 0,
        'all_zero': 0,
        'no_first_year': 0,
        'no_last_year': 0,
        'processed': 0
    }
    
    for state_id in unique_states_filtered:
        state_forecast = forecast_df_filtered[forecast_df_filtered['stateid'] == state_id].sort_values('period')
        state_historical = historical_df[historical_df['stateid'] == state_id].sort_values('period')
        
        if len(state_forecast) == 0:
            skipped_states['no_forecast'] += 1
            continue
        if len(state_historical) == 0:
            skipped_states['no_historical'] += 1
            continue
        
        # Filter out zero forecasts (likely data quality issues)
        state_forecast_valid = state_forecast[state_forecast[value_col] > 0].copy()
        
        if len(state_forecast_valid) == 0:
            skipped_states['all_zero'] += 1
            continue
        
        # Get first and last year of VALID (non-zero) forecast data
        # This is more lenient than using min/max of all data
        forecast_start = state_forecast_valid['period'].min()
        forecast_end = state_forecast_valid['period'].max()
        
        # Calculate average for first year and last year of forecast (using valid data only)
        first_year_data = state_forecast_valid[
            state_forecast_valid['period'].dt.year == forecast_start.year
        ][value_col]
        last_year_data = state_forecast_valid[
            state_forecast_valid['period'].dt.year == forecast_end.year
        ][value_col]
        
        if len(first_year_data) == 0:
            skipped_states['no_first_year'] += 1
            continue
        if len(last_year_data) == 0:
            skipped_states['no_last_year'] += 1
            continue
        
        first_year_avg = first_year_data.mean()
        last_year_avg = last_year_data.mean()
        
        # Calculate growth rate (handle edge cases)
        if first_year_avg > 0:
            total_growth_pct = ((last_year_avg - first_year_avg) / first_year_avg * 100)
        else:
            # If first year is zero but last year has data, growth is undefined
            total_growth_pct = 0
        
        # Calculate volatility (coefficient of variation) using valid forecasts only
        forecast_std = state_forecast_valid[value_col].std()
        forecast_mean = state_forecast_valid[value_col].mean()
        volatility = (forecast_std / forecast_mean * 100) if forecast_mean > 0 else 0
        
        # Get historical baseline (filter zeros if they're data quality issues)
        historical_valid = state_historical[state_historical[hist_value_col] > 0]
        if len(historical_valid) > 0:
            historical_mean = historical_valid[hist_value_col].mean()
            historical_max = historical_valid[hist_value_col].max()
        else:
            historical_mean = state_historical[hist_value_col].mean()
            historical_max = state_historical[hist_value_col].max()
        
        # Calculate forecast peak (using valid forecasts)
        forecast_peak = state_forecast_valid[value_col].max() if len(state_forecast_valid) > 0 else 0
        
        # Risk level
        if total_growth_pct > 15:
            risk_level = "High Growth"
        elif total_growth_pct < -5:
            risk_level = "Declining"
        elif volatility > 20:
            risk_level = "Volatile"
        else:
            risk_level = "Stable"
        
        state_metrics.append({
            'stateid': state_id,
            'historical_mean': historical_mean,
            'historical_max': historical_max,
            'forecast_start_avg': first_year_avg,
            'forecast_end_avg': last_year_avg,
            'total_growth_pct': total_growth_pct,
            'volatility_pct': volatility,
            'forecast_peak': forecast_peak,
            'risk_level': risk_level,
            'forecast_periods': len(state_forecast)
        })
        skipped_states['processed'] += 1
    
    # Log skipped states for debugging
    if len(state_metrics) == 0:
        import warnings
        total_skipped = sum(skipped_states.values())
        warnings.warn(
            f"No states processed. States checked: {len(unique_states_filtered)}, "
            f"Skipped breakdown: {skipped_states}. "
            f"Total skipped: {total_skipped}"
        )
    
    # Ensure DataFrame has expected columns even if empty
    expected_columns = [
        'stateid', 'historical_mean', 'historical_max', 'forecast_start_avg',
        'forecast_end_avg', 'total_growth_pct', 'volatility_pct', 
        'forecast_peak', 'risk_level', 'forecast_periods'
    ]
    
    if len(state_metrics) == 0:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=expected_columns)
    
    return pd.DataFrame(state_metrics)


def identify_high_growth_states(
    state_metrics_df: pd.DataFrame,
    growth_threshold: float = 15.0
) -> pd.DataFrame:
    """
    Identify states with high growth.
    
    Args:
        state_metrics_df: DataFrame from calculate_state_growth_metrics
        growth_threshold: Minimum growth percentage to be considered high growth
    
    Returns:
        DataFrame with high growth states
    """
    # Handle empty DataFrame or missing required column
    if state_metrics_df is None or len(state_metrics_df) == 0:
        expected_cols = ['stateid', 'total_growth_pct', 'volatility_pct', 'risk_level']
        return pd.DataFrame(columns=expected_cols)
    
    # Check if required column exists
    if 'total_growth_pct' not in state_metrics_df.columns:
        # Return empty DataFrame with same structure
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    # Filter and sort
    filtered = state_metrics_df[
        state_metrics_df['total_growth_pct'] >= growth_threshold
    ]
    
    if len(filtered) == 0:
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    return filtered.sort_values('total_growth_pct', ascending=False)


def identify_declining_states(
    state_metrics_df: pd.DataFrame,
    decline_threshold: float = -5.0
) -> pd.DataFrame:
    """
    Identify states with falling demand.
    
    Args:
        state_metrics_df: DataFrame from calculate_state_growth_metrics
        decline_threshold: Maximum growth percentage to be considered declining
    
    Returns:
        DataFrame with declining states
    """
    # Handle empty DataFrame or missing required column
    if state_metrics_df is None or len(state_metrics_df) == 0:
        expected_cols = ['stateid', 'total_growth_pct', 'volatility_pct', 'risk_level']
        return pd.DataFrame(columns=expected_cols)
    
    # Check if required column exists
    if 'total_growth_pct' not in state_metrics_df.columns:
        # Return empty DataFrame with same structure
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    # Filter and sort
    filtered = state_metrics_df[
        state_metrics_df['total_growth_pct'] <= decline_threshold
    ]
    
    if len(filtered) == 0:
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    return filtered.sort_values('total_growth_pct', ascending=True)


def identify_volatile_states(
    state_metrics_df: pd.DataFrame,
    volatility_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Identify states with unstable/volatile demand.
    
    Args:
        state_metrics_df: DataFrame from calculate_state_growth_metrics
        volatility_threshold: Minimum volatility percentage to be considered volatile
    
    Returns:
        DataFrame with volatile states
    """
    # Handle empty DataFrame or missing required column
    if state_metrics_df is None or len(state_metrics_df) == 0:
        expected_cols = ['stateid', 'total_growth_pct', 'volatility_pct', 'risk_level']
        return pd.DataFrame(columns=expected_cols)
    
    # Check if required column exists
    if 'volatility_pct' not in state_metrics_df.columns:
        # Return empty DataFrame with same structure
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    # Filter and sort
    filtered = state_metrics_df[
        state_metrics_df['volatility_pct'] >= volatility_threshold
    ]
    
    if len(filtered) == 0:
        return pd.DataFrame(columns=state_metrics_df.columns)
    
    return filtered.sort_values('volatility_pct', ascending=False)


def plot_state_forecast_trends(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    state_ids: List[str],
    value_col: str = 'value',
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot forecast trend lines for multiple states.
    
    Args:
        forecast_df: DataFrame with forecasts
        historical_df: DataFrame with historical data
        state_ids: List of state IDs to plot
        value_col: Name of value column
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_states = len(state_ids)
    n_cols = 3
    n_rows = (n_states + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_states > 1 else [axes]
    
    for idx, state_id in enumerate(state_ids):
        ax = axes[idx]
        
        # Get state data
        state_forecast = forecast_df[forecast_df['stateid'] == state_id].sort_values('period')
        state_historical = historical_df[historical_df['stateid'] == state_id].sort_values('period')
        
        if len(state_forecast) > 0:
            # Plot historical
            if len(state_historical) > 0:
                ax.plot(state_historical['period'], state_historical[value_col], 
                       label='Historical', color='steelblue', linewidth=2, marker='o', markersize=3)
            
            # Plot forecast
            ax.plot(state_forecast['period'], state_forecast[value_col], 
                   label='Forecast', color='coral', linewidth=2, linestyle='--', marker='s', markersize=3)
            
            ax.set_title(f'State: {state_id}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Period', fontsize=10)
            ax.set_ylabel('Demand (MWh)', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_states, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def diagnose_data_issues(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame
) -> Dict:
    """
    Diagnose potential data issues that might cause empty results.
    
    Returns a dictionary with diagnostic information.
    """
    diagnostics = {
        'forecast_rows': len(forecast_df),
        'historical_rows': len(historical_df),
        'forecast_columns': list(forecast_df.columns),
        'historical_columns': list(historical_df.columns),
    }
    
    # Check for value columns
    forecast_value_col = 'value' if 'value' in forecast_df.columns else ('sales' if 'sales' in forecast_df.columns else None)
    hist_value_col = 'value' if 'value' in historical_df.columns else ('sales' if 'sales' in historical_df.columns else None)
    
    if forecast_value_col:
        diagnostics['forecast_non_zero'] = len(forecast_df[forecast_df[forecast_value_col] > 0])
        diagnostics['forecast_total_sum'] = forecast_df[forecast_value_col].sum()
        diagnostics['forecast_mean'] = forecast_df[forecast_value_col].mean()
        diagnostics['forecast_min'] = forecast_df[forecast_value_col].min()
        diagnostics['forecast_max'] = forecast_df[forecast_value_col].max()
    
    if hist_value_col:
        diagnostics['historical_non_zero'] = len(historical_df[historical_df[hist_value_col] > 0])
        diagnostics['historical_total_sum'] = historical_df[hist_value_col].sum()
    
    # Check state IDs
    if 'stateid' in forecast_df.columns and 'stateid' in historical_df.columns:
        forecast_states = set(forecast_df['stateid'].unique())
        hist_states = set(historical_df['stateid'].unique())
        common_states = forecast_states & hist_states
        
        diagnostics['forecast_unique_states'] = len(forecast_states)
        diagnostics['historical_unique_states'] = len(hist_states)
        diagnostics['common_states'] = len(common_states)
        diagnostics['forecast_only_states'] = len(forecast_states - hist_states)
        diagnostics['historical_only_states'] = len(hist_states - forecast_states)
        
        if len(common_states) < 5:
            diagnostics['sample_forecast_states'] = list(forecast_states)[:10]
            diagnostics['sample_historical_states'] = list(hist_states)[:10]
            diagnostics['common_states_list'] = list(common_states)[:10]
    
    # Check periods
    if 'period' in forecast_df.columns and 'period' in historical_df.columns:
        try:
            forecast_periods = pd.to_datetime(forecast_df['period'])
            hist_periods = pd.to_datetime(historical_df['period'])
            diagnostics['forecast_period_range'] = (forecast_periods.min(), forecast_periods.max())
            diagnostics['historical_period_range'] = (hist_periods.min(), hist_periods.max())
        except:
            pass
    
    return diagnostics


def generate_regional_insights_report(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    verbose: bool = False
) -> Dict:
    """
    Generate comprehensive regional insights report.
    
    Args:
        forecast_df: DataFrame with forecasts
        historical_df: DataFrame with historical data
        verbose: If True, include diagnostic information in the output
    
    Returns:
        Dictionary with insights and dataframes
    """
    # Run diagnostics if verbose or if we expect issues
    diagnostics = None
    if verbose:
        diagnostics = diagnose_data_issues(forecast_df, historical_df)
    
    # Calculate state metrics
    state_metrics = calculate_state_growth_metrics(forecast_df, historical_df)
    
    # If no states found, run diagnostics automatically
    if len(state_metrics) == 0 and not verbose:
        diagnostics = diagnose_data_issues(forecast_df, historical_df)
    
    # Ensure state_metrics has the expected structure
    if state_metrics is None:
        state_metrics = pd.DataFrame(columns=[
            'stateid', 'historical_mean', 'historical_max', 'forecast_start_avg',
            'forecast_end_avg', 'total_growth_pct', 'volatility_pct', 
            'forecast_peak', 'risk_level', 'forecast_periods'
        ])
    elif len(state_metrics) > 0 and 'total_growth_pct' not in state_metrics.columns:
        # If we have data but missing column, something went wrong
        # Return empty DataFrame with correct structure
        state_metrics = pd.DataFrame(columns=[
            'stateid', 'historical_mean', 'historical_max', 'forecast_start_avg',
            'forecast_end_avg', 'total_growth_pct', 'volatility_pct', 
            'forecast_peak', 'risk_level', 'forecast_periods'
        ])
    
    # Identify different state categories
    high_growth = identify_high_growth_states(state_metrics)
    declining = identify_declining_states(state_metrics)
    volatile = identify_volatile_states(state_metrics)
    
    # Summary statistics
    if len(state_metrics) == 0:
        summary = {
            'total_states': 0,
            'high_growth_count': 0,
            'declining_count': 0,
            'volatile_count': 0,
            'stable_count': 0,
            'avg_growth_pct': 0.0,
            'max_growth_pct': 0.0,
            'min_growth_pct': 0.0
        }
    else:
        summary = {
            'total_states': len(state_metrics),
            'high_growth_count': len(high_growth),
            'declining_count': len(declining),
            'volatile_count': len(volatile),
            'stable_count': len(state_metrics) - len(high_growth) - len(declining) - len(volatile),
            'avg_growth_pct': state_metrics['total_growth_pct'].mean() if 'total_growth_pct' in state_metrics.columns else 0.0,
            'max_growth_pct': state_metrics['total_growth_pct'].max() if 'total_growth_pct' in state_metrics.columns else 0.0,
            'min_growth_pct': state_metrics['total_growth_pct'].min() if 'total_growth_pct' in state_metrics.columns else 0.0
        }
    
    result = {
        'state_metrics': state_metrics,
        'high_growth_states': high_growth,
        'declining_states': declining,
        'volatile_states': volatile,
        'summary': summary
    }
    
    if diagnostics is not None:
        result['diagnostics'] = diagnostics
    
    return result

