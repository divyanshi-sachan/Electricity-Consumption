"""
Advanced visualization functions for dashboard.

This module provides interactive visualizations including heatmaps, maps, and dashboards.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available. Some visualizations will use matplotlib instead.")

from typing import Dict, List, Optional, Tuple


def create_demand_heatmap(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    year: int = 2030,
    value_col: Optional[str] = None,
    interactive: bool = True
):
    """
    Create a heatmap showing current demand vs forecasted demand.
    
    Args:
        forecast_df: DataFrame with forecasts
        historical_df: DataFrame with historical data
        year: Year to visualize
        value_col: Name of value column (auto-detected if None)
        interactive: Whether to use Plotly (interactive) or matplotlib
    
    Returns:
        Figure object
    """
    # Prepare data
    forecast_df_original = forecast_df.copy()
    forecast_df = forecast_df.copy()
    historical_df = historical_df.copy()
    
    # Debug: Check input data
    print(f"\n🔍 DEBUG: Input data check")
    print(f"   Forecast shape: {forecast_df.shape}")
    print(f"   Forecast columns: {list(forecast_df.columns)}")
    print(f"   Requested value_col: {value_col}")
    
    # Auto-detect value column if not specified
    if value_col is None:
        if 'sales' in forecast_df.columns:
            value_col = 'sales'
        elif 'value' in forecast_df.columns:
            value_col = 'value'
        else:
            raise ValueError("No 'sales' or 'value' column found in forecast_df")
    
    if value_col not in forecast_df.columns:
        print(f"   ⚠️ WARNING: Column '{value_col}' not found! Available: {list(forecast_df.columns)}")
        # Try to use 'value' as fallback
        if 'value' in forecast_df.columns:
            print(f"   → Falling back to 'value' column")
            value_col = 'value'
        else:
            raise ValueError(f"Column '{value_col}' not found in forecast_df. Available: {list(forecast_df.columns)}")
    
    print(f"   Using column: '{value_col}'")
    if value_col in forecast_df.columns:
        non_zero_input = (forecast_df[value_col] > 0).sum()
        total_input = forecast_df[value_col].sum()
        print(f"   Input data - Non-zero: {non_zero_input}/{len(forecast_df)}, Total: {total_input:,.0f}")
    
    # Convert period to datetime first
    if 'period' in forecast_df.columns:
        forecast_df['period'] = pd.to_datetime(forecast_df['period'])
    if 'period' in historical_df.columns:
        historical_df['period'] = pd.to_datetime(historical_df['period'])
    
    # Filter forecast to only actual forecast rows (exclude component rows)
    print(f"\n🔍 DEBUG: After type filtering")
    if 'type' in forecast_df.columns:
        valid_types = forecast_df['type'].unique()
        print(f"   Available types: {list(valid_types)}")
        forecast_keywords = ['forecast', 'hybrid', 'final', 'prediction']
        matching_types = [t for t in valid_types if any(kw in str(t).lower() for kw in forecast_keywords)]
        
        if len(matching_types) > 0:
            print(f"   Matching types: {list(matching_types)}")
            forecast_df = forecast_df[forecast_df['type'].isin(matching_types)]
            print(f"   Rows after type filter: {len(forecast_df)}")
        elif any('component' not in str(t).lower() for t in valid_types):
            # Use non-component types
            non_component_types = [t for t in valid_types if 'component' not in str(t).lower()]
            if len(non_component_types) > 0:
                print(f"   Using non-component types: {list(non_component_types)}")
                forecast_df = forecast_df[forecast_df['type'].isin(non_component_types)]
                print(f"   Rows after type filter: {len(forecast_df)}")
    else:
        print(f"   No 'type' column found, skipping type filtering")
    
    # Check available years in forecast
    if len(forecast_df) == 0:
        print(f"   ⚠️ WARNING: No forecast data after filtering by type. Using all original data.")
        forecast_df = forecast_df_original.copy()
        if 'period' in forecast_df.columns:
            forecast_df['period'] = pd.to_datetime(forecast_df['period'])
        print(f"   Rows after restoring original: {len(forecast_df)}")
    
    # Get available years
    available_years = sorted(forecast_df['period'].dt.year.unique()) if 'period' in forecast_df.columns and len(forecast_df) > 0 else []
    if len(available_years) == 0:
        raise ValueError("No forecast data available")
    
    print(f"\n🔍 DEBUG: Available years")
    print(f"   Available years: {available_years}")
    print(f"   Requested year: {year}")
    
    # Check which years have non-zero data
    years_with_data = []
    for y in available_years:
        year_data = forecast_df[forecast_df['period'].dt.year == y]
        if 'value' in year_data.columns and (year_data['value'] > 0).sum() > 0:
            years_with_data.append(y)
    
    print(f"   Years with non-zero data: {years_with_data}")
    
    if year not in available_years:
        closest_year = min(available_years, key=lambda x: abs(x - year))
        print(f"   ⚠️ Year {year} not found! Using closest year: {closest_year}")
        year = closest_year
    elif year not in years_with_data:
        # Year exists but has no data - use closest year with data
        if len(years_with_data) > 0:
            closest_with_data = min(years_with_data, key=lambda x: abs(x - year))
            print(f"   ⚠️ Year {year} has no non-zero data! Using closest year with data: {closest_with_data}")
            year = closest_with_data
        else:
            print(f"   ⚠️ WARNING: No years have non-zero data!")
    else:
        print(f"   ✓ Year {year} found in data with non-zero values")
    
    # Filter by year first, then filter out zero/negative values
    forecast_year_data = forecast_df[forecast_df['period'].dt.year == year].copy()
    
    print(f"\n🔍 DEBUG: After filtering for year {year}")
    print(f"   Rows for year {year}: {len(forecast_year_data)}")
    
    # Check both 'sales' and 'value' columns to see which has data
    if 'sales' in forecast_year_data.columns:
        sales_non_zero = (forecast_year_data['sales'] > 0).sum()
        sales_total = forecast_year_data['sales'].sum()
        print(f"   'sales' column - Non-zero: {sales_non_zero}/{len(forecast_year_data)}, Total: {sales_total:,.0f}")
    
    if 'value' in forecast_year_data.columns:
        value_non_zero = (forecast_year_data['value'] > 0).sum()
        value_total = forecast_year_data['value'].sum()
        print(f"   'value' column - Non-zero: {value_non_zero}/{len(forecast_year_data)}, Total: {value_total:,.0f}")
    
    # Use the column that has data - always prefer 'value' if 'sales' is zeros
    if value_col == 'sales' and 'sales' in forecast_year_data.columns and 'value' in forecast_year_data.columns:
        sales_has_data = (forecast_year_data['sales'] > 0).sum() > 0
        value_has_data = (forecast_year_data['value'] > 0).sum() > 0
        
        if not sales_has_data and value_has_data:
            print(f"   ⚠️ 'sales' column is all zeros, switching to 'value' column")
            value_col = 'value'
            non_zero_count = value_non_zero
        elif sales_has_data:
            print(f"   ✓ Using 'sales' column (has {sales_non_zero} non-zero values)")
            non_zero_count = sales_non_zero
        else:
            print(f"   ⚠️ Both 'sales' and 'value' are zeros for year {year}!")
            print(f"   → This suggests year {year} may not have forecast data. Check available years above.")
            # Still try to use 'value' as it's the source column
            value_col = 'value'
            non_zero_count = 0
    else:
        # If 'sales' not available, use 'value'
        if value_col == 'sales' and 'value' in forecast_year_data.columns:
            print(f"   ⚠️ 'sales' column not found, using 'value' column")
            value_col = 'value'
        non_zero_count = (forecast_year_data[value_col] > 0).sum() if value_col in forecast_year_data.columns else 0
    
    # Check if value_col exists
    if value_col not in forecast_year_data.columns:
        raise ValueError(f"Column '{value_col}' not found in forecast data. Available columns: {list(forecast_year_data.columns)}")
    
    # Filter out zero/negative values if we have non-zero data
    if non_zero_count > 0:
        forecast_year_data = forecast_year_data[forecast_year_data[value_col] > 0].copy()
    else:
        import warnings
        warnings.warn(f"No non-zero forecast data for year {year} in column '{value_col}'. Using all values (including zeros).")
    
    # Get current (latest historical) and forecasted demand
    latest_historical = historical_df[historical_df['period'] == historical_df['period'].max()]
    
    # Aggregate by state
    hist_value_col = 'sales' if 'sales' in historical_df.columns else ('value' if 'value' in historical_df.columns else value_col)
    current_demand = latest_historical.groupby('stateid')[hist_value_col].sum().reset_index()
    current_demand.columns = ['stateid', 'current_demand']
    
    # Aggregate by state - use sum for annual total (monthly values summed)
    if value_col not in forecast_year_data.columns:
        raise ValueError(f"Column '{value_col}' not found in forecast_year_data. Available: {list(forecast_year_data.columns)}")
    
    # Debug: Check data before aggregation (use print so it's always visible)
    print(f"\n🔍 DEBUG: Forecast data for year {year}")
    print(f"   Rows after filtering: {len(forecast_year_data)}")
    print(f"   Columns available: {list(forecast_year_data.columns)}")
    print(f"   Using column: '{value_col}'")
    
    if len(forecast_year_data) == 0:
        print(f"   ⚠️ ERROR: No forecast data for year {year} after filtering!")
    else:
        sample_state = forecast_year_data['stateid'].iloc[0] if len(forecast_year_data) > 0 else None
        sample_value = forecast_year_data[value_col].iloc[0] if len(forecast_year_data) > 0 else None
        non_zero = (forecast_year_data[value_col] > 0).sum()
        total_val = forecast_year_data[value_col].sum()
        print(f"   Sample state: {sample_state}, sample value: {sample_value}")
        print(f"   Non-zero values: {non_zero} / {len(forecast_year_data)}")
        print(f"   Total sum: {total_val:,.0f}")
    
    forecast_demand = forecast_year_data.groupby('stateid')[value_col].sum().reset_index()
    forecast_demand.columns = ['stateid', 'forecast_demand']
    
    # Debug: Check if we have data after aggregation
    print(f"\n🔍 DEBUG: After aggregation by state")
    if len(forecast_demand) == 0:
        print(f"   ⚠️ ERROR: No forecast data after aggregation!")
        print(f"   Forecast year data shape: {forecast_year_data.shape}")
    else:
        total_forecast = forecast_demand['forecast_demand'].sum()
        avg_forecast = forecast_demand['forecast_demand'].mean()
        max_forecast = forecast_demand['forecast_demand'].max()
        min_forecast = forecast_demand['forecast_demand'].min()
        zero_count = (forecast_demand['forecast_demand'] == 0).sum()
        print(f"   States: {len(forecast_demand)}")
        print(f"   Total forecast: {total_forecast:,.0f}")
        print(f"   Average: {avg_forecast:,.0f}")
        print(f"   Max: {max_forecast:,.0f}")
        print(f"   Min: {min_forecast:,.0f}")
        print(f"   Zero values: {zero_count}")
        if zero_count > 0:
            print(f"   ⚠️ WARNING: {zero_count} states have zero forecast demand!")
    
    # Merge and calculate gap
    heatmap_data = current_demand.merge(forecast_demand, on='stateid', how='outer')
    
    # Fill NaN values with 0 for missing data
    heatmap_data['forecast_demand'] = heatmap_data['forecast_demand'].fillna(0)
    heatmap_data['current_demand'] = heatmap_data['current_demand'].fillna(0)
    
    # Remove rows where both are zero (no data)
    heatmap_data = heatmap_data[(heatmap_data['current_demand'] > 0) | (heatmap_data['forecast_demand'] > 0)].copy()
    heatmap_data['demand_gap'] = heatmap_data['forecast_demand'] - heatmap_data['current_demand']
    heatmap_data['growth_pct'] = (
        (heatmap_data['demand_gap'] / heatmap_data['current_demand']) * 100
    ).fillna(0)
    
    # Risk level
    heatmap_data['risk_level'] = heatmap_data['growth_pct'].apply(
        lambda x: 'Critical' if x > 20 else ('High' if x > 10 else ('Medium' if x > 5 else ('Low' if x > -5 else 'Declining')))
    )
    
    # Create size column (use absolute value for visualization, but keep original for hover)
    heatmap_data['size_value'] = heatmap_data['growth_pct'].abs()
    # Ensure minimum size for visibility
    heatmap_data['size_value'] = heatmap_data['size_value'].clip(lower=0.1)
    
    if interactive and HAS_PLOTLY:
        # Interactive Plotly heatmap
        fig = px.scatter(
            heatmap_data,
            x='current_demand',
            y='forecast_demand',
            size='size_value',
            color='risk_level',
            hover_data=['stateid', 'demand_gap', 'growth_pct'],
            title=f'Demand Forecast Heatmap - {year}',
            labels={
                'current_demand': 'Current Demand (MWh)',
                'forecast_demand': 'Forecasted Demand (MWh)',
                'growth_pct': 'Growth %',
                'risk_level': 'Risk Level',
                'size_value': '|Growth| %'
            },
            color_discrete_map={
                'Critical': 'red',
                'High': 'orange',
                'Medium': 'yellow',
                'Low': 'lightgreen',
                'Declining': 'lightblue'
            }
        )
        return fig
    else:
        # Static matplotlib heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot_table(
            values='growth_pct',
            index='stateid',
            columns='risk_level',
            aggfunc='mean',
            fill_value=0
        )
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Growth %'},
            ax=ax
        )
        
        ax.set_title(f'Demand Forecast Heatmap - {year}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


def create_state_comparison_chart(
    forecast_df: pd.DataFrame,
    state_ids: List[str],
    value_col: Optional[str] = None,
    interactive: bool = True
):
    """
    Create comparison chart for multiple states.
    
    Args:
        forecast_df: DataFrame with forecasts
        state_ids: List of state IDs to compare
        value_col: Name of value column (auto-detected if None)
        interactive: Whether to use Plotly
    
    Returns:
        Figure object
    """
    forecast_df = forecast_df.copy()
    
    # Auto-detect value column if not specified
    if value_col is None:
        if 'sales' in forecast_df.columns:
            value_col = 'sales'
        elif 'value' in forecast_df.columns:
            value_col = 'value'
        else:
            raise ValueError("No 'sales' or 'value' column found in forecast_df")
    
    if value_col not in forecast_df.columns:
        raise ValueError(f"Column '{value_col}' not found in forecast_df")
    
    # Filter forecast to only actual forecast rows (exclude component rows)
    if 'type' in forecast_df.columns:
        valid_types = forecast_df['type'].unique()
        forecast_keywords = ['forecast', 'hybrid', 'final', 'prediction']
        matching_types = [t for t in valid_types if any(kw in str(t).lower() for kw in forecast_keywords)]
        
        if len(matching_types) > 0:
            forecast_df = forecast_df[forecast_df['type'].isin(matching_types)]
        elif any('component' not in str(t).lower() for t in valid_types):
            # Use non-component types
            non_component_types = [t for t in valid_types if 'component' not in str(t).lower()]
            if len(non_component_types) > 0:
                forecast_df = forecast_df[forecast_df['type'].isin(non_component_types)]
    
    # Filter out zero/negative values
    forecast_df = forecast_df[forecast_df[value_col] > 0].copy()
    
    if 'period' in forecast_df.columns:
        forecast_df['period'] = pd.to_datetime(forecast_df['period'])
    
    # Filter to selected states
    state_data = forecast_df[forecast_df['stateid'].isin(state_ids)].sort_values(['stateid', 'period'])
    
    if interactive and HAS_PLOTLY:
        fig = px.line(
            state_data,
            x='period',
            y=value_col,
            color='stateid',
            title='State Comparison - Forecast Trends',
            labels={'period': 'Period', value_col: 'Demand (MWh)', 'stateid': 'State'}
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        for state_id in state_ids:
            state_forecast = state_data[state_data['stateid'] == state_id]
            ax.plot(state_forecast['period'], state_forecast[value_col], 
                   label=state_id, marker='o', markersize=3)
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Demand (MWh)', fontsize=12)
        ax.set_title('State Comparison - Forecast Trends', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def create_risk_dashboard(
    risk_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame
):
    """
    Create comprehensive risk dashboard.
    
    Args:
        risk_df: DataFrame with risk assessment
        forecast_df: DataFrame with forecasts
        historical_df: DataFrame with historical data
    
    Returns:
        Plotly figure with subplots
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for interactive dashboard. Install with: pip install plotly")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Score by State', 'Growth vs Volatility', 
                       'Top 10 High-Risk States', 'Risk Distribution'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # 1. Risk Score by State (bar chart)
    top_risky = risk_df.nlargest(15, 'risk_score')
    fig.add_trace(
        go.Bar(x=top_risky['stateid'], y=top_risky['risk_score'], 
               name='Risk Score', marker_color='coral'),
        row=1, col=1
    )
    
    # 2. Growth vs Volatility (scatter)
    # Use absolute value for size (Plotly requires size >= 0)
    # Handle NaN and negative values
    risk_scores_clean = risk_df['risk_score'].fillna(0).abs()
    min_size = 5  # Minimum marker size for visibility
    max_size = 50  # Maximum marker size
    
    # Normalize to range [min_size, max_size]
    size_range = risk_scores_clean.max() - risk_scores_clean.min()
    if size_range > 0:
        normalized_size = min_size + (risk_scores_clean - risk_scores_clean.min()) / size_range * (max_size - min_size)
    else:
        # All values are the same, use a fixed size
        normalized_size = pd.Series([(min_size + max_size) / 2] * len(risk_df))
    
    # Ensure all sizes are positive and finite (safety check)
    normalized_size = normalized_size.fillna(min_size).clip(lower=min_size, upper=max_size)
    
    # Final check - ensure no negative or infinite values
    if (normalized_size < 0).any() or (normalized_size == float('inf')).any() or (normalized_size == float('-inf')).any():
        print(f"⚠️ WARNING: Invalid size values detected, using minimum size")
        normalized_size = pd.Series([min_size] * len(risk_df))
    
    fig.add_trace(
        go.Scatter(
            x=risk_df['total_growth_pct'],
            y=risk_df['volatility_pct'],
            mode='markers',
            text=risk_df['stateid'],
            marker=dict(
                size=normalized_size,
                color=risk_df['risk_score'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            name='States'
        ),
        row=1, col=2
    )
    
    # 3. Top 10 High-Risk States
    top_10 = risk_df.nlargest(10, 'risk_score')
    fig.add_trace(
        go.Bar(x=top_10['stateid'], y=top_10['risk_score'],
               name='Top 10 Risk', marker_color='red'),
        row=2, col=1
    )
    
    # 4. Risk Distribution
    fig.add_trace(
        go.Histogram(x=risk_df['risk_score'], nbinsx=20, name='Risk Distribution'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Comprehensive Risk Assessment Dashboard",
        showlegend=False
    )
    
    return fig

