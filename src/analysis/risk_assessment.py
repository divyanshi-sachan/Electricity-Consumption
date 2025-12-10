"""
Risk assessment and gap analysis functions.

This module provides functions to identify high-risk regions, calculate demand-supply gaps,
and generate policy recommendations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def calculate_demand_supply_gap(
    forecast_df: pd.DataFrame,
    current_capacity_df: Optional[pd.DataFrame] = None,
    state_id_col: str = 'stateid',
    demand_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate demand-supply gap for each state.
    
    Args:
        forecast_df: DataFrame with forecasted demand
        current_capacity_df: Optional DataFrame with current generation capacity per state
        state_id_col: Name of state ID column
        demand_col: Name of demand/value column (auto-detected if None)
    
    Returns:
        DataFrame with gap analysis
    """
    forecast_df = forecast_df.copy()
    
    # Auto-detect demand column if not specified
    if demand_col is None:
        if 'sales' in forecast_df.columns:
            demand_col = 'sales'
        elif 'value' in forecast_df.columns:
            demand_col = 'value'
        else:
            raise ValueError("No 'sales' or 'value' column found in forecast_df")
    
    if demand_col not in forecast_df.columns:
        raise ValueError(f"Column '{demand_col}' not found in forecast_df. Available columns: {list(forecast_df.columns)}")
    
    # Filter to only actual forecast rows (exclude component rows)
    forecast_df_filtered = forecast_df.copy()
    if 'type' in forecast_df_filtered.columns:
        # Filter to keep only actual forecast types (not components)
        valid_types = forecast_df_filtered['type'].unique()
        forecast_keywords = ['forecast', 'hybrid', 'final', 'prediction']
        matching_types = [t for t in valid_types if any(kw in str(t).lower() for kw in forecast_keywords)]
        
        if len(matching_types) > 0:
            forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['type'].isin(matching_types)]
        elif any('component' not in str(t).lower() for t in valid_types):
            # Use non-component types
            non_component_types = [t for t in valid_types if 'component' not in str(t).lower()]
            if len(non_component_types) > 0:
                forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['type'].isin(non_component_types)]
    
    # Convert period to datetime if needed
    if 'period' in forecast_df_filtered.columns:
        forecast_df_filtered['period'] = pd.to_datetime(forecast_df_filtered['period'])
        
        # Filter to only years with non-zero data
        available_years = sorted(forecast_df_filtered['period'].dt.year.unique())
        years_with_data = []
        for y in available_years:
            year_data = forecast_df_filtered[forecast_df_filtered['period'].dt.year == y]
            if (year_data[demand_col] > 0).sum() > 0:
                years_with_data.append(y)
        
        if len(years_with_data) > 0:
            # Use only years with actual data (exclude years with all zeros like 2031-2035)
            latest_year_with_data = max(years_with_data)
            print(f"📊 Risk Assessment: Using forecast data from years with actual values")
            print(f"   Available years: {available_years}")
            print(f"   Years with data: {years_with_data}")
            print(f"   Using latest year with data: {latest_year_with_data}")
            forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['period'].dt.year.isin(years_with_data)].copy()
        else:
            import warnings
            warnings.warn("No years with non-zero forecast data found!")
    
    # Filter out zero/negative values
    forecast_df_filtered = forecast_df_filtered[forecast_df_filtered[demand_col] > 0].copy()
    
    if len(forecast_df_filtered) == 0:
        import warnings
        warnings.warn("No valid forecast data after filtering. Using all data.")
        forecast_df_filtered = forecast_df.copy()
        if 'type' in forecast_df_filtered.columns:
            # Still try to filter by type if possible
            valid_types = forecast_df_filtered['type'].unique()
            forecast_keywords = ['forecast', 'hybrid', 'final', 'prediction']
            matching_types = [t for t in valid_types if any(kw in str(t).lower() for kw in forecast_keywords)]
            if len(matching_types) > 0:
                forecast_df_filtered = forecast_df_filtered[forecast_df_filtered['type'].isin(matching_types)]
    
    # Get forecasted demand by state
    state_forecasts = forecast_df_filtered.groupby(state_id_col)[demand_col].agg([
        'mean', 'max', 'min'
    ]).reset_index()
    state_forecasts.columns = [state_id_col, 'forecast_avg_demand', 'forecast_peak_demand', 'forecast_min_demand']
    
    # If capacity data provided, calculate gap
    if current_capacity_df is not None:
        gap_df = state_forecasts.merge(
            current_capacity_df,
            on=state_id_col,
            how='left'
        )
        
        # Calculate gaps
        gap_df['demand_gap_avg'] = gap_df['forecast_avg_demand'] - gap_df.get('current_capacity', 0)
        gap_df['demand_gap_peak'] = gap_df['forecast_peak_demand'] - gap_df.get('current_capacity', 0)
        gap_df['gap_pct'] = (gap_df['demand_gap_peak'] / gap_df.get('current_capacity', 1) * 100).fillna(0)
        
        # Risk level based on gap
        gap_df['risk_level'] = gap_df['gap_pct'].apply(
            lambda x: 'Critical' if x > 20 else ('High' if x > 10 else ('Medium' if x > 5 else 'Low'))
        )
    else:
        # Use forecast statistics to estimate risk
        gap_df = state_forecasts.copy()
        gap_df['risk_level'] = 'Unknown'  # Need capacity data for proper assessment
    
    return gap_df


def identify_high_risk_regions(
    gap_df: pd.DataFrame,
    growth_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify high-risk regions based on gap and growth metrics.
    
    Args:
        gap_df: DataFrame from calculate_demand_supply_gap
        growth_metrics_df: DataFrame from calculate_state_growth_metrics
    
    Returns:
        DataFrame with high-risk regions
    """
    # Merge gap and growth data
    risk_df = gap_df.merge(
        growth_metrics_df[['stateid', 'total_growth_pct', 'volatility_pct']],
        on='stateid',
        how='inner'
    )
    
    # Calculate risk score (higher = more risk)
    risk_df['risk_score'] = (
        (risk_df.get('gap_pct', 0) / 100) * 0.4 +  # Gap weight
        (risk_df['total_growth_pct'] / 100) * 0.3 +  # Growth weight
        (risk_df['volatility_pct'] / 100) * 0.3  # Volatility weight
    ) * 100
    
    # Classify risk levels
    risk_df['risk_category'] = risk_df['risk_score'].apply(
        lambda x: 'Critical' if x > 15 else ('High' if x > 10 else ('Medium' if x > 5 else 'Low'))
    )
    
    # Sort by risk score
    risk_df = risk_df.sort_values('risk_score', ascending=False)
    
    return risk_df


def generate_policy_recommendations(
    risk_df: pd.DataFrame,
    growth_metrics_df: pd.DataFrame
) -> Dict[str, List[Dict]]:
    """
    Generate policy recommendations for different risk categories.
    
    Args:
        risk_df: DataFrame from identify_high_risk_regions
        growth_metrics_df: DataFrame with growth metrics
    
    Returns:
        Dictionary with recommendations by category
    """
    recommendations = {
        'high_growth': [],
        'declining': [],
        'volatile': [],
        'critical_risk': []
    }
    
    # High growth states
    high_growth = growth_metrics_df[growth_metrics_df['total_growth_pct'] > 15]
    for _, row in high_growth.iterrows():
        recommendations['high_growth'].append({
            'state': row['stateid'],
            'growth_pct': row['total_growth_pct'],
            'recommendations': [
                'Invest in grid expansion and transmission infrastructure',
                'Deploy renewable energy (solar, wind) to meet growing demand',
                'Install battery storage systems for peak demand management',
                'Implement smart meters and demand-response programs',
                'Consider time-of-use pricing to shift peak demand'
            ]
        })
    
    # Declining states
    declining = growth_metrics_df[growth_metrics_df['total_growth_pct'] < -5]
    for _, row in declining.iterrows():
        recommendations['declining'].append({
            'state': row['stateid'],
            'growth_pct': row['total_growth_pct'],
            'recommendations': [
                'Optimize existing infrastructure investments',
                'Focus on maintenance rather than expansion',
                'Consider repurposing excess capacity',
                'Explore energy export opportunities to neighboring states',
                'Invest in energy efficiency programs'
            ]
        })
    
    # Volatile states
    volatile = growth_metrics_df[growth_metrics_df['volatility_pct'] > 20]
    for _, row in volatile.iterrows():
        recommendations['volatile'].append({
            'state': row['stateid'],
            'volatility_pct': row['volatility_pct'],
            'recommendations': [
                'Implement flexible generation capacity',
                'Deploy demand-side management programs',
                'Use energy storage to smooth demand fluctuations',
                'Develop real-time pricing mechanisms',
                'Invest in grid resilience and backup systems'
            ]
        })
    
    # Critical risk regions
    critical = risk_df[risk_df['risk_category'] == 'Critical']
    for _, row in critical.iterrows():
        recommendations['critical_risk'].append({
            'state': row['stateid'],
            'risk_score': row.get('risk_score', 0),
            'gap_pct': row.get('gap_pct', 0),
            'recommendations': [
                'URGENT: Immediate capacity expansion required',
                'Deploy emergency generation capacity',
                'Implement aggressive demand-response programs',
                'Consider load shedding protocols',
                'Accelerate renewable energy deployment',
                'Explore inter-state power import agreements'
            ]
        })
    
    return recommendations


def create_risk_heatmap(
    risk_df: pd.DataFrame,
    forecast_year: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a heatmap showing risk levels by state.
    
    Args:
        risk_df: DataFrame with risk assessment
        forecast_year: Year to show in heatmap (defaults to 2030 if None, as 2031+ may have no data)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Use 2030 as default since 2031-2035 have no data in forecast files
    if forecast_year is None:
        forecast_year = 2030
    # Prepare data for heatmap
    heatmap_data = risk_df.pivot_table(
        values='risk_score',
        index='stateid',
        columns='risk_category',
        aggfunc='mean',
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red = high risk)
        cbar_kws={'label': 'Risk Score'},
        ax=ax
    )
    
    ax.set_title(f'Risk Assessment Heatmap - {forecast_year}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Risk Category', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    
    plt.tight_layout()
    return fig

