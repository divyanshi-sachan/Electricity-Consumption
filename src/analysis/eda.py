"""
Exploratory Data Analysis functions.

This module provides functions for statistical summaries, data quality checks, and visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    return missing_df


def check_negative_values(df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, int]:
    """
    Check for negative values in numeric columns.
    
    Args:
        df: DataFrame to check
        numeric_cols: List of numeric columns. Default: ['customers', 'price', 'revenue', 'sales']
    
    Returns:
        Dictionary mapping column names to count of negative values
    """
    if numeric_cols is None:
        numeric_cols = ['customers', 'price', 'revenue', 'sales']
    
    negative_counts = {}
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                negative_counts[col] = negatives
    
    return negative_counts


def get_statistical_summary(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """
    Get statistical summary of numeric columns.
    
    Args:
        df: DataFrame
        numeric_cols: List of numeric columns. Default: ['customers', 'price', 'revenue', 'sales']
    
    Returns:
        DataFrame with statistical summary
    """
    if numeric_cols is None:
        numeric_cols = ['customers', 'price', 'revenue', 'sales']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    return df[available_cols].describe()


def analyze_state_ids(df: pd.DataFrame) -> Dict:
    """
    Analyze state ID distribution.
    
    Args:
        df: DataFrame with 'stateid' column
    
    Returns:
        Dictionary with analysis results
    """
    state_lengths = df['stateid'].astype(str).str.len().value_counts().sort_index()
    
    # Regional aggregates (more than 2 letters)
    regional = df[df['stateid'].astype(str).str.len() > 2]['stateid'].unique()
    
    # Valid states (2 letters)
    valid_states = df[df['stateid'].astype(str).str.len() == 2]['stateid'].unique()
    
    return {
        'state_length_distribution': state_lengths.to_dict(),
        'regional_aggregates': sorted(regional),
        'valid_states': sorted(valid_states),
        'num_valid_states': len(valid_states)
    }


def plot_sales_by_year(df: pd.DataFrame, ax=None):
    """
    Plot total sales by year.
    
    Args:
        df: DataFrame with 'year' and 'sales' columns
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df.groupby('year')['sales'].sum().plot(kind='line', marker='o', color='steelblue', ax=ax)
    ax.set_title('Total Sales by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Sales (MWh)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return ax


def plot_sales_by_month(df: pd.DataFrame, ax=None):
    """
    Plot average sales by month (seasonal pattern).
    
    Args:
        df: DataFrame with 'month' and 'sales' columns
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df.groupby('month')['sales'].mean().plot(kind='line', marker='o', color='steelblue', ax=ax)
    ax.set_title('Average Sales by Month (Seasonal Pattern)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Sales (MWh)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    
    return ax


def plot_price_vs_sales(df: pd.DataFrame, sample_size: int = 5000, ax=None):
    """
    Plot price change vs sales (scatter plot with sampling).
    
    Args:
        df: DataFrame with 'price_change' and 'sales' columns
        sample_size: Number of samples to plot (default: 5000)
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data for better visualization
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    ax.scatter(sample_df['price_change'], sample_df['sales'], alpha=0.3, s=10)
    ax.set_xlabel('Price Change (cents per kWh)')
    ax.set_ylabel('Sales (MWh)')
    ax.set_title('Price Change vs Sales (Sampled)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return ax


def generate_eda_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive EDA report.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        String report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
    report_lines.append("=" * 70)
    
    # Dataset overview
    report_lines.append(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if 'period' in df.columns:
        report_lines.append(f"📅 Date range: {df['period'].min()} to {df['period'].max()}")
    
    # Missing values
    missing_df = analyze_missing_values(df)
    if len(missing_df) > 0:
        report_lines.append(f"\n❗ Missing Values: {len(missing_df)} columns")
        report_lines.append(missing_df.to_string())
    else:
        report_lines.append("\n✅ No missing values")
    
    # Negative values
    negatives = check_negative_values(df)
    if negatives:
        report_lines.append(f"\n❗ Negative values found:")
        for col, count in negatives.items():
            report_lines.append(f"   - {col}: {count} rows")
    else:
        report_lines.append("\n✅ No negative values")
    
    # Statistical summary
    report_lines.append("\n📈 Statistical Summary:")
    summary = get_statistical_summary(df)
    report_lines.append(summary.to_string())
    
    return "\n".join(report_lines)

