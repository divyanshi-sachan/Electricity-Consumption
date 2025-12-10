"""
Data preprocessing functions for electricity consumption data.

This module handles data cleaning, missing value imputation, encoding, and feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional, Dict
from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw data from file or find the most recent raw data file.
    
    Args:
        file_path: Optional path to specific file. If None, finds most recent.
    
    Returns:
        DataFrame with raw data
    """
    if file_path is None:
        if not RAW_DATA_DIR.exists():
            raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
        
        csv_files = list(RAW_DATA_DIR.glob('eia_retail_sales_raw_*.csv'))
        if not csv_files:
            raise FileNotFoundError("No raw data files found. Please run data collection first.")
        
        file_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"📂 Loading from: {file_path.name}")
    
    df = pd.read_csv(file_path)
    return df


def convert_period_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert period column to datetime, handling multiple formats.
    
    Args:
        df: DataFrame with 'period' column
    
    Returns:
        DataFrame with converted period column
    """
    df = df.copy()
    
    if 'period' not in df.columns:
        raise ValueError("DataFrame must have 'period' column")
    
    # Try different datetime formats
    try:
        df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
    except ValueError:
        try:
            df['period'] = pd.to_datetime(df['period'], format='%Y-%m-%d')
        except ValueError:
            df['period'] = pd.to_datetime(df['period'], errors='coerce')
            if df['period'].dtype == 'datetime64[ns]':
                df['period'] = df['period'].dt.to_period('M').dt.to_timestamp()
    
    return df


def filter_valid_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to valid U.S. states (2-letter codes only).
    
    Args:
        df: DataFrame with 'stateid' column
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    print(f"📊 Before filtering: {len(df):,} rows")
    
    # Filter to 2-letter state codes
    df = df[df['stateid'].astype(str).str.len() == 2].copy()
    print(f"   After state filter: {len(df):,} rows")
    
    return df


def filter_sector(df: pd.DataFrame, sector: str = 'ALL') -> pd.DataFrame:
    """
    Filter to specific sector.
    
    Args:
        df: DataFrame with 'sectorid' column
        sector: Sector ID to filter (default: 'ALL')
    
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df = df[df['sectorid'] == sector].copy()
    print(f"   After sector filter ({sector}): {len(df):,} rows")
    
    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """
    Convert specified columns to numeric type.
    
    Args:
        df: DataFrame
        numeric_cols: List of column names to convert. Default: ['customers', 'price', 'revenue', 'sales']
    
    Returns:
        DataFrame with converted columns
    """
    if numeric_cols is None:
        numeric_cols = ['customers', 'price', 'revenue', 'sales']
    
    df = df.copy()
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def handle_missing_values(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values using forward fill and mean imputation.
    
    Args:
        df: DataFrame
        numeric_cols: List of numeric columns to process
    
    Returns:
        DataFrame with missing values handled
    """
    if numeric_cols is None:
        numeric_cols = ['customers', 'price', 'revenue', 'sales']
    
    df = df.copy()
    
    # Clip revenue to non-negative
    if 'revenue' in df.columns:
        df['revenue'] = df['revenue'].clip(lower=0)
    
    # Fill customers with 0
    if 'customers' in df.columns:
        df['customers'] = df['customers'].fillna(0)
    
    # Forward fill and mean imputation for other numeric columns
    for col in numeric_cols:
        if col in df.columns and col != 'customers':
            # Forward fill by state
            df[col] = df.groupby('stateid')[col].ffill()
            # Mean imputation if still missing
            if df[col].isnull().any():
                df[col] = df.groupby('stateid')[col].transform(lambda x: x.fillna(x.mean()))
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    ⚠️ IMPORTANT: Does NOT create leakage features like:
    - revenue_per_customer (derived from revenue)
    - sales_per_customer (derived from target 'sales')
    - price (can leak information)
    
    Instead creates proper time series features:
    - lag features (lag_1_month, lag_12_month)
    - rolling statistics (rolling_mean_3, rolling_mean_12)
    - temporal features (year, month, season)
    
    Args:
        df: DataFrame with period, customers, sales columns
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    df = df.sort_values(['stateid', 'period']).reset_index(drop=True)
    
    # Extract year and month from period
    if 'period' in df.columns and df['period'].dtype == 'datetime64[ns]':
        if 'year' not in df.columns:
            df['year'] = df['period'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['period'].dt.month
        # Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
        if 'season' not in df.columns:
            df['season'] = df['period'].dt.month % 12 // 3 + 1
    
    # ⚠️ REMOVED: revenue_per_customer and sales_per_customer (leakage features)
    # ⚠️ REMOVED: price (can leak information)
    
    # Create time series features per state
    if 'sales' in df.columns:
        # Lag features (1 month and 12 months ago)
        df['lag_1_month'] = df.groupby('stateid')['sales'].shift(1)
        df['lag_12_month'] = df.groupby('stateid')['sales'].shift(12)
        
        # Rolling statistics
        df['rolling_mean_3'] = df.groupby('stateid')['sales'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['rolling_mean_12'] = df.groupby('stateid')['sales'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())
        df['rolling_std_12'] = df.groupby('stateid')['sales'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
    
    # Replace inf and NaN with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df


def encode_categorical_features(df: pd.DataFrame, encode_stateid: bool = True) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame with categorical columns
        encode_stateid: Whether to encode stateid column
    
    Returns:
        Tuple of (DataFrame with encoded features, LabelEncoder for stateid if used)
    """
    df = df.copy()
    le_state = None
    
    if encode_stateid and 'stateid' in df.columns:
        if df['stateid'].dtype == 'object':
            le_state = LabelEncoder()
            df['stateid_encoded'] = le_state.fit_transform(df['stateid'].astype(str))
    
    return df, le_state


def prepare_modeling_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'sales'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix and target vector for modeling.
    
    Args:
        df: Preprocessed DataFrame
        feature_cols: List of feature column names. If None, uses default features.
        target_col: Name of target column
    
    Returns:
        Tuple of (feature matrix X, target vector y, feature column names)
    """
    if feature_cols is None:
        # ⚠️ REMOVED LEAKAGE FEATURES:
        # - 'price' (can leak information)
        # - 'revenue' (can leak information)
        # - 'revenue_per_customer' (derived from revenue)
        # - 'sales_per_customer' (derived from target 'sales')
        
        # Use only non-leaky features
        feature_cols = ['customers', 'year', 'month']
        
        # Add time series features if they exist
        if 'season' in df.columns:
            feature_cols.append('season')
        if 'lag_1_month' in df.columns:
            feature_cols.append('lag_1_month')
        if 'lag_12_month' in df.columns:
            feature_cols.append('lag_12_month')
        if 'rolling_mean_3' in df.columns:
            feature_cols.append('rolling_mean_3')
        if 'rolling_mean_12' in df.columns:
            feature_cols.append('rolling_mean_12')
        if 'rolling_std_12' in df.columns:
            feature_cols.append('rolling_std_12')
        
        # Add stateid encoding if available (important for per-state modeling)
        if 'stateid_encoded' in df.columns:
            feature_cols.append('stateid_encoded')
        elif 'stateid' in df.columns:
            feature_cols.append('stateid')
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
        print(f"   Using available features: {feature_cols}")
    
    # Create feature matrix and target vector
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    y = y.replace([np.inf, -np.inf], 0).fillna(0)
    
    return X, y, feature_cols


def validate_state_data_quality(
    df: pd.DataFrame,
    state_id: str,
    min_valid_months: int = 24,
    max_missing_pct: float = 0.3
) -> Tuple[bool, Dict]:
    """
    Validate data quality for a specific state.
    
    Args:
        df: DataFrame with state data
        state_id: State ID to validate
        min_valid_months: Minimum number of valid months required
        max_missing_pct: Maximum percentage of missing sales allowed
    
    Returns:
        Tuple of (is_valid, quality_metrics)
    """
    state_data = df[df['stateid'] == state_id].copy()
    
    if len(state_data) == 0:
        return False, {'reason': 'No data', 'months': 0}
    
    # Check for sufficient data points
    if len(state_data) < min_valid_months:
        return False, {'reason': f'Insufficient data ({len(state_data)} < {min_valid_months})', 'months': len(state_data)}
    
    # Check missing sales percentage
    if 'sales' in state_data.columns:
        missing_sales = state_data['sales'].isnull().sum()
        missing_pct = missing_sales / len(state_data)
        if missing_pct > max_missing_pct:
            return False, {'reason': f'Too many missing sales ({missing_pct:.1%} > {max_missing_pct:.1%})', 'missing_pct': missing_pct}
        
        # Check for too many zeros (might indicate data quality issue)
        zero_sales = (state_data['sales'] == 0).sum()
        zero_pct = zero_sales / len(state_data)
        if zero_pct > 0.5:  # More than 50% zeros is suspicious
            return False, {'reason': f'Too many zero sales ({zero_pct:.1%})', 'zero_pct': zero_pct}
        
        # Check for reasonable variance (constant values are suspicious)
        if state_data['sales'].std() == 0:
            return False, {'reason': 'No variance in sales (constant values)', 'std': 0}
    
    return True, {'months': len(state_data), 'valid': True}


def filter_quality_states(
    df: pd.DataFrame,
    min_valid_months: int = 24,
    max_missing_pct: float = 0.3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter out states with poor data quality.
    
    Args:
        df: DataFrame with state data
        min_valid_months: Minimum number of valid months required
        max_missing_pct: Maximum percentage of missing sales allowed
    
    Returns:
        Tuple of (filtered DataFrame, list of excluded state IDs)
    """
    df = df.copy()
    excluded_states = []
    
    unique_states = df['stateid'].unique()
    for state_id in unique_states:
        is_valid, metrics = validate_state_data_quality(
            df, state_id, min_valid_months, max_missing_pct
        )
        if not is_valid:
            excluded_states.append(state_id)
            print(f"   ⚠️  Excluding {state_id}: {metrics.get('reason', 'Unknown issue')}")
    
    if excluded_states:
        df_filtered = df[~df['stateid'].isin(excluded_states)].copy()
        print(f"   📊 Filtered out {len(excluded_states)} states with poor data quality")
        print(f"   ✅ Remaining states: {df_filtered['stateid'].nunique()}")
    else:
        df_filtered = df
    
    return df_filtered, excluded_states


def preprocess_pipeline(
    file_path: Optional[Path] = None,
    sector: str = 'ALL',
    encode_stateid: bool = True,
    filter_quality: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], Optional[LabelEncoder]]:
    """
    Complete preprocessing pipeline from raw data to modeling-ready data.
    
    Args:
        file_path: Optional path to raw data file
        sector: Sector to filter (default: 'ALL')
        encode_stateid: Whether to encode stateid
    
    Returns:
        Tuple of (df_model, X, y, feature_cols, le_state)
    """
    # Load raw data
    df = load_raw_data(file_path)
    
    # Convert period to datetime
    df = convert_period_to_datetime(df)
    
    # Filter to valid states
    df = filter_valid_states(df)
    
    # Filter to specific sector
    df = filter_sector(df, sector)
    
    # Convert numeric columns
    df = convert_numeric_columns(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Filter out states with poor data quality (if enabled)
    if filter_quality:
        df, excluded_states = filter_quality_states(df, min_valid_months=24, max_missing_pct=0.3)
        if excluded_states:
            print(f"   ⚠️  Excluded {len(excluded_states)} states: {excluded_states[:5]}{'...' if len(excluded_states) > 5 else ''}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical features
    df, le_state = encode_categorical_features(df, encode_stateid)
    
    # Sort by period for time series analysis
    df = df.sort_values('period').reset_index(drop=True)
    
    # Prepare modeling data
    X, y, feature_cols = prepare_modeling_data(df)
    
    print(f"✅ Preprocessing complete: {len(df):,} rows")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Date range: {df['period'].min()} to {df['period'].max()}")
    print(f"   States: {df['stateid'].nunique()}")
    
    return df, X, y, feature_cols, le_state


def save_processed_data(df: pd.DataFrame, filename: Optional[str] = None) -> Path:
    """
    Save processed data to processed directory.
    
    Args:
        df: Processed DataFrame
        filename: Optional filename. If None, generates timestamped filename.
    
    Returns:
        Path to saved file
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"df_model_{timestamp}.csv"
    
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    print(f"✅ Saved processed data to: {file_path}")
    
    return file_path


def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load processed data from file or find the most recent processed data file.
    
    Args:
        file_path: Optional path to specific file. If None, finds most recent.
    
    Returns:
        DataFrame with processed data
    """
    if file_path is None:
        if not PROCESSED_DATA_DIR.exists():
            raise FileNotFoundError(f"Processed data directory not found: {PROCESSED_DATA_DIR}")
        
        csv_files = list(PROCESSED_DATA_DIR.glob('df_model_*.csv'))
        if not csv_files:
            raise FileNotFoundError("No processed data files found. Please run preprocessing first.")
        
        file_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"📂 Loading from: {file_path.name}")
    
    df = pd.read_csv(file_path)
    
    # Convert period to datetime if it exists
    if 'period' in df.columns:
        df = convert_period_to_datetime(df)
    
    return df

