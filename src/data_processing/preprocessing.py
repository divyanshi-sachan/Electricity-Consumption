"""
Data Preprocessing Module
Feature engineering and data transformation for analysis
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class for preprocessing electricity consumption data
    """
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        pass
    
    def extract_temporal_features(self, df: pd.DataFrame, date_column: str = 'period') -> pd.DataFrame:
        """
        Extract temporal features from date column
        
        Args:
            df: Input DataFrame
            date_column: Name of date/datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return df
        
        df_processed = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_processed[date_column]):
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
        
        # Extract temporal features
        df_processed['hour'] = df_processed[date_column].dt.hour
        df_processed['day_of_week'] = df_processed[date_column].dt.dayofweek
        df_processed['day_of_month'] = df_processed[date_column].dt.day
        df_processed['week_of_year'] = df_processed[date_column].dt.isocalendar().week
        df_processed['month'] = df_processed[date_column].dt.month
        df_processed['quarter'] = df_processed[date_column].dt.quarter
        df_processed['year'] = df_processed[date_column].dt.year
        
        # Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
        df_processed['season'] = df_processed['month'].apply(
            lambda x: 1 if x in [3, 4, 5] else (2 if x in [6, 7, 8] else (3 if x in [9, 10, 11] else 4))
        )
        
        # Is weekend
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        
        # Is holiday (simplified - can be enhanced with actual holiday calendar)
        # df_processed['is_holiday'] = ...  # Implement holiday detection
        
        logger.info("Temporal features extracted")
        
        return df_processed
    
    def create_lag_features(self, 
                            df: pd.DataFrame,
                            value_column: str,
                            lags: List[int] = [1, 7, 30, 365]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Input DataFrame (should be sorted by date)
            value_column: Column to create lags for
            lags: List of lag periods (in days/periods)
            
        Returns:
            DataFrame with lag features added
        """
        if value_column not in df.columns:
            logger.warning(f"Value column '{value_column}' not found")
            return df
        
        df_processed = df.copy()
        
        for lag in lags:
            df_processed[f'{value_column}_lag_{lag}'] = df_processed[value_column].shift(lag)
        
        logger.info(f"Created lag features for {value_column}")
        
        return df_processed
    
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               value_column: str,
                               windows: List[int] = [7, 30, 90, 365]) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df: Input DataFrame
            value_column: Column to create rolling features for
            windows: List of window sizes (in days/periods)
            
        Returns:
            DataFrame with rolling features added
        """
        if value_column not in df.columns:
            logger.warning(f"Value column '{value_column}' not found")
            return df
        
        df_processed = df.copy()
        
        for window in windows:
            df_processed[f'{value_column}_rolling_mean_{window}'] = (
                df_processed[value_column].rolling(window=window, min_periods=1).mean()
            )
            df_processed[f'{value_column}_rolling_std_{window}'] = (
                df_processed[value_column].rolling(window=window, min_periods=1).std()
            )
            df_processed[f'{value_column}_rolling_max_{window}'] = (
                df_processed[value_column].rolling(window=window, min_periods=1).max()
            )
            df_processed[f'{value_column}_rolling_min_{window}'] = (
                df_processed[value_column].rolling(window=window, min_periods=1).min()
            )
        
        logger.info(f"Created rolling features for {value_column}")
        
        return df_processed
    
    def calculate_demand_supply_ratio(self,
                                     df: pd.DataFrame,
                                     demand_column: str = 'demand',
                                     supply_column: str = 'supply') -> pd.DataFrame:
        """
        Calculate demand-supply ratio
        
        Args:
            df: Input DataFrame
            demand_column: Name of demand column
            supply_column: Name of supply column
            
        Returns:
            DataFrame with demand-supply ratio added
        """
        df_processed = df.copy()
        
        if demand_column in df_processed.columns and supply_column in df_processed.columns:
            df_processed['demand_supply_ratio'] = (
                df_processed[demand_column] / df_processed[supply_column]
            )
            df_processed['reserve_margin'] = (
                (df_processed[supply_column] - df_processed[demand_column]) / 
                df_processed[demand_column] * 100
            )
            logger.info("Calculated demand-supply metrics")
        else:
            logger.warning(f"Demand or supply columns not found")
        
        return df_processed
    
    def calculate_growth_rates(self,
                              df: pd.DataFrame,
                              value_column: str,
                              periods: List[int] = [1, 7, 30, 365]) -> pd.DataFrame:
        """
        Calculate growth rates over different periods
        
        Args:
            df: Input DataFrame
            value_column: Column to calculate growth for
            periods: Periods for growth calculation
            
        Returns:
            DataFrame with growth rate features added
        """
        if value_column not in df.columns:
            logger.warning(f"Value column '{value_column}' not found")
            return df
        
        df_processed = df.copy()
        
        for period in periods:
            df_processed[f'{value_column}_growth_{period}'] = (
                df_processed[value_column].pct_change(period) * 100
            )
        
        logger.info(f"Calculated growth rates for {value_column}")
        
        return df_processed
    
    def create_regional_dummies(self, df: pd.DataFrame, region_column: str = 'region') -> pd.DataFrame:
        """
        Create dummy variables for regions
        
        Args:
            df: Input DataFrame
            region_column: Name of region column
            
        Returns:
            DataFrame with regional dummy variables
        """
        if region_column not in df.columns:
            logger.warning(f"Region column '{region_column}' not found")
            return df
        
        df_processed = df.copy()
        dummies = pd.get_dummies(df_processed[region_column], prefix='region')
        df_processed = pd.concat([df_processed, dummies], axis=1)
        
        logger.info(f"Created regional dummy variables")
        
        return df_processed
    
    def normalize_features(self,
                          df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          method: str = 'standardize') -> pd.DataFrame:
        """
        Normalize features
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize (None for all numeric)
            method: Normalization method ('standardize', 'min_max', 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_processed = df.copy()
        
        for col in columns:
            if method == 'standardize':
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                if std > 0:
                    df_processed[f'{col}_normalized'] = (df_processed[col] - mean) / std
            elif method == 'min_max':
                min_val = df_processed[col].min()
                max_val = df_processed[col].max()
                if max_val > min_val:
                    df_processed[f'{col}_normalized'] = (
                        (df_processed[col] - min_val) / (max_val - min_val)
                    )
            elif method == 'robust':
                median = df_processed[col].median()
                iqr = df_processed[col].quantile(0.75) - df_processed[col].quantile(0.25)
                if iqr > 0:
                    df_processed[f'{col}_normalized'] = (
                        (df_processed[col] - median) / iqr
                    )
        
        logger.info(f"Normalized features using {method} method")
        
        return df_processed
    
    def preprocess_dataset(self, 
                          df: pd.DataFrame,
                          date_column: str = 'period',
                          value_column: str = 'value',
                          region_column: Optional[str] = 'region') -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column
            region_column: Name of region column (if exists)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")
        
        df_processed = df.copy()
        
        # Extract temporal features
        if date_column in df_processed.columns:
            df_processed = self.extract_temporal_features(df_processed, date_column)
        
        # Create lag features
        if value_column in df_processed.columns:
            df_processed = self.create_lag_features(df_processed, value_column)
            df_processed = self.create_rolling_features(df_processed, value_column)
            df_processed = self.calculate_growth_rates(df_processed, value_column)
        
        # Calculate demand-supply ratio if columns exist
        df_processed = self.calculate_demand_supply_ratio(df_processed)
        
        # Create regional dummies
        if region_column and region_column in df_processed.columns:
            df_processed = self.create_regional_dummies(df_processed, region_column)
        
        logger.info("Data preprocessing pipeline complete")
        
        return df_processed

