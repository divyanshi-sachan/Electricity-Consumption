"""
Data Cleaning Module
Handles missing values, outliers, and data quality issues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Class for cleaning electricity consumption data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataCleaner
        
        Args:
            config: Configuration dictionary for cleaning parameters
        """
        self.config = config or {}
        self.missing_threshold = self.config.get('missing_threshold', 0.5)
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.z_score_threshold = self.config.get('z_score_threshold', 3)
        
    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Assess data quality and return summary statistics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Add statistics for numeric columns
        if quality_report['numeric_columns']:
            quality_report['numeric_stats'] = df[quality_report['numeric_columns']].describe().to_dict()
        
        logger.info(f"Data quality assessment complete: {quality_report['total_rows']} rows, "
                   f"{quality_report['total_columns']} columns")
        
        return quality_report
    
    def handle_missing_values(self, 
                             df: pd.DataFrame,
                             strategy: str = 'forward_fill',
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                - 'forward_fill': Forward fill (default for time series)
                - 'backward_fill': Backward fill
                - 'interpolate': Linear interpolation
                - 'mean': Fill with mean (for non-time series)
                - 'median': Fill with median
                - 'drop': Drop rows with missing values
            columns: Specific columns to process (None for all)
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        for col in columns:
            if df_cleaned[col].isnull().sum() == 0:
                continue
            
            missing_count = df_cleaned[col].isnull().sum()
            missing_pct = missing_count / len(df_cleaned) * 100
            
            logger.info(f"Column {col}: {missing_count} missing values ({missing_pct:.2f}%)")
            
            if missing_pct > self.missing_threshold * 100:
                logger.warning(f"Column {col} has {missing_pct:.2f}% missing values. "
                             f"Consider dropping or investigating.")
            
            if strategy == 'forward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            elif strategy == 'backward_fill':
                df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
            elif strategy == 'interpolate':
                df_cleaned[col] = df_cleaned[col].interpolate()
            elif strategy == 'mean':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strategy == 'median':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
            else:
                logger.warning(f"Unknown strategy: {strategy}. Skipping column {col}")
        
        return df_cleaned
    
    def detect_outliers(self, 
                       df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: Optional[str] = None) -> pd.DataFrame:
        """
        Detect outliers in numeric columns
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None for all numeric columns)
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
                If None, uses self.outlier_method
        
        Returns:
            DataFrame with boolean columns indicating outliers
        """
        if method is None:
            method = self.outlier_method
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_flags[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > self.z_score_threshold
                
            else:
                logger.warning(f"Unknown method: {method}")
        
        return outlier_flags
    
    def handle_outliers(self,
                       df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'cap',
                       outlier_flags: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Handle outliers in the dataset
        
        Args:
            df: Input DataFrame
            columns: Columns to process (None for all numeric columns)
            method: Treatment method
                - 'cap': Cap outliers at bounds
                - 'remove': Remove outlier rows
                - 'log': Apply log transformation
            outlier_flags: Pre-computed outlier flags (if None, will detect)
            
        Returns:
            DataFrame with outliers handled
        """
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if outlier_flags is None:
            outlier_flags = self.detect_outliers(df, columns)
        
        for col in columns:
            flag_col = f'{col}_outlier'
            if flag_col not in outlier_flags.columns:
                continue
            
            outlier_count = outlier_flags[flag_col].sum()
            if outlier_count == 0:
                continue
            
            logger.info(f"Column {col}: {outlier_count} outliers detected")
            
            if method == 'cap':
                # Cap at percentiles
                lower_bound = df_cleaned[col].quantile(0.01)
                upper_bound = df_cleaned[col].quantile(0.99)
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                
            elif method == 'remove':
                df_cleaned = df_cleaned[~outlier_flags[flag_col]]
                
            elif method == 'log':
                # Apply log transformation (only if all values are positive)
                if (df_cleaned[col] > 0).all():
                    df_cleaned[col] = np.log1p(df_cleaned[col])
                else:
                    logger.warning(f"Cannot apply log transform to {col}: negative values present")
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates (None for all columns)
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=subset)
        removed_count = initial_count - len(df_cleaned)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return df_cleaned
    
    def standardize_regions(self, df: pd.DataFrame, region_column: str = 'region') -> pd.DataFrame:
        """
        Standardize region names/codes
        
        Args:
            df: Input DataFrame
            region_column: Name of region column
            
        Returns:
            DataFrame with standardized region names
        """
        if region_column not in df.columns:
            logger.warning(f"Region column '{region_column}' not found")
            return df
        
        df_cleaned = df.copy()
        
        # Common standardization mappings
        # Add more as needed based on your data
        region_mappings = {
            # Example mappings - adjust based on actual data
            'CA': 'California',
            'TX': 'Texas',
            'NY': 'New York',
            # Add more mappings
        }
        
        # Apply mappings
        df_cleaned[region_column] = df_cleaned[region_column].replace(region_mappings)
        
        # Standardize case
        df_cleaned[region_column] = df_cleaned[region_column].str.strip().str.title()
        
        logger.info(f"Standardized region names in column '{region_column}'")
        
        return df_cleaned
    
    def clean_dataset(self, 
                     df: pd.DataFrame,
                     missing_strategy: str = 'forward_fill',
                     outlier_method: str = 'cap',
                     remove_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data cleaning pipeline
        
        Args:
            df: Input DataFrame
            missing_strategy: Strategy for missing values
            outlier_method: Method for handling outliers
            remove_duplicates: Whether to remove duplicates
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        logger.info("Starting data cleaning pipeline")
        
        # Initial assessment
        initial_report = self.assess_data_quality(df)
        
        df_cleaned = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            df_cleaned = self.remove_duplicates(df_cleaned)
        
        # Handle missing values
        df_cleaned = self.handle_missing_values(df_cleaned, strategy=missing_strategy)
        
        # Handle outliers
        df_cleaned = self.handle_outliers(df_cleaned, method=outlier_method)
        
        # Standardize regions
        if 'region' in df_cleaned.columns:
            df_cleaned = self.standardize_regions(df_cleaned)
        
        # Final assessment
        final_report = self.assess_data_quality(df_cleaned)
        
        cleaning_report = {
            'initial': initial_report,
            'final': final_report,
            'rows_removed': initial_report['total_rows'] - final_report['total_rows'],
            'duplicates_removed': initial_report['duplicate_rows']
        }
        
        logger.info("Data cleaning pipeline complete")
        
        return df_cleaned, cleaning_report

