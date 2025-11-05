"""
Exploratory Data Analysis Module
Performs exploratory analysis on electricity consumption data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExploratoryAnalyzer:
    """
    Class for exploratory data analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ExploratoryAnalyzer
        
        Args:
            df: Input DataFrame with electricity consumption data
        """
        self.df = df.copy()
    
    def temporal_analysis(self, 
                         value_column: str = 'value',
                         date_column: str = 'period') -> Dict:
        """
        Analyze temporal patterns in electricity consumption
        
        Args:
            value_column: Column with consumption values
            date_column: Column with dates
            
        Returns:
            Dictionary with temporal analysis results
        """
        if date_column not in self.df.columns or value_column not in self.df.columns:
            logger.error("Required columns not found")
            return {}
        
        df_analysis = self.df[[date_column, value_column]].copy()
        df_analysis[date_column] = pd.to_datetime(df_analysis[date_column])
        df_analysis = df_analysis.sort_values(date_column)
        
        results = {}
        
        # Daily patterns
        if 'hour' in self.df.columns:
            results['daily_pattern'] = self.df.groupby('hour')[value_column].agg(['mean', 'std', 'min', 'max']).to_dict()
        
        # Weekly patterns
        if 'day_of_week' in self.df.columns:
            results['weekly_pattern'] = self.df.groupby('day_of_week')[value_column].agg(['mean', 'std']).to_dict()
        
        # Monthly patterns
        if 'month' in self.df.columns:
            results['monthly_pattern'] = self.df.groupby('month')[value_column].agg(['mean', 'std']).to_dict()
        
        # Seasonal patterns
        if 'season' in self.df.columns:
            results['seasonal_pattern'] = self.df.groupby('season')[value_column].agg(['mean', 'std']).to_dict()
        
        # Yearly trends
        if 'year' in self.df.columns:
            results['yearly_trend'] = self.df.groupby('year')[value_column].agg(['mean', 'sum']).to_dict()
        
        # Peak identification
        results['peak_value'] = df_analysis[value_column].max()
        results['peak_date'] = df_analysis.loc[df_analysis[value_column].idxmax(), date_column]
        results['average_value'] = df_analysis[value_column].mean()
        results['std_value'] = df_analysis[value_column].std()
        
        logger.info("Temporal analysis complete")
        
        return results
    
    def regional_analysis(self, 
                         region_column: str = 'region',
                         value_column: str = 'value') -> Dict:
        """
        Analyze regional variations in electricity consumption
        
        Args:
            region_column: Column with region identifiers
            value_column: Column with consumption values
            
        Returns:
            Dictionary with regional analysis results
        """
        if region_column not in self.df.columns:
            logger.warning(f"Region column '{region_column}' not found")
            return {}
        
        results = {}
        
        # Regional statistics
        regional_stats = self.df.groupby(region_column)[value_column].agg([
            'mean', 'std', 'min', 'max', 'sum', 'count'
        ]).to_dict()
        
        results['regional_stats'] = regional_stats
        
        # Top consuming regions
        top_regions = self.df.groupby(region_column)[value_column].sum().sort_values(ascending=False).head(10)
        results['top_regions'] = top_regions.to_dict()
        
        # Regional growth rates (if year column exists)
        if 'year' in self.df.columns:
            regional_growth = self.df.groupby([region_column, 'year'])[value_column].sum().reset_index()
            regional_growth = regional_growth.pivot(index=region_column, columns='year', values=value_column)
            growth_rates = ((regional_growth.iloc[:, -1] - regional_growth.iloc[:, 0]) / 
                           regional_growth.iloc[:, 0] * 100)
            results['regional_growth_rates'] = growth_rates.to_dict()
        
        logger.info("Regional analysis complete")
        
        return results
    
    def sector_analysis(self,
                       sector_column: str = 'sector',
                       value_column: str = 'value') -> Dict:
        """
        Analyze differences between residential, commercial, and industrial sectors
        
        Args:
            sector_column: Column with sector identifiers
            value_column: Column with consumption values
            
        Returns:
            Dictionary with sector analysis results
        """
        if sector_column not in self.df.columns:
            logger.warning(f"Sector column '{sector_column}' not found")
            return {}
        
        results = {}
        
        # Sector statistics
        sector_stats = self.df.groupby(sector_column)[value_column].agg([
            'mean', 'std', 'min', 'max', 'sum'
        ]).to_dict()
        
        results['sector_stats'] = sector_stats
        
        # Sector proportions
        sector_totals = self.df.groupby(sector_column)[value_column].sum()
        sector_proportions = (sector_totals / sector_totals.sum() * 100).to_dict()
        results['sector_proportions'] = sector_proportions
        
        # Temporal patterns by sector
        if 'hour' in self.df.columns:
            sector_hourly = self.df.groupby([sector_column, 'hour'])[value_column].mean().unstack(level=0)
            results['sector_hourly_patterns'] = sector_hourly.to_dict()
        
        logger.info("Sector analysis complete")
        
        return results
    
    def correlation_analysis(self, 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            columns: Columns to include (None for all numeric columns)
            
        Returns:
            Correlation matrix DataFrame
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols
        
        available_cols = [col for col in columns if col in self.df.columns]
        
        if not available_cols:
            logger.warning("No numeric columns found for correlation analysis")
            return pd.DataFrame()
        
        correlation_matrix = self.df[available_cols].corr()
        
        logger.info("Correlation analysis complete")
        
        return correlation_matrix
    
    def identify_demand_spikes(self,
                              value_column: str = 'value',
                              threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify demand spikes that could stress the grid
        
        Args:
            value_column: Column with consumption values
            threshold: Z-score threshold for spike detection
            
        Returns:
            DataFrame with identified spikes
        """
        if value_column not in self.df.columns:
            logger.error(f"Value column '{value_column}' not found")
            return pd.DataFrame()
        
        mean = self.df[value_column].mean()
        std = self.df[value_column].std()
        
        z_scores = np.abs((self.df[value_column] - mean) / std)
        spikes = self.df[z_scores > threshold].copy()
        spikes['z_score'] = z_scores[z_scores > threshold]
        
        logger.info(f"Identified {len(spikes)} demand spikes")
        
        return spikes
    
    def calculate_peak_to_average_ratio(self,
                                       value_column: str = 'value',
                                       period: str = 'daily') -> float:
        """
        Calculate peak-to-average ratio
        
        Args:
            value_column: Column with consumption values
            period: Period for calculation ('daily', 'weekly', 'monthly')
            
        Returns:
            Peak-to-average ratio
        """
        if value_column not in self.df.columns:
            logger.error(f"Value column '{value_column}' not found")
            return 0.0
        
        if period == 'daily' and 'hour' in self.df.columns:
            hourly_avg = self.df.groupby('hour')[value_column].mean()
            peak = hourly_avg.max()
            average = hourly_avg.mean()
        else:
            peak = self.df[value_column].max()
            average = self.df[value_column].mean()
        
        ratio = peak / average if average > 0 else 0.0
        
        logger.info(f"Peak-to-average ratio ({period}): {ratio:.2f}")
        
        return ratio
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(self.df),
            'date_range': {
                'start': self.df.select_dtypes(include=['datetime64']).min().min() if 
                        self.df.select_dtypes(include=['datetime64']).shape[1] > 0 else None,
                'end': self.df.select_dtypes(include=['datetime64']).max().max() if 
                      self.df.select_dtypes(include=['datetime64']).shape[1] > 0 else None
            },
            'numeric_summary': self.df.select_dtypes(include=[np.number]).describe().to_dict(),
            'categorical_summary': {}
        }
        
        for col in self.df.select_dtypes(include=['object']).columns:
            summary['categorical_summary'][col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(10).to_dict()
            }
        
        logger.info("Summary statistics generated")
        
        return summary

