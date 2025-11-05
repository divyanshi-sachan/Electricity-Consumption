"""
EIA API Data Collection Module
Collects electricity consumption data from the U.S. Energy Information Administration (EIA) API
"""
import requests
import pandas as pd
import time
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EIADataCollector:
    """
    Class to collect electricity consumption data from EIA API
    
    API Documentation: https://www.eia.gov/opendata/browser/electricity
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize EIA API collector
        
        Args:
            api_key: EIA API key (can also be set via EIA_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv('EIA_API_KEY')
        self.base_url = "https://api.eia.gov/v2"
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning(
                "EIA API key not found. Get one from https://www.eia.gov/opendata/register.php"
            )
            logger.warning("Set EIA_API_KEY environment variable or pass api_key parameter")
    
    def _make_request(self, 
                     endpoint: str,
                     params: Optional[Dict] = None,
                     max_retries: int = 3,
                     retry_delay: int = 5) -> Optional[Dict]:
        """
        Make API request with retry logic
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            
        Returns:
            JSON response as dictionary
        """
        if not self.api_key:
            raise ValueError("EIA API key is required")
        
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params['api_key'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
        
        return None
    
    def get_electricity_data(self, 
                            series_id: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            frequency: str = 'monthly',
                            offset: int = 0,
                            length: int = 5000) -> pd.DataFrame:
        """
        Fetch electricity data for a given series
        
        Args:
            series_id: EIA series identifier (e.g., 'ELEC.CONS_TOT.SE-US-99.M')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (monthly, daily, annual)
            offset: Pagination offset
            length: Number of records to fetch (max 5000)
            
        Returns:
            DataFrame with electricity data
        """
        endpoint = f"electricity/rto/region-data/data"
        
        params = {
            'frequency': frequency,
            'data[0]': 'value',
            'facets[series][]': series_id,
            'offset': offset,
            'length': length
        }
        
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        
        response = self._make_request(endpoint, params)
        
        if not response or 'response' not in response:
            logger.error(f"No data returned for series {series_id}")
            return pd.DataFrame()
        
        data = response['response'].get('data', [])
        
        if not data:
            logger.warning(f"No data found for series {series_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert period to datetime
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period'])
            df = df.sort_values('period').reset_index(drop=True)
        
        return df
    
    def get_regional_data(self, 
                         region: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get electricity consumption data for a specific region
        
        Args:
            region: Region identifier (e.g., state code, utility name)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with regional electricity data
        """
        # This is a placeholder - actual implementation depends on EIA API structure
        # You'll need to identify the correct series_id for your region
        logger.info(f"Fetching data for region: {region}")
        
        # Example: You would need to construct the appropriate series_id
        # series_id = f"ELEC.CONS_TOT.{region}.M"
        
        # For now, return empty DataFrame with expected structure
        return pd.DataFrame()
    
    def get_sector_data(self, 
                       sector: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get electricity consumption data for a specific sector
        
        Args:
            sector: Sector type ('residential', 'commercial', 'industrial', 'total')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sector electricity data
        """
        logger.info(f"Fetching data for sector: {sector}")
        
        # This is a placeholder - actual implementation depends on EIA API structure
        # You'll need to identify the correct series_id for your sector
        # series_id = f"ELEC.CONS_{sector.upper()}.US-99.M"
        
        return pd.DataFrame()
    
    def get_all_available_series(self, 
                                 category: Optional[str] = None) -> pd.DataFrame:
        """
        Get list of all available data series
        
        Args:
            category: Optional category filter
            
        Returns:
            DataFrame with available series information
        """
        endpoint = "electricity/rto/region-data"
        
        params = {}
        if category:
            params['category'] = category
        
        response = self._make_request(endpoint, params)
        
        if not response or 'response' not in response:
            return pd.DataFrame()
        
        # Parse series information from response
        # Structure depends on EIA API response format
        return pd.DataFrame()
    
    def save_data(self, 
                  df: pd.DataFrame,
                  filename: str,
                  output_dir: Optional[str] = None) -> Path:
        """
        Save collected data to file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory (default: data/raw)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        if filename.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filename.endswith('.json'):
            df.to_json(filepath, orient='records', date_format='iso')
        elif filename.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            # Default to CSV
            filepath = filepath.with_suffix('.csv')
            df.to_csv(filepath, index=False)
        
        logger.info(f"Data saved to {filepath}")
        return filepath


def main():
    """
    Example usage of EIADataCollector
    """
    # Initialize collector
    collector = EIADataCollector()
    
    # Example: Get electricity data for a series
    # series_id = "ELEC.CONS_TOT.SE-US-99.M"  # Example series ID
    # df = collector.get_electricity_data(series_id, start_date="2020-01-01")
    # print(df.head())
    
    print("EIA Data Collector initialized")
    print("Set EIA_API_KEY environment variable to use the API")


if __name__ == "__main__":
    main()

