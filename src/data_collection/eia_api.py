"""
EIA API data collection functions.

This module handles fetching electricity retail sales data from the EIA API.
"""
import pandas as pd
import requests
from typing import Tuple
from src.config import EIA_API_KEY, EIA_RETAIL_SALES_URL, DEFAULT_BATCH_SIZE


def fetch_data(offset: int = 0, length: int = DEFAULT_BATCH_SIZE) -> Tuple[pd.DataFrame, int, bool]:
    """
    Fetch retail sales data from EIA API.
    
    Args:
        offset: Starting record number (for pagination)
        length: Number of records to fetch (max 5000 per request)
    
    Returns:
        Tuple of (DataFrame, total_count, has_more)
        - DataFrame: Fetched data
        - total_count: Total number of records available
        - has_more: Whether there are more records to fetch
    """
    url = EIA_RETAIL_SALES_URL.format(offset, length)
    
    # Add API key to URL
    url_with_key = url + f"&api_key={EIA_API_KEY}"
    
    response = requests.get(url_with_key)
    response.raise_for_status()  # Raise an error for bad status codes
    
    json_response = response.json()
    data = json_response['response']['data']
    total_count = json_response['response'].get('total', len(data))
    
    # Convert total_count to int if it's a string
    try:
        total_count = int(total_count) if total_count is not None else len(data)
    except (ValueError, TypeError):
        total_count = len(data)
    
    df = pd.DataFrame(data)
    
    # Determine if there's more data to fetch
    # More data exists if we got a full batch AND haven't reached the total count
    has_more = len(data) == length and (offset + length) < total_count
    
    return df, total_count, has_more


def fetch_all_data() -> pd.DataFrame:
    """
    Fetch all retail sales data from EIA API (handles pagination).
    
    Returns:
        DataFrame with all electricity retail sales data
    """
    print("🔄 Fetching data from EIA API...")
    all_dataframes = []
    offset = 0
    length = DEFAULT_BATCH_SIZE
    total_count = None
    batch_num = 1
    
    while True:
        print(f"   Fetching batch {batch_num} (offset: {offset})...", end=" ")
        df_batch, total_count, has_more = fetch_data(offset=offset, length=length)
        all_dataframes.append(df_batch)
        print(f"✅ Got {len(df_batch)} records")
        
        if not has_more:
            break
        
        offset += length
        batch_num += 1
    
    # Combine all batches
    df_all = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ Total records fetched: {len(df_all):,}")
    print(f"   Expected total: {total_count:,}")
    
    return df_all

