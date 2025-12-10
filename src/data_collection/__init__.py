"""
Data collection module for EIA API and other data sources
"""

from .eia_api import fetch_data, fetch_all_data

__all__ = ['fetch_data', 'fetch_all_data']

