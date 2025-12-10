"""
Forecasting functions for electricity consumption.
"""

from .forecast_generation import (
    generate_future_features, save_forecast_data
)

__all__ = ['generate_future_features', 'save_forecast_data']