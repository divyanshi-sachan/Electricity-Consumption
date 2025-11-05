"""
Forecasting Module
Predicts electricity demand for 5-10 years ahead
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingModel:
    """
    Class for forecasting electricity demand
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ForecastingModel
        
        Args:
            df: Input DataFrame with time series data
        """
        self.df = df.copy()
        self.models = {}
        self.forecasts = {}
    
    def prepare_data(self,
                    value_column: str = 'value',
                    date_column: str = 'period',
                    train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for forecasting
        
        Args:
            value_column: Column with values to forecast
            date_column: Column with dates
            train_size: Proportion of data for training
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if date_column not in self.df.columns or value_column not in self.df.columns:
            logger.error("Required columns not found")
            return pd.DataFrame(), pd.DataFrame()
        
        df_prep = self.df[[date_column, value_column]].copy()
        df_prep[date_column] = pd.to_datetime(df_prep[date_column])
        df_prep = df_prep.sort_values(date_column).reset_index(drop=True)
        df_prep = df_prep.set_index(date_column)
        
        # Split train/test
        split_idx = int(len(df_prep) * train_size)
        train_data = df_prep.iloc[:split_idx]
        test_data = df_prep.iloc[split_idx:]
        
        logger.info(f"Data prepared: {len(train_data)} training, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def arima_forecast(self,
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      value_column: str = 'value',
                      forecast_horizon: int = 365) -> Dict:
        """
        Forecast using ARIMA model
        
        Args:
            train_data: Training data
            test_data: Test data
            value_column: Column to forecast
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from pmdarima import auto_arima
        except ImportError:
            logger.error("ARIMA libraries not installed. Install statsmodels and pmdarima")
            return {}
        
        try:
            # Auto ARIMA to find best parameters
            model = auto_arima(
                train_data[value_column],
                seasonal=True,
                m=12,  # Monthly seasonality
                stepwise=True,
                suppress_warnings=True
            )
            
            # Fit model
            fitted_model = model.fit(train_data[value_column])
            
            # Forecast
            forecast = fitted_model.predict(n_periods=len(test_data))
            forecast_index = test_data.index
            
            # Long-term forecast (5-10 years)
            long_term_forecast = fitted_model.predict(n_periods=forecast_horizon)
            long_term_dates = pd.date_range(
                start=train_data.index[-1],
                periods=forecast_horizon + 1,
                freq='D'
            )[1:]
            
            # Calculate metrics
            mse = mean_squared_error(test_data[value_column], forecast)
            mae = mean_absolute_error(test_data[value_column], forecast)
            rmse = np.sqrt(mse)
            
            results = {
                'model': 'ARIMA',
                'forecast': pd.Series(forecast, index=forecast_index),
                'long_term_forecast': pd.Series(long_term_forecast, index=long_term_dates),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': np.mean(np.abs((test_data[value_column] - forecast) / test_data[value_column])) * 100
                },
                'model_params': model.get_params()
            }
            
            self.models['arima'] = fitted_model
            self.forecasts['arima'] = results
            
            logger.info(f"ARIMA forecast complete. RMSE: {rmse:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return {}
    
    def prophet_forecast(self,
                        train_data: pd.DataFrame,
                        test_data: pd.DataFrame,
                        value_column: str = 'value',
                        forecast_horizon: int = 365) -> Dict:
        """
        Forecast using Prophet model
        
        Args:
            train_data: Training data
            test_data: Test data
            value_column: Column to forecast
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Install prophet")
            return {}
        
        try:
            # Prepare data for Prophet
            prophet_df = train_data.reset_index()
            prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: 'ds', value_column: 'y'})
            
            # Fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Test forecast
            test_forecast = forecast.iloc[-len(test_data):]['yhat'].values
            
            # Long-term forecast (5-10 years)
            long_future = model.make_future_dataframe(periods=forecast_horizon)
            long_forecast = model.predict(long_future)
            long_term = long_forecast.iloc[-forecast_horizon:]['yhat'].values
            long_term_dates = long_forecast.iloc[-forecast_horizon:]['ds'].values
            
            # Calculate metrics
            mse = mean_squared_error(test_data[value_column], test_forecast)
            mae = mean_absolute_error(test_data[value_column], test_forecast)
            rmse = np.sqrt(mse)
            
            results = {
                'model': 'Prophet',
                'forecast': pd.Series(test_forecast, index=test_data.index),
                'long_term_forecast': pd.Series(long_term, index=pd.to_datetime(long_term_dates)),
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': np.mean(np.abs((test_data[value_column] - test_forecast) / test_data[value_column])) * 100
                },
                'model': model
            }
            
            self.models['prophet'] = model
            self.forecasts['prophet'] = results
            
            logger.info(f"Prophet forecast complete. RMSE: {rmse:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            return {}
    
    def ensemble_forecast(self,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         value_column: str = 'value',
                         forecast_horizon: int = 365) -> Dict:
        """
        Ensemble forecast combining multiple models
        
        Args:
            train_data: Training data
            test_data: Test data
            value_column: Column to forecast
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with ensemble forecast results
        """
        forecasts = []
        weights = []
        
        # Get ARIMA forecast
        arima_results = self.arima_forecast(train_data, test_data, value_column, forecast_horizon)
        if arima_results:
            forecasts.append(arima_results['forecast'])
            weights.append(1.0 / arima_results['metrics']['rmse'])
        
        # Get Prophet forecast
        prophet_results = self.prophet_forecast(train_data, test_data, value_column, forecast_horizon)
        if prophet_results:
            forecasts.append(prophet_results['forecast'])
            weights.append(1.0 / prophet_results['metrics']['rmse'])
        
        if not forecasts:
            logger.error("No forecasts available for ensemble")
            return {}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_forecast = sum(f * w for f, w in zip(forecasts, weights))
        
        # Long-term forecast
        long_term_forecasts = []
        if arima_results and 'long_term_forecast' in arima_results:
            long_term_forecasts.append(arima_results['long_term_forecast'])
        if prophet_results and 'long_term_forecast' in prophet_results:
            long_term_forecasts.append(prophet_results['long_term_forecast'])
        
        if long_term_forecasts:
            ensemble_long_term = sum(f * w for f, w in zip(long_term_forecasts, weights))
        else:
            ensemble_long_term = None
        
        # Calculate metrics
        mse = mean_squared_error(test_data[value_column], ensemble_forecast)
        mae = mean_absolute_error(test_data[value_column], ensemble_forecast)
        rmse = np.sqrt(mse)
        
        results = {
            'model': 'Ensemble',
            'forecast': ensemble_forecast,
            'long_term_forecast': ensemble_long_term,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': np.mean(np.abs((test_data[value_column] - ensemble_forecast) / test_data[value_column])) * 100
            },
            'weights': dict(zip(['arima', 'prophet'], weights))
        }
        
        self.forecasts['ensemble'] = results
        
        logger.info(f"Ensemble forecast complete. RMSE: {rmse:.2f}")
        
        return results
    
    def forecast_by_region(self,
                          region: str,
                          value_column: str = 'value',
                          date_column: str = 'period',
                          forecast_horizon: int = 365) -> Dict:
        """
        Forecast electricity demand for a specific region
        
        Args:
            region: Region identifier
            value_column: Column with consumption values
            date_column: Column with dates
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary with regional forecast results
        """
        if 'region' not in self.df.columns:
            logger.error("Region column not found")
            return {}
        
        region_data = self.df[self.df['region'] == region].copy()
        
        if len(region_data) == 0:
            logger.error(f"No data found for region: {region}")
            return {}
        
        # Prepare data
        train_data, test_data = self.prepare_data(value_column, date_column)
        
        # Generate ensemble forecast
        forecast_results = self.ensemble_forecast(train_data, test_data, value_column, forecast_horizon)
        
        forecast_results['region'] = region
        
        return forecast_results

