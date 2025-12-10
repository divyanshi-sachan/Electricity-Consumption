"""
Configuration file for Electricity Consumption Analysis Pipeline.

Centralizes all configuration settings including API keys, file paths, and model parameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# API Configuration
EIA_API_KEY = os.getenv('EIA_API_KEY')
if not EIA_API_KEY:
    raise ValueError('EIA_API_KEY environment variable is required. Please set it in your .env file.')

BASE_URL = "https://api.eia.gov/v2"

# EIA API Endpoints
EIA_RETAIL_SALES_URL = (
    "https://api.eia.gov/v2/electricity/retail-sales/data/"
    "?frequency=monthly"
    "&data[0]=customers"
    "&data[1]=price"
    "&data[2]=revenue"
    "&data[3]=sales"
    "&sort[0][column]=period"
    "&sort[0][direction]=desc"
    "&offset={}&length={}"
)

# Data fetching parameters
DEFAULT_BATCH_SIZE = 5000
MAX_BATCH_SIZE = 5000

# Model parameters
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test
FORECAST_PERIODS = 120  # 10 years (120 months)

# Model hyperparameters
# ⚠️ UPDATED to reduce overfitting
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,  # Reduced from 6 to prevent overfitting
    'learning_rate': 0.05,  # Reduced from 0.1 for better generalization
    'subsample': 0.7,  # Add subsampling
    'colsample_bytree': 0.7,  # Add column sampling
    'reg_alpha': 5,  # L1 regularization
    'reg_lambda': 10,  # L2 regularization
    'random_state': 42,
    'n_jobs': -1
}

# SARIMA configurations to try
SARIMA_CONFIGS = [
    ((1, 1, 1), (1, 1, 1, 12)),
    ((2, 1, 2), (1, 1, 1, 12)),
    ((1, 1, 0), (1, 1, 0, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
]

# File naming patterns
RAW_DATA_PATTERN = "eia_retail_sales_raw_{timestamp}.csv"
PROCESSED_DATA_PATTERN = "df_model_{timestamp}.csv"
FORECAST_DATA_PATTERN = "{model}_forecasts_{timestamp}.csv"

