# Mining Electricity Consumption Data for Resource Allocation and Supply Chain Enhancement

## Project Overview

This project provides a comprehensive data-driven analysis of electricity consumption patterns across the United States. The system analyzes historical consumption data, generates long-term forecasts (5-10 years), identifies high-risk regions, and provides actionable recommendations for infrastructure investment and supply chain optimization.

### Key Objectives

1. **Pattern Identification**: Analyze temporal, regional, and sectoral variations in electricity consumption
2. **Demand Forecasting**: Generate accurate 5-10 year forecasts using advanced machine learning models
3. **Risk Assessment**: Identify regions at risk of under-supply based on demand growth projections
4. **Strategic Recommendations**: Provide data-driven recommendations for infrastructure investment

## Team Members

- **Divyanshi Sachan**
- **Soniya Malviya**
- **Aryan Soni**

## Research Questions

1. **Temporal and Regional Variations**: How does electricity usage vary by region, time of day, and season?
2. **Demand Spikes**: Are there recurring demand spikes that could stress the grid (e.g., during summer heatwaves or festivals)?
3. **Sector Analysis**: How do residential vs. commercial vs. industrial users differ in consumption?
4. **Forecasting**: Can we forecast electricity demand for the next 5–10 years in specific regions?
5. **Risk Assessment**: Which areas are at risk of under-supply if current trends continue?

## Dataset

**Primary Data Source**: [EIA Open Data API](https://www.eia.gov/opendata/browser/electricity)

- **Dataset**: Electricity retail sales data from U.S. Energy Information Administration
- **Frequency**: Monthly data
- **Coverage**: All U.S. states
- **Metrics**: Customers, price, revenue, sales (MWh)
- **Time Period**: Historical data from 2010 onwards

## Project Architecture

### System Design

The project follows a modular, pipeline-based architecture:

```
Data Collection → Preprocessing → EDA → Feature Engineering → 
Modeling → Validation → Forecasting → Risk Analysis → Visualization
```

### Core Components

1. **Data Collection Module** (`src/data_collection/`)
   - EIA API integration with pagination support
   - Automated data fetching and storage
   - Metadata collection and validation

2. **Data Processing Module** (`src/data_processing/`)
   - Data cleaning and validation
   - Missing value imputation
   - Feature engineering (temporal, lag, rolling statistics)
   - State-level data quality filtering

3. **Modeling Module** (`src/models/`)
   - **Standard ML Models**: Linear Regression, Random Forest, XGBoost
   - **Time Series Models**: SARIMA, Prophet
   - **Hybrid Model**: Per-state SARIMA + XGBoost ensemble

4. **Forecasting Module** (`src/forecasting/`)
   - Multi-horizon forecasting (up to 120 months)
   - Recursive feature generation for future predictions
   - State-level forecast aggregation

5. **Analysis Module** (`src/analysis/`)
   - Exploratory Data Analysis (EDA)
   - Regional insights and trend analysis
   - Risk assessment and gap analysis
   - Forecast validation

6. **Dashboard Module** (`src/dashboard/`)
   - Interactive visualizations (Plotly)
   - Geographical heatmaps
   - Risk dashboards
   - State comparison charts

## Project Structure

```
Electricity-Consumption/
├── data/
│   ├── raw/                          # Raw data from EIA API
│   │   └── eia_retail_sales_raw_*.csv
│   ├── processed/                    # Cleaned and processed data
│   │   ├── df_model_*.csv           # Preprocessed modeling data
│   │   └── hybrid_forecasts_*.csv   # Generated forecasts
│   └── external/                    # Additional data sources
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_collection.ipynb     # Data collection workflow
│   ├── 02_preprocessing.ipynb        # Data preprocessing
│   ├── 03_eda.ipynb                  # Exploratory data analysis
│   ├── 04_ml_modeling.ipynb         # Machine learning models
│   ├── 05_forecasting.ipynb          # Forecast generation
│   ├── 06_dashboard.ipynb            # Dashboard creation
│   ├── 07_comprehensive_analysis.ipynb  # Comprehensive analysis
│   └── requirements.txt              # Notebook dependencies
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── config.py                     # Configuration and constants
│   │
│   ├── data_collection/              # Data collection module
│   │   ├── __init__.py
│   │   ├── eia_api.py               # EIA API client
│   │   └── ingest_eia_data.py       # Multi-route data ingestion
│   │
│   ├── data_processing/              # Data preprocessing module
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Complete preprocessing pipeline
│   │
│   ├── models/                       # Model training module
│   │   ├── __init__.py
│   │   ├── training.py              # Standard ML models (LR, RF, XGBoost)
│   │   ├── time_series.py           # Time series models (SARIMA, Prophet)
│   │   └── hybrid.py                # Hybrid SARIMA + XGBoost model
│   │
│   ├── forecasting/                  # Forecast generation module
│   │   ├── __init__.py
│   │   └── forecast_generation.py   # Future forecast generation
│   │
│   ├── analysis/                     # Analysis module
│   │   ├── __init__.py
│   │   ├── eda.py                   # Exploratory data analysis
│   │   ├── regional_insights.py     # Regional trend analysis
│   │   ├── risk_assessment.py       # Risk analysis and gap calculation
│   │   └── forecast_validation.py   # Model validation metrics
│   │
│   ├── dashboard/                    # Visualization module
│   │   ├── __init__.py
│   │   ├── visualizations.py       # Interactive visualizations
│   │   └── dashboard_utils.py      # Dashboard utilities
│   │
│   └── utils/                        # Utility functions
│       └── __init__.py
│
├── research/                         # Research documentation
│   ├── README.md                     # Research overview
│   ├── hypotheses/                   # Research hypotheses
│   ├── methodology/                  # Research methodology
│   └── research_questions/           # Research questions
│
├── docs/                             # General documentation
│   ├── methodology.md                # Detailed methodology
│   ├── analysis_philosophy.md       # Analysis principles
│   └── work_planning.md             # Project planning
│
├── models/                           # Saved model files
├── activate_venv.sh                  # Virtual environment activation script
└── README.md                         # This file
```

## Detailed Module Documentation

### 1. Configuration (`src/config.py`)

Central configuration file containing:
- **API Settings**: EIA API key and endpoints
- **Data Paths**: Raw and processed data directories
- **Model Parameters**: 
  - XGBoost hyperparameters (optimized to prevent overfitting)
  - SARIMA configurations for monthly data
  - Train/test split ratio (80/20)
  - Forecast periods (120 months = 10 years)

### 2. Data Collection (`src/data_collection/`)

#### `eia_api.py`
- **`fetch_data(offset, length)`**: Fetches a batch of data from EIA API
- **`fetch_all_data()`**: Handles pagination to fetch all available data
- Features: Automatic pagination, error handling, progress tracking

#### `ingest_eia_data.py`
- **`run_ingestion()`**: Multi-route data ingestion pipeline
- Fetches metadata and data for all electricity routes
- Saves data in both CSV and Parquet formats
- Includes logging functionality

### 3. Data Processing (`src/data_processing/preprocessing.py`)

Complete preprocessing pipeline with the following functions:

#### Data Loading
- **`load_raw_data(file_path)`**: Loads raw data (auto-finds most recent if not specified)
- **`load_processed_data(file_path)`**: Loads processed data

#### Data Cleaning
- **`convert_period_to_datetime(df)`**: Converts period column to datetime
- **`filter_valid_states(df)`**: Filters to valid 2-letter state codes
- **`filter_sector(df, sector)`**: Filters by sector (default: 'ALL')
- **`convert_numeric_columns(df)`**: Converts columns to numeric type
- **`handle_missing_values(df)`**: Forward fill and mean imputation

#### Feature Engineering
- **`engineer_features(df)`**: Creates time series features:
  - Temporal: year, month, season
  - Lag features: lag_1_month, lag_12_month
  - Rolling statistics: rolling_mean_3, rolling_mean_12, rolling_std_12
  - **Note**: Avoids leakage features (no revenue_per_customer, sales_per_customer, price)

#### Data Quality
- **`validate_state_data_quality(df, state_id)`**: Validates data quality per state
- **`filter_quality_states(df)`**: Filters out states with poor data quality
  - Minimum 24 months of data required
  - Maximum 30% missing values allowed
  - Excludes states with >50% zero values

#### Complete Pipeline
- **`preprocess_pipeline(file_path, sector, encode_stateid, filter_quality)`**: 
  - Runs complete preprocessing pipeline
  - Returns: df_model, X, y, feature_cols, le_state

### 4. Models (`src/models/`)

#### Standard ML Models (`training.py`)
- **Linear Regression**: Baseline model
- **Random Forest**: Tree-based ensemble
- **XGBoost**: Gradient boosting (optimized hyperparameters)
- **Evaluation Metrics**: RMSE, MAE, MAPE, R², NRMSE
- **Time-based Split**: 80/20 temporal split

#### Time Series Models (`time_series.py`)
- **SARIMA**: Seasonal ARIMA with multiple configurations
  - Tries multiple (p,d,q)(P,D,Q,s) configurations
  - Selects best based on AIC
- **Prophet**: Facebook's Prophet for trend and seasonality
- **Time Series Preparation**: Handles state-level aggregation

#### Hybrid Model (`hybrid.py`)
**Per-State Hybrid Architecture**:
1. **SARIMA Component**: Captures trend, seasonality, and autoregression per state
2. **XGBoost Residual Model**: Learns nonlinear patterns in SARIMA residuals per state
3. **Final Prediction**: SARIMA forecast + XGBoost residual forecast

**Key Features**:
- Per-state model training (accounts for regional differences)
- Time-based splitting for both components
- Recursive feature generation for future forecasts
- Handles states with insufficient data gracefully

### 5. Forecasting (`src/forecasting/forecast_generation.py`)

- **`generate_future_features(df_model, state_id, forecast_dates)`**: 
  - Generates features for future periods
  - Uses last known values and recursive predictions
- **`save_forecast_data(forecast_df, model_name)`**: 
  - Saves forecasts with timestamps

### 6. Analysis (`src/analysis/`)

#### EDA (`eda.py`)
- Missing value analysis
- Statistical summaries
- Negative value detection
- State ID distribution analysis
- Visualization functions (sales by year, month, price vs sales)

#### Regional Insights (`regional_insights.py`)
- **`calculate_state_growth_metrics()`**: Calculates growth metrics per state
  - Historical mean/max
  - Forecast start/end averages
  - Total growth percentage
  - Volatility (coefficient of variation)
  - Risk level classification
- **`identify_high_growth_states()`**: States with >15% growth
- **`identify_declining_states()`**: States with <-5% growth
- **`identify_volatile_states()`**: States with >20% volatility
- **`generate_regional_insights_report()`**: Comprehensive regional analysis

#### Risk Assessment (`risk_assessment.py`)
- **`calculate_demand_supply_gap()`**: Calculates demand-supply gaps
  - Filters to years with actual data (handles zero forecasts)
  - Aggregates by state
  - Calculates growth percentages
  - Classifies risk levels
- **`identify_high_risk_regions()`**: Merges gap and growth data
- **`generate_policy_recommendations()`**: Generates recommendations by category:
  - High growth states
  - Declining states
  - Volatile states
  - Critical risk regions

#### Forecast Validation (`forecast_validation.py`)
- **`calculate_forecast_metrics()`**: Comprehensive metrics (RMSE, MAE, MAPE, R², NRMSE)
- **`plot_training_validation_curve()`**: Training vs validation curves
- **`plot_residuals()`**: Residual analysis plots
- **`plot_feature_importance()`**: Feature importance visualization
- **`generate_validation_report()`**: Comprehensive validation report with all plots

### 7. Dashboard (`src/dashboard/`)

#### Visualizations (`visualizations.py`)
- **`create_demand_heatmap()`**: Interactive heatmap showing current vs forecasted demand
  - Handles zero forecasts gracefully
  - Auto-detects value columns
  - Filters to years with actual data
  - Color-coded by risk level
- **`create_state_comparison_chart()`**: Multi-state comparison
- **`create_risk_dashboard()`**: Comprehensive risk dashboard with subplots

#### Dashboard Utils (`dashboard_utils.py`)
- **`load_forecast_data(model_name)`**: Loads forecast data
- **`get_forecast_for_state()`**: Gets forecast for specific state
- **`prepare_dashboard_data()`**: Combines historical and forecast data

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager
- EIA API key ([Get one here](https://www.eia.gov/opendata/register.php))

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Electricity-Consumption
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
   Or use the activation script:
   ```bash
   bash activate_venv.sh
   ```

3. **Install dependencies**
   ```bash
   # Install from notebooks requirements (basic dependencies)
   pip install -r notebooks/requirements.txt
   
   # Install additional ML dependencies
   pip install scikit-learn xgboost statsmodels prophet plotly seaborn matplotlib
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file in project root
   echo "EIA_API_KEY=your_api_key_here" > .env
   ```
   
   Or manually create `.env` file:
   ```
   EIA_API_KEY=your_api_key_here
   ```

5. **Run data collection**
   ```bash
   # Option 1: Use the main EIA API client
   python src/data_collection/eia_api.py
   
   # Option 2: Use the multi-route ingestion pipeline
   python src/data_collection/ingest_eia_data.py
   ```

6. **Start Jupyter notebooks**
   ```bash
   jupyter notebook
   ```
   
   Then open notebooks in order:
   - `01_data_collection.ipynb` - Collect data
   - `02_preprocessing.ipynb` - Preprocess data
   - `03_eda.ipynb` - Exploratory analysis
   - `04_ml_modeling.ipynb` - Train models
   - `05_forecasting.ipynb` - Generate forecasts
   - `06_dashboard.ipynb` - Create visualizations
   - `07_comprehensive_analysis.ipynb` - Comprehensive analysis

## Usage Examples

### 1. Data Collection

```python
from src.data_collection.eia_api import fetch_all_data

# Fetch all retail sales data
df_raw = fetch_all_data()
print(f"Fetched {len(df_raw):,} records")
```

### 2. Data Preprocessing

```python
from src.data_processing.preprocessing import preprocess_pipeline

# Run complete preprocessing pipeline
df_model, X, y, feature_cols, le_state = preprocess_pipeline(
    sector='ALL',
    encode_stateid=True,
    filter_quality=True
)

print(f"Processed {len(df_model):,} rows")
print(f"Features: {feature_cols}")
```

### 3. Model Training

```python
from src.models.training import train_xgboost, time_based_split
from src.models.hybrid import train_hybrid_per_state

# Standard XGBoost
X_train, X_test, y_train, y_test = time_based_split(df_model, X, y)
model, train_metrics, test_metrics = train_xgboost(X_train, y_train, X_test, y_test)

# Hybrid model (per-state)
state_models, state_evaluations, test_preds, test_actuals = train_hybrid_per_state(
    df_model.sort_values('period'), X, feature_cols
)
```

### 4. Forecasting

```python
from src.models.hybrid import generate_hybrid_forecast
from datetime import datetime

# Generate 10-year forecast
forecast_start = pd.Timestamp('2025-01-01')
forecast_df = generate_hybrid_forecast(
    state_hybrid_models,
    df_model.sort_values('period'),
    forecast_start,
    forecast_periods=120  # 10 years
)
```

### 5. Regional Analysis

```python
from src.analysis.regional_insights import generate_regional_insights_report

# Generate comprehensive regional insights
insights = generate_regional_insights_report(forecast_df, df_model)

print(f"High growth states: {len(insights['high_growth_states'])}")
print(f"Declining states: {len(insights['declining_states'])}")
print(f"Volatile states: {len(insights['volatile_states'])}")
```

### 6. Risk Assessment

```python
from src.analysis.risk_assessment import calculate_demand_supply_gap, generate_policy_recommendations

# Calculate demand-supply gaps
gap_df = calculate_demand_supply_gap(forecast_df)

# Generate policy recommendations
recommendations = generate_policy_recommendations(gap_df, insights['state_metrics'])
```

### 7. Visualization

```python
from src.dashboard.visualizations import create_demand_heatmap

# Create interactive heatmap
fig = create_demand_heatmap(
    forecast_df,
    df_model,
    year=2030,
    interactive=True
)
fig.show()
```

## Model Details

### Hybrid Model Architecture

The hybrid model combines the strengths of SARIMA and XGBoost:

1. **SARIMA Component**:
   - Captures linear trends, seasonality, and autoregressive patterns
   - Trained per-state to account for regional differences
   - Multiple configurations tested, best selected by AIC

2. **XGBoost Residual Model**:
   - Learns nonlinear patterns in SARIMA residuals
   - Uses time series features (lags, rolling statistics)
   - Also trained per-state

3. **Final Prediction**:
   - `forecast = SARIMA_forecast + XGBoost_residual_forecast`
   - Clipped to non-negative values

### Model Hyperparameters

**XGBoost** (optimized to prevent overfitting):
- `n_estimators`: 500
- `max_depth`: 3 (reduced from 6)
- `learning_rate`: 0.05 (reduced from 0.1)
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `reg_alpha`: 5 (L1 regularization)
- `reg_lambda`: 10 (L2 regularization)

**SARIMA Configurations** (tested for monthly data):
- (1,1,1) x (1,1,1,12)
- (2,1,2) x (1,1,1,12)
- (1,1,0) x (1,1,0,12)
- (0,1,1) x (0,1,1,12)

### Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **NRMSE**: Normalized RMSE (as percentage)

## Data Pipeline

### 1. Raw Data Collection
- Fetches from EIA API
- Handles pagination automatically
- Saves with timestamp: `eia_retail_sales_raw_YYYYMMDD_HHMMSS.csv`

### 2. Preprocessing
- Converts period to datetime
- Filters valid states (2-letter codes)
- Handles missing values (forward fill + mean imputation)
- Engineers features (temporal, lag, rolling statistics)
- Filters states with poor data quality
- Saves processed data: `df_model_YYYYMMDD_HHMMSS.csv`

### 3. Model Training
- Time-based train/test split (80/20)
- Per-state model training for hybrid model
- Model evaluation and validation

### 4. Forecasting
- Generates 120-month forecasts (10 years)
- Recursive feature generation
- Saves forecasts: `hybrid_forecasts_YYYYMMDD_HHMMSS.csv`

### 5. Analysis
- Regional insights calculation
- Risk assessment
- Policy recommendations

## Key Features

### 1. Data Quality Assurance
- State-level data quality validation
- Minimum data requirements (24 months)
- Missing value threshold (30%)
- Zero value detection (>50% zeros excluded)

### 2. Leakage Prevention
- No leakage features in modeling
- Removed: `revenue_per_customer`, `sales_per_customer`, `price`
- Uses only legitimate time series features

### 3. Per-State Modeling
- Accounts for regional differences
- State-specific SARIMA models
- State-specific XGBoost residual models

### 4. Robust Forecasting
- Handles states with insufficient data
- Recursive feature generation
- Zero forecast detection and handling

### 5. Comprehensive Analysis
- Growth metrics calculation
- Volatility analysis
- Risk level classification
- Policy recommendations

## Dependencies

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing

### Data Collection
- `requests>=2.31.0` - HTTP requests
- `python-dotenv>=1.0.0` - Environment variables

### Machine Learning
- `scikit-learn` - Standard ML models
- `xgboost` - Gradient boosting
- `statsmodels` - SARIMA models
- `prophet` - Facebook Prophet (optional)

### Visualization
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations
- `plotly` - Interactive visualizations

### Utilities
- `jupyter>=1.0.0` - Jupyter notebooks
- `tqdm>=4.65.0` - Progress bars

## Methodology

See `docs/methodology.md` for detailed methodology covering:
- Data collection strategy
- Preprocessing approach
- Feature engineering
- Model selection rationale
- Validation strategy
- Risk analysis methodology

## Analysis Philosophy

See `docs/analysis_philosophy.md` for core principles:
- Data-driven decision making
- Multi-perspective analysis
- Causal thinking
- Uncertainty quantification
- Actionable insights

## Research Documentation

See `research/README.md` for comprehensive research documentation including:
- **Literature Review**: 20-30+ academic papers and reports
- **Hypotheses**: Novel, well-founded research hypotheses
- **Methodology**: Theoretical foundation and research design
- **Innovation**: Novel contributions and insights
- **References**: Complete bibliography

**Note**: Research documentation has maximum weightage (40 marks combined) in evaluation criteria.

## Key Deliverables

- ✅ **Interactive Geographical Heatmap**: Showing demand/supply by region
- ✅ **Predictive Analysis**: 5-10 year forecasts using hybrid model
- ✅ **Risk Assessment**: Identifying under-supply areas
- ✅ **Root Cause Analysis**: Detailed analysis of risk factors
- ✅ **Policy Recommendations**: Actionable infrastructure investment recommendations
- ✅ **Comprehensive Documentation**: Complete documentation of findings

## Timeline

- **Week 1**: Data collection and initial exploration
- **Week 2**: Data preprocessing and exploratory analysis
- **Week 3**: Predictive modeling and risk analysis
- **Week 4**: Visualization, documentation, and finalization

## File Naming Conventions

- **Raw Data**: `eia_retail_sales_raw_YYYYMMDD_HHMMSS.csv`
- **Processed Data**: `df_model_YYYYMMDD_HHMMSS.csv`
- **Forecasts**: `{model}_forecasts_YYYYMMDD_HHMMSS.csv`
  - Example: `hybrid_forecasts_20251204_100708.csv`

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `.env` file exists with `EIA_API_KEY`
   - Verify API key is valid at https://www.eia.gov/opendata/

2. **No Data Found**
   - Check if raw data files exist in `data/raw/`
   - Run data collection first

3. **Model Training Fails**
   - Check data quality (minimum 24 months per state)
   - Verify feature columns exist
   - Check for missing values

4. **Forecast Generation Issues**
   - Ensure models are trained first
   - Check forecast start date is after last historical date
   - Verify state models exist for all states in forecast

5. **Visualization Errors**
   - Install plotly: `pip install plotly`
   - Check for zero forecasts (may indicate data issues)
   - Verify year exists in forecast data

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Commit with clear messages
4. Push and create a pull request

### Code Style
- Follow PEP 8 style guide
- Use type hints where possible
- Document functions with docstrings
- Add comments for complex logic

## License

[Add license information here]

## Acknowledgments

- U.S. Energy Information Administration (EIA) for providing open data
- Open source community for excellent tools and libraries

## Contact

For questions or issues, please contact the project team members.

---

**Last Updated**: December 2024
