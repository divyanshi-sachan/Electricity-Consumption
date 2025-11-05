# Validation Strategy

## Overview

This document outlines our comprehensive validation strategy for forecasting models and analysis results. Validation ensures model reliability, accuracy, and generalizability for electricity demand forecasting.

## Validation Philosophy

Our validation approach follows these principles:

1. **Rigorous**: Multiple validation methods to ensure reliability
2. **Comprehensive**: Validate at multiple levels (model, regional, temporal)
3. **Transparent**: Clear documentation of validation process and results
4. **Reproducible**: Validation procedures are reproducible and documented

## Model Validation Strategy

### 1. Train/Test Split

#### Approach

**Temporal Split**: 80% training, 20% testing
- **Rationale**: Preserves temporal order (critical for time series)
- **Method**: Chronological split (earlier data for training, later for testing)
- **Implementation**: Split at time point t, where t = 0.8 × total period

#### Why Temporal Split?

1. **Time Series Nature**: 
   - Time series data is ordered
   - Future predictions should not use future information
   - Random splits would violate temporal ordering

2. **Realistic Evaluation**:
   - Mimics real-world forecasting scenario
   - Tests model's ability to predict future from past
   - More realistic than random splits

3. **Prevents Data Leakage**:
   - Ensures no future information leaks into training
   - Maintains temporal integrity
   - Proper evaluation of forecast accuracy

#### Validation

- Training period: 2010-2020 (or 80% of available data)
- Test period: 2021-2024 (or 20% of available data)
- Ensures sufficient training data for model learning
- Sufficient test data for reliable evaluation

---

### 2. Cross-Validation

#### Time Series Cross-Validation

**Approach**: Walk-forward validation (expanding window)

**Method**:
1. Start with initial training window (e.g., 2010-2015)
2. Train model on training window
3. Forecast next period (e.g., 2016)
4. Evaluate forecast accuracy
5. Expand training window to include 2016
6. Repeat steps 2-5 for subsequent periods

**Benefits**:
- Multiple validation points
- Tests model stability over time
- Identifies performance degradation
- More robust evaluation than single split

#### Expanding vs Rolling Window

**Expanding Window** (Our Approach):
- Training window grows over time
- Uses all historical data available
- More data = better model performance
- Realistic for forecasting applications

**Rolling Window** (Alternative):
- Fixed-size training window
- Drops older data as new data arrives
- Tests model adaptability
- Useful for non-stationary data

**Our Choice**: Expanding window for better use of historical data

---

### 3. Out-of-Sample Testing

#### Purpose

Validate model performance on completely unseen data.

#### Approach

1. **Hold-out Set**: Reserve final period (e.g., last 6 months) as hold-out
2. **Model Training**: Train on all data except hold-out
3. **Forecast**: Generate forecasts for hold-out period
4. **Evaluation**: Compare forecasts with actual values
5. **No Retraining**: Model not retrained using hold-out data

#### Benefits

- Tests true predictive ability
- Prevents overfitting to test set
- Most realistic evaluation scenario
- Final validation before deployment

---

### 4. Backtesting

#### Purpose

Validate model performance on historical data to assess reliability.

#### Approach

1. **Historical Validation**: Test model on past periods
2. **Multiple Periods**: Validate across different time periods
3. **Performance Tracking**: Track accuracy over time
4. **Stability Assessment**: Assess model stability

#### Implementation

- Validate on multiple historical periods (2010-2015, 2015-2020, etc.)
- Compare performance across periods
- Identify periods of poor performance
- Understand model limitations

---

## Validation Metrics

### 1. Root Mean Squared Error (RMSE)

**Formula**: RMSE = √(Σ(actual - forecast)² / n)

**Purpose**: 
- Measures average forecast error magnitude
- Sensitive to outliers
- Same units as target variable

**Interpretation**:
- Lower RMSE = better forecast accuracy
- Expressed in units of electricity consumption (e.g., MWh)
- Useful for comparing models

**When to Use**:
- Primary metric for model comparison
- Point forecast evaluation
- Overall accuracy assessment

---

### 2. Mean Absolute Error (MAE)

**Formula**: MAE = Σ|actual - forecast| / n

**Purpose**:
- Measures average forecast error magnitude
- Less sensitive to outliers than RMSE
- Robust error metric

**Interpretation**:
- Lower MAE = better forecast accuracy
- Easier to interpret than RMSE
- Less affected by extreme errors

**When to Use**:
- Alternative to RMSE for robustness
- When outliers are a concern
- Error magnitude assessment

---

### 3. Mean Absolute Percentage Error (MAPE)

**Formula**: MAPE = (Σ|actual - forecast| / actual) / n × 100

**Purpose**:
- Measures relative forecast error
- Expressed as percentage
- Useful for comparing across regions

**Interpretation**:
- Lower MAPE = better forecast accuracy
- Percentage makes it scale-independent
- Easy to communicate to stakeholders

**When to Use**:
- Comparing forecasts across regions
- Communicating accuracy to non-technical audiences
- When relative error is important

**Limitations**:
- Can be undefined if actual = 0
- May be skewed by small actual values
- Use with caution for low-demand periods

---

### 4. Coverage Intervals

#### Purpose

Quantify forecast uncertainty and assess prediction intervals.

#### Approach

1. **Prediction Intervals**: Generate 80%, 90%, 95% prediction intervals
2. **Coverage Assessment**: Check if actual values fall within intervals
3. **Coverage Rate**: Calculate percentage of actuals within intervals

#### Validation

- **80% Interval**: Should contain ~80% of actual values
- **90% Interval**: Should contain ~90% of actual values
- **95% Interval**: Should contain ~95% of actual values

#### Interpretation

- **Over-coverage**: Intervals too wide (conservative)
- **Under-coverage**: Intervals too narrow (overconfident)
- **Good Coverage**: Intervals match expected coverage rates

---

### 5. Directional Accuracy

#### Purpose

Assess model's ability to predict direction of change (increase/decrease).

#### Approach

1. **Direction Prediction**: Predict if demand increases or decreases
2. **Actual Direction**: Compare with actual direction
3. **Accuracy**: Calculate percentage of correct direction predictions

#### Interpretation

- **High Accuracy**: Model captures trend direction well
- **Low Accuracy**: Model struggles with directional changes
- **Useful For**: Trend forecasting, growth predictions

---

## Validation Levels

### Level 1: Individual Model Validation

**Purpose**: Validate each model (ARIMA, Prophet, XGBoost) independently.

**Process**:
1. Train each model on training data
2. Forecast on test data
3. Calculate metrics (RMSE, MAE, MAPE)
4. Assess individual model performance

**Criteria**:
- RMSE < threshold (e.g., 10% of mean demand)
- MAPE < threshold (e.g., 15%)
- Consistent performance across regions

---

### Level 2: Ensemble Model Validation

**Purpose**: Validate ensemble model combining individual models.

**Process**:
1. Train individual models
2. Determine ensemble weights
3. Generate ensemble forecasts
4. Evaluate ensemble performance

**Criteria**:
- Ensemble RMSE < best individual model RMSE
- Ensemble MAPE < best individual model MAPE
- Improved robustness over individual models

---

### Level 3: Regional Validation

**Purpose**: Validate models across different regions.

**Process**:
1. Train models on each region's data
2. Validate on each region's test data
3. Compare performance across regions
4. Identify region-specific patterns

**Criteria**:
- Consistent performance across regions
- Regional variations in model performance
- Region-specific optimization if needed

---

### Level 4: Long-term Forecast Validation

**Purpose**: Validate long-term forecasts (5-10 years).

**Process**:
1. Train models on historical data
2. Generate 5-10 year forecasts
3. Compare with available historical data (if possible)
4. Assess forecast uncertainty

**Challenges**:
- Limited validation data for long horizons
- Uncertainty increases with horizon
- External factors become more important

**Mitigation**:
- Scenario analysis
- Sensitivity testing
- Uncertainty quantification
- Regular model updates

---

## Sensitivity Analysis

### 1. Parameter Sensitivity Testing

#### Purpose

Assess model sensitivity to parameter choices.

#### Approach

1. **Parameter Ranges**: Vary key parameters within reasonable ranges
2. **Performance Tracking**: Track performance for each parameter set
3. **Sensitivity Assessment**: Identify sensitive parameters
4. **Robustness Check**: Ensure model is robust to parameter choices

#### Parameters to Test

**ARIMA**:
- p, d, q values
- Seasonal parameters (P, D, Q)
- Seasonal period (m)

**Prophet**:
- Seasonality modes (additive, multiplicative)
- Changepoint prior scale
- Seasonality prior scale

**XGBoost**:
- Learning rate
- Max depth
- Number of estimators
- Regularization parameters

---

### 2. Scenario Analysis

#### Purpose

Validate forecasts under different future scenarios.

#### Scenarios

1. **Baseline Scenario**:
   - Current trends continue
   - No major disruptions
   - Standard growth rates

2. **High Growth Scenario**:
   - Accelerated economic growth
   - Higher demand growth
   - More aggressive projections

3. **Climate Change Scenario**:
   - Increased peak demand
   - More extreme weather events
   - Higher cooling/heating needs

4. **Renewable Transition Scenario**:
   - Accelerated renewable adoption
   - Changed consumption patterns
   - Grid integration challenges

#### Validation

- Generate forecasts for each scenario
- Compare scenario outcomes
- Assess model behavior under different assumptions
- Quantify uncertainty across scenarios

---

### 3. Robustness Testing

#### Purpose

Test model robustness to data perturbations.

#### Approach

1. **Data Perturbation**: Add noise, remove outliers, introduce missing values
2. **Performance Assessment**: Evaluate performance on perturbed data
3. **Robustness Measure**: Quantify sensitivity to perturbations

#### Tests

- **Noise Addition**: Add random noise to training data
- **Outlier Removal**: Remove outliers and retrain
- **Missing Data**: Introduce missing values
- **Feature Perturbation**: Vary external features

---

## Statistical Validation

### 1. Residual Analysis

#### Purpose

Assess model assumptions through residual analysis.

#### Approach

1. **Calculate Residuals**: actual - forecast
2. **Residual Distribution**: Check for normality
3. **Autocorrelation**: Test for residual autocorrelation
4. **Heteroscedasticity**: Test for constant variance

#### Tests

- **Normality Test**: Shapiro-Wilk test, Q-Q plots
- **Autocorrelation**: Ljung-Box test
- **Heteroscedasticity**: Breusch-Pagan test

#### Expected Results

- **Residuals**: Should be normally distributed (approximately)
- **No Autocorrelation**: Residuals should be independent
- **Constant Variance**: Residuals should have constant variance

---

### 2. Stationarity Tests

#### Purpose

Validate stationarity assumptions for ARIMA models.

#### Approach

1. **Augmented Dickey-Fuller (ADF) Test**: Test for unit root
2. **KPSS Test**: Test for stationarity
3. **Transformation**: Apply differencing if needed

#### Implementation

- Test raw data for stationarity
- Apply transformations if needed
- Retest after transformation
- Document transformation decisions

---

### 3. Autocorrelation Tests

#### Purpose

Assess autocorrelation in time series data.

#### Approach

1. **ACF/PACF Plots**: Visual inspection of autocorrelation
2. **Ljung-Box Test**: Statistical test for autocorrelation
3. **Model Selection**: Use ACF/PACF for model selection

---

## Validation Documentation

### Requirements

1. **Validation Report**: Document all validation results
2. **Metrics Summary**: Summary table of all metrics
3. **Visualizations**: Graphs showing forecast vs actual
4. **Sensitivity Results**: Parameter sensitivity analysis
5. **Scenario Analysis**: Results from scenario testing

### Validation Checklist

- [ ] Train/test split completed
- [ ] Cross-validation performed
- [ ] Out-of-sample testing done
- [ ] All metrics calculated (RMSE, MAE, MAPE)
- [ ] Coverage intervals assessed
- [ ] Sensitivity analysis completed
- [ ] Scenario analysis performed
- [ ] Residual analysis done
- [ ] Stationarity tests completed
- [ ] Validation report written

---

## Literature Support

### Validation Methodologies

- [Reference 1]: Time series cross-validation methods
- [Reference 2]: Validation metrics for electricity forecasting
- [Reference 3]: Sensitivity analysis in energy forecasting
- [Reference 4]: Scenario analysis for long-term forecasts

---

## Implementation

See `src/analysis/forecasting.py` for validation implementation:
- Train/test split: `prepare_data()` method
- Model validation: Individual model methods
- Metrics calculation: Built into each model method
- Ensemble validation: `ensemble_forecast()` method

---

## References

- See `../references/references.md` for full bibliography
- See `model_selection_rationale.md` for model details
- See `research_design.md` for overall methodology

