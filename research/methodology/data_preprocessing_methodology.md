# Data Preprocessing Methodology

## Overview

This document comprehensively documents our data preprocessing methodology, including data quality assessment, missing value handling, outlier detection and treatment, feature engineering, and data validation. All preprocessing decisions are justified with rationale and literature support.

## Purpose

The goal of data preprocessing is to transform raw electricity consumption data into clean, consistent, and analysis-ready data that:
- Preserves important information
- Handles data quality issues appropriately
- Enables accurate forecasting and analysis
- Maintains reproducibility

## Preprocessing Pipeline

### Pipeline Stages

1. **Data Quality Assessment** → Initial assessment of data quality
2. **Missing Value Handling** → Address missing data appropriately
3. **Outlier Detection** → Identify and investigate outliers
4. **Outlier Treatment** → Handle outliers based on analysis
5. **Data Cleaning** → Remove duplicates, standardize formats
6. **Feature Engineering** → Create temporal, lag, and derived features
7. **Data Validation** → Final quality checks and validation

---

## Data Quality Assessment

### Initial Assessment

**Purpose**: Understand data quality before preprocessing.

**Metrics Collected**:
- Total rows and columns
- Missing value counts and percentages
- Duplicate row count
- Data types for each column
- Numeric vs categorical column identification
- Statistical summaries for numeric columns

**Implementation**: See `src/data_processing/cleaning.py` → `assess_data_quality()`

**Rationale**:
- Identifies data quality issues early
- Informs preprocessing decisions
- Provides baseline for validation
- Documents data characteristics

**Output**: Quality report with metrics for all columns

---

## Missing Value Handling

### Strategy Selection

#### Forward Fill (Primary Strategy for Time Series)

**Strategy**: Forward fill (ffill) - use previous value to fill missing values

**Justification**:
1. **Temporal Order Preservation**: 
   - Time series data is ordered
   - Forward fill preserves temporal order
   - Assumes demand continues from previous period

2. **Literature Support**:
   - Standard approach for time series missing values
   - Preserves autocorrelation structure
   - Maintains temporal relationships

3. **Domain Knowledge**:
   - Electricity demand is relatively stable day-to-day
   - Missing values likely due to data collection issues
   - Previous period is reasonable estimate

**When to Use**:
- Time series data (default)
- Missing values are temporary (not systematic)
- Demand is relatively stable

**Limitations**:
- May propagate errors if previous value is incorrect
- May underestimate changes during missing period
- Assumes demand remains constant

**Mitigation**:
- Validate imputed values
- Compare with interpolation where appropriate
- Flag imputed values for sensitivity analysis

#### Backward Fill

**Strategy**: Backward fill (bfill) - use next value to fill missing values

**Justification**:
- Alternative to forward fill
- Useful when future values are more reliable
- May be appropriate for recent missing values

**When to Use**:
- Recent missing values
- When future values are more reliable
- Complementary to forward fill

#### Interpolation

**Strategy**: Linear interpolation between known values

**Justification**:
- Smooth transition between values
- May capture gradual changes better
- Useful for short gaps

**When to Use**:
- Short gaps in data
- When demand changes gradually
- When smooth transitions are expected

**Limitations**:
- May overestimate for sudden changes
- Less appropriate for long gaps
- May introduce artificial patterns

#### Mean/Median Imputation

**Strategy**: Fill missing values with mean or median

**Justification**:
- Maintains distribution characteristics
- Simple and fast
- Useful for non-time series data

**When to Use**:
- Non-time series numeric columns
- When temporal order is not critical
- As baseline for comparison

**Limitations**:
- Ignores temporal patterns
- May reduce variance
- Not appropriate for time series

#### Drop Missing Values

**Strategy**: Remove rows with missing values

**Justification**:
- Preserves data integrity
- Avoids imputation errors
- Simple approach

**When to Use**:
- Large percentage of missing values
- When missing values are systematic
- When imputation is not appropriate

**Limitations**:
- Loss of data
- May introduce bias if missing is systematic
- Reduces sample size

### Threshold Criteria

**Missing Threshold**: 50% (columns with >50% missing flagged)

**Rationale**:
- Columns with >50% missing have limited information
- Imputation for >50% missing may be unreliable
- May need investigation or exclusion

**Decision Rules**:
1. **<10% Missing**: Use forward fill (time series) or mean/median (non-time series)
2. **10-50% Missing**: Use forward fill with validation
3. **>50% Missing**: Flag for investigation, consider dropping
4. **Systematic Missing**: Investigate cause, may need different approach

**Documentation**: All decisions documented with rationale

### Validation

**Statistical Tests**:
- Compare imputed vs. original distributions
- Test for bias in imputed values
- Validate imputation quality

**Methods**:
1. **Distribution Comparison**: Compare histograms before/after imputation
2. **Statistical Tests**: Test for distribution changes
3. **Sensitivity Analysis**: Test impact of imputation on results

**Literature Support**: 
- [Reference 1]: Missing value handling in time series
- [Reference 2]: Imputation methods for energy data
- [Reference 3]: Validation of imputation quality

---

## Outlier Detection and Treatment

### Detection Methods

#### 1. IQR Method (Primary Method)

**Method**: Interquartile Range (IQR) method

**Approach**:
- Calculate Q1 (25th percentile) and Q3 (75th percentile)
- Calculate IQR = Q3 - Q1
- Identify outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR

**Justification**:
1. **Statistical Foundation**: 
   - Based on quartiles, robust to extreme values
   - Standard method for outlier detection
   - Well-established in statistics

2. **Robustness**:
   - Less sensitive to extreme values than mean-based methods
   - Works well for skewed distributions
   - Appropriate for electricity demand data

3. **Literature Support**:
   - Widely used in time series analysis
   - Standard approach for energy data
   - Proven effectiveness

**When to Use**:
- Primary method for outlier detection
- Skewed distributions
- When robust method is needed

**Limitations**:
- May flag legitimate extreme values (e.g., heat waves)
- May miss outliers in tails
- Requires sufficient data for quartile calculation

**Mitigation**:
- Investigate flagged outliers
- Use domain knowledge to distinguish errors from real events
- Consider event-specific handling (e.g., heat waves)

#### 2. Z-Score Method

**Method**: Z-score method

**Approach**:
- Calculate z-scores: z = (value - mean) / std
- Identify outliers: |z| > threshold (default: 3)

**Justification**:
- Standard statistical method
- Based on mean and standard deviation
- Simple and interpretable

**When to Use**:
- Normal distributions
- When mean-based method is appropriate
- Complementary to IQR method

**Limitations**:
- Sensitive to extreme values
- Assumes normal distribution
- May not work well for skewed data

**Mitigation**:
- Use with caution for skewed distributions
- Combine with IQR method
- Validate with domain knowledge

#### 3. Domain Knowledge

**Method**: Expected ranges based on domain knowledge

**Approach**:
- Define expected ranges (e.g., negative consumption is invalid)
- Flag values outside expected ranges

**Justification**:
- Incorporates domain expertise
- Identifies domain-specific outliers
- Complements statistical methods

**When to Use**:
- Domain-specific validation (e.g., negative consumption)
- Known error patterns
- Complementary to statistical methods

### Treatment Strategy

#### Capping (Primary Method)

**Strategy**: Cap outliers at reasonable bounds (1st/99th percentiles)

**Justification**:
1. **Preserves Data**:
   - Keeps all observations
   - Maintains sample size
   - Preserves temporal order

2. **Reduces Impact**:
   - Limits extreme value influence
   - Prevents outliers from dominating analysis
   - Maintains distribution shape

3. **Bounds Selection**:
   - 1st/99th percentiles as reasonable bounds
   - Preserves 98% of data
   - Removes most extreme outliers

**When to Use**:
- Outliers are likely errors
- When preserving data is important
- When removing outliers would lose important information

**Limitations**:
- May underestimate peak demand
- May hide legitimate extreme events
- May affect risk assessment

**Mitigation**:
- Investigate outliers before capping
- Preserve extreme events (e.g., heat waves)
- Document capping decisions
- Sensitivity analysis with different bounds

#### Removal

**Strategy**: Remove rows with outliers

**Justification**:
- Eliminates problematic values
- Preserves data integrity
- Simple approach

**When to Use**:
- Outliers are clearly errors
- When removing doesn't affect analysis
- When sample size is sufficient

**Limitations**:
- Loss of data
- May introduce bias
- May affect temporal continuity

**Mitigation**:
- Only remove clear errors
- Document removal decisions
- Validate impact on analysis

#### Log Transformation

**Strategy**: Apply log transformation to reduce outlier impact

**Justification**:
- Reduces impact of extreme values
- Normalizes skewed distributions
- Preserves relative relationships

**When to Use**:
- Skewed distributions
- When log transformation is appropriate
- When all values are positive

**Limitations**:
- Only works for positive values
- Changes interpretation
- May not address root cause

**Mitigation**:
- Validate log transformation appropriateness
- Document transformation decisions
- Consider alternative transformations

### Justification Summary

**Why Capping vs. Removal**:
- Capping preserves data while reducing outlier impact
- Removal loses information and may affect analysis
- Capping is more appropriate for time series

**How Bounds are Determined**:
- 1st/99th percentiles as standard bounds
- Preserves 98% of data
- Removes most extreme outliers
- Can be adjusted based on domain knowledge

**Impact on Analysis**:
- Reduces extreme value influence
- May affect peak demand estimation
- Documented for transparency

**Literature Support**:
- [Reference 1]: Outlier treatment in energy data
- [Reference 2]: Capping vs. removal strategies
- [Reference 3]: Impact of outlier treatment on forecasting

---

## Data Cleaning

### Duplicate Removal

**Strategy**: Remove duplicate rows based on all columns or specific columns

**Justification**:
- Duplicates may indicate data collection errors
- Duplicates can bias analysis
- Standard practice for data cleaning

**When to Use**:
- When duplicates are identified
- When duplicates are clearly errors
- When preserving one instance is sufficient

**Documentation**: 
- Count of duplicates removed
- Rationale for removal
- Impact on analysis

### Data Type Standardization

**Strategy**: Ensure correct data types for all columns

**Approach**:
- Convert date columns to datetime
- Ensure numeric columns are numeric
- Standardize categorical columns

**Justification**:
- Enables proper analysis
- Prevents errors in calculations
- Ensures consistency

### Date/Time Handling

**Strategy**: Standardize date/time formats and timezones

**Approach**:
- Convert to standard datetime format
- Handle timezone issues
- Ensure chronological ordering

**Justification**:
- Temporal analysis requires consistent dates
- Prevents timezone-related errors
- Enables proper time series analysis

### Regional Name Standardization

**Strategy**: Standardize regional names and codes

**Approach**:
- Map variations to standard names
- Standardize case and formatting
- Handle abbreviations consistently

**Justification**:
- Enables regional analysis
- Prevents grouping errors
- Ensures consistency

### Unit Conversion

**Strategy**: Convert units to standard format (if needed)

**Approach**:
- Convert to consistent units (e.g., MWh)
- Document conversion factors
- Validate conversions

**Justification**:
- Enables comparison across regions
- Prevents unit-related errors
- Ensures consistency

---

## Feature Engineering

### Temporal Features

**Features Created**:
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6, Monday=0)
- `day_of_month`: Day of month (1-31)
- `week_of_year`: Week of year (1-52)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `year`: Year
- `season`: Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
- `is_weekend`: Binary (1 if weekend, 0 otherwise)

**Justification**:
1. **Captures Temporal Patterns**:
   - Daily patterns (hour, day_of_week)
   - Seasonal patterns (month, season, quarter)
   - Long-term trends (year)

2. **Literature Support**:
   - Temporal features essential for electricity forecasting
   - Captures known consumption patterns
   - Improves forecast accuracy

3. **Domain Knowledge**:
   - Electricity demand varies by hour (peak hours)
   - Weekday vs weekend differences
   - Seasonal variations (summer cooling, winter heating)

**Implementation**: See `src/data_processing/preprocessing.py` → `extract_temporal_features()`

---

### Lag Features

**Features Created**:
- `value_lag_1`: Previous day value
- `value_lag_7`: Previous week value
- `value_lag_30`: Previous month value
- `value_lag_365`: Previous year value

**Justification**:
1. **Captures Autocorrelation**:
   - Electricity demand is autocorrelated
   - Previous values influence current demand
   - Essential for time series models

2. **Seasonality Capture**:
   - Lag 365 captures yearly seasonality
   - Lag 7 captures weekly patterns
   - Lag 30 captures monthly patterns

3. **Literature Support**:
   - Lag features improve forecast accuracy
   - Standard practice for time series forecasting
   - Captures temporal dependencies

**Implementation**: See `src/data_processing/preprocessing.py` → `create_lag_features()`

---

### Rolling Statistics

**Features Created**:
- `value_rolling_mean_7`: 7-day rolling mean
- `value_rolling_std_7`: 7-day rolling standard deviation
- `value_rolling_max_7`: 7-day rolling maximum
- `value_rolling_min_7`: 7-day rolling minimum
- Similar for windows: 30, 90, 365 days

**Justification**:
1. **Trends and Volatility**:
   - Rolling mean captures trends
   - Rolling std captures volatility
   - Rolling max/min capture extremes

2. **Patterns**:
   - Short-term trends (7-day)
   - Medium-term trends (30-day)
   - Long-term trends (90, 365-day)

3. **Literature Support**:
   - Rolling statistics improve forecast accuracy
   - Captures local patterns
   - Standard feature engineering practice

**Implementation**: See `src/data_processing/preprocessing.py` → `create_rolling_features()`

---

### Derived Features

#### Demand-Supply Ratio

**Feature**: `demand_supply_ratio = demand / supply`

**Justification**:
- Key metric for risk assessment
- Indicates supply adequacy
- Essential for infrastructure planning

#### Reserve Margin

**Feature**: `reserve_margin = (supply - demand) / demand × 100`

**Justification**:
- Standard metric for grid reliability
- Used in industry (NERC standards)
- Critical for risk assessment

#### Growth Rates

**Features**: `value_growth_1`, `value_growth_7`, `value_growth_30`, `value_growth_365`

**Justification**:
- Captures demand growth trends
- Essential for forecasting
- Used in risk assessment

#### Peak-to-Average Ratio

**Feature**: `peak_to_avg_ratio = peak_demand / average_demand`

**Justification**:
- Measures demand volatility
- Indicates peak demand stress
- Important for capacity planning

**Implementation**: See `src/data_processing/preprocessing.py` → `calculate_growth_rates()`, `calculate_demand_supply_ratio()`

---

## Normalization/Standardization

### When Normalization is Used

**Purpose**: Normalize features for models requiring normalized inputs

**When to Use**:
- Machine learning models (XGBoost) may benefit from normalization
- When features have different scales
- When normalization improves performance

**Methods**:

1. **Standardization** (Z-score):
   - Transform: (value - mean) / std
   - Mean = 0, Std = 1
   - Preserves distribution shape

2. **Min-Max Scaling**:
   - Transform: (value - min) / (max - min)
   - Range: [0, 1]
   - Preserves relative relationships

3. **Robust Scaling**:
   - Transform: (value - median) / IQR
   - Robust to outliers
   - Uses median and IQR

**Justification**:
- Some models require normalized inputs
- Improves model convergence
- Reduces feature scale effects

**Implementation**: See `src/data_processing/preprocessing.py` → `normalize_features()`

---

## Data Validation

### Post-Processing Validation

**Purpose**: Ensure data quality after preprocessing

**Checks**:
1. **Completeness**: No unexpected missing values
2. **Consistency**: Values within expected ranges
3. **Temporal Integrity**: Chronological ordering maintained
4. **Feature Validity**: All features are valid

### Quality Checks

**Checks Performed**:
- Missing value counts (should be minimal)
- Outlier counts (should be reasonable)
- Data type validation
- Range validation (e.g., no negative consumption)
- Temporal consistency

### Consistency Checks

**Checks**:
- Temporal ordering (dates in order)
- Regional consistency (valid region names)
- Sector consistency (valid sector names)
- Unit consistency (consistent units)

### Final Data Quality Report

**Report Includes**:
- Preprocessing summary
- Missing value treatment summary
- Outlier treatment summary
- Feature engineering summary
- Final data quality metrics
- Validation results

---

## Preprocessing Pipeline

### Step-by-Step Pipeline

1. **Load Raw Data**
   - Load data from EIA API or files
   - Initial data quality assessment

2. **Data Quality Assessment**
   - Assess missing values, outliers, duplicates
   - Generate quality report

3. **Missing Value Handling**
   - Apply forward fill for time series
   - Apply mean/median for non-time series
   - Document decisions

4. **Outlier Detection**
   - Detect outliers using IQR method
   - Investigate outliers
   - Document findings

5. **Outlier Treatment**
   - Cap outliers at percentiles
   - Document treatment decisions

6. **Data Cleaning**
   - Remove duplicates
   - Standardize data types
   - Standardize regional names

7. **Feature Engineering**
   - Extract temporal features
   - Create lag features
   - Create rolling features
   - Calculate derived features

8. **Normalization** (if needed)
   - Normalize features for ML models

9. **Data Validation**
   - Final quality checks
   - Consistency validation
   - Generate final report

### Order of Operations

**Critical Order**:
1. Missing value handling before outlier detection (outliers may be missing-related)
2. Outlier treatment before feature engineering (clean data for features)
3. Data cleaning before feature engineering (standardized data)
4. Feature engineering before normalization (features on original scale first)

### Dependencies

**Dependencies**:
- Temporal features require date column
- Lag features require sorted data
- Rolling features require temporal ordering
- Derived features require demand/supply columns

### Error Handling

**Error Handling**:
- Graceful handling of missing columns
- Logging of all preprocessing steps
- Validation at each step
- Rollback capability if needed

---

## Documentation Standards

### Requirements

1. **All Decisions Documented**: Rationale for every preprocessing decision
2. **All Transformations Documented**: Every transformation explained
3. **All Parameters Documented**: All parameters and thresholds documented
4. **Reproducibility Ensured**: Pipeline is reproducible

### Documentation Elements

**For Each Preprocessing Step**:
- What was done
- Why it was done (rationale)
- How it was done (method)
- Parameters used
- Results/impact

### Reproducibility

**Ensured Through**:
- Code versioning
- Parameter documentation
- Random seed documentation (where applicable)
- Clear pipeline documentation

---

## Implementation

**Code Location**:
- `src/data_processing/cleaning.py` - Data cleaning implementation
- `src/data_processing/preprocessing.py` - Feature engineering implementation

**Usage**:
```python
from src.data_processing.cleaning import DataCleaner
from src.data_processing.preprocessing import DataPreprocessor

# Clean data
cleaner = DataCleaner()
df_cleaned, report = cleaner.clean_dataset(df)

# Preprocess data
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_dataset(df_cleaned)
```

---

## Literature Support

### Missing Value Handling

- [Reference 1]: Missing value handling in time series data
- [Reference 2]: Forward fill for electricity demand data
- [Reference 3]: Validation of imputation methods

### Outlier Treatment

- [Reference 1]: Outlier detection in energy data
- [Reference 2]: Capping vs. removal strategies
- [Reference 3]: Impact on forecasting accuracy

### Feature Engineering

- [Reference 1]: Temporal features for electricity forecasting
- [Reference 2]: Lag features in time series
- [Reference 3]: Rolling statistics for demand forecasting

---

## References

- See `../references/references.md` for full bibliography
- See `research_design.md` for overall methodology
- See `docs/methodology.md` for general methodology

