"""
Analysis module for exploratory data analysis, forecast validation, and regional insights
"""

from .eda import (
    analyze_missing_values, check_negative_values, get_statistical_summary,
    analyze_state_ids, plot_sales_by_year, plot_sales_by_month,
    plot_price_vs_sales, generate_eda_report
)

from .forecast_validation import (
    calculate_forecast_metrics, plot_training_validation_curve,
    plot_residuals, plot_feature_importance, generate_validation_report
)

from .regional_insights import (
    calculate_state_growth_metrics, identify_high_growth_states,
    identify_declining_states, identify_volatile_states,
    plot_state_forecast_trends, generate_regional_insights_report
)

from .risk_assessment import (
    calculate_demand_supply_gap, identify_high_risk_regions,
    generate_policy_recommendations, create_risk_heatmap
)

__all__ = [
    # EDA
    'analyze_missing_values', 'check_negative_values', 'get_statistical_summary',
    'analyze_state_ids', 'plot_sales_by_year', 'plot_sales_by_month',
    'plot_price_vs_sales', 'generate_eda_report',
    # Forecast Validation
    'calculate_forecast_metrics', 'plot_training_validation_curve',
    'plot_residuals', 'plot_feature_importance', 'generate_validation_report',
    # Regional Insights
    'calculate_state_growth_metrics', 'identify_high_growth_states',
    'identify_declining_states', 'identify_volatile_states',
    'plot_state_forecast_trends', 'generate_regional_insights_report',
    # Risk Assessment
    'calculate_demand_supply_gap', 'identify_high_risk_regions',
    'generate_policy_recommendations', 'create_risk_heatmap'
]

