# Example Streamlit Dashboard Code
# Save this to: app/dashboard.py or src/visualization/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Load forecast data
@st.cache_data
def load_forecast_data():
    project_root = Path(__file__).parent.parent.parent
    forecast_path = project_root / 'data' / 'processed' / 'xgboost_forecasts.csv'
    combined_path = project_root / 'data' / 'processed' / 'historical_and_forecasts.csv'
    
    forecast_df = pd.read_csv(forecast_path)
    combined_df = pd.read_csv(combined_path)
    
    # Convert period to datetime
    forecast_df['period'] = pd.to_datetime(forecast_df['period'])
    combined_df['period'] = pd.to_datetime(combined_df['period'])
    
    return forecast_df, combined_df

# Load model
@st.cache_resource
def load_model():
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'src' / 'analysis' / 'xgboost_forecast_model.pkl'
    model_data = joblib.load(model_path)
    return model_data

# Streamlit App
st.title("Electricity Consumption Forecast Dashboard")
st.markdown("### XGBoost-Based Forecasting")

# Load data
forecast_df, combined_df = load_forecast_data()
model_data = load_model()

# Sidebar filters
st.sidebar.header("Filters")
selected_state = st.sidebar.selectbox(
    "Select State",
    options=sorted(combined_df['stateid'].unique())
)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    state_data = combined_df[combined_df['stateid'] == selected_state]
    hist_mean = state_data[state_data['type'] == 'Historical']['value'].mean()
    st.metric("Historical Average", f"{hist_mean:,.0f} MWh")

with col2:
    fore_mean = state_data[state_data['type'] == 'Forecasted']['value'].mean()
    st.metric("Forecasted Average", f"{fore_mean:,.0f} MWh")

with col3:
    growth = ((fore_mean - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0
    st.metric("Projected Growth", f"{growth:+.1f}%")

# Plot forecast
fig = go.Figure()

# Historical data
hist_data = state_data[state_data['type'] == 'Historical']
fig.add_trace(go.Scatter(
    x=hist_data['period'],
    y=hist_data['value'],
    mode='lines+markers',
    name='Historical',
    line=dict(color='steelblue', width=2)
))

# Forecasted data
fore_data = state_data[state_data['type'] == 'Forecasted']
fig.add_trace(go.Scatter(
    x=fore_data['period'],
    y=fore_data['value'],
    mode='lines+markers',
    name='Forecasted',
    line=dict(color='red', width=2, dash='dash')
))

fig.update_layout(
    title=f'Electricity Sales Forecast: {selected_state}',
    xaxis_title='Date',
    yaxis_title='Sales (MWh)',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Model info
with st.expander("Model Information"):
    st.write(f"**Model:** {model_data['model_metrics']['Model']}")
    st.write(f"**R² Score:** {model_data['model_metrics']['R²']:.4f}")
    st.write(f"**RMSE:** {model_data['model_metrics']['RMSE']:.2f}")
    st.write(f"**Forecast Start:** {model_data['forecast_start']}")
    st.write(f"**Forecast Periods:** {model_data['forecast_periods']} months")