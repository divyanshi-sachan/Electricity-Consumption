"""
Interactive Dashboard for Electricity Consumption Analysis
Built with Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.heatmap import ElectricityHeatmap
from src.analysis.exploratory import ExploratoryAnalyzer
from src.analysis.forecasting import ForecastingModel
from src.analysis.risk_analysis import RiskAnalyzer

# Page config
st.set_page_config(
    page_title="Electricity Consumption Analysis",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Electricity Consumption Analysis Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Overview", "Regional Analysis", "Forecasting", "Risk Analysis", "Interactive Heatmap"]
)

# Load data (placeholder - you'll need to load your actual data)
@st.cache_data
def load_data():
    """
    Load electricity consumption data
    Replace this with your actual data loading logic
    """
    # This is a placeholder - replace with actual data loading
    return pd.DataFrame()


def main():
    """Main dashboard function"""
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.warning("⚠️ No data loaded. Please load your electricity consumption data.")
        st.info("""
        To use this dashboard:
        1. Load your data in the `load_data()` function
        2. Ensure data has columns: 'region', 'period', 'demand', 'supply'
        3. Optionally include 'latitude' and 'longitude' for map visualization
        """)
        return
    
    if page == "Overview":
        show_overview(data)
    elif page == "Regional Analysis":
        show_regional_analysis(data)
    elif page == "Forecasting":
        show_forecasting(data)
    elif page == "Risk Analysis":
        show_risk_analysis(data)
    elif page == "Interactive Heatmap":
        show_heatmap(data)


def show_overview(data):
    """Show overview statistics"""
    st.header("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Regions", len(data['region'].unique()) if 'region' in data.columns else 0)
    
    with col2:
        total_demand = data['demand'].sum() if 'demand' in data.columns else 0
        st.metric("Total Demand", f"{total_demand:,.0f}")
    
    with col3:
        total_supply = data['supply'].sum() if 'supply' in data.columns else 0
        st.metric("Total Supply", f"{total_supply:,.0f}")
    
    with col4:
        if 'demand' in data.columns and 'supply' in data.columns:
            avg_reserve = ((data['supply'].sum() - data['demand'].sum()) / data['demand'].sum() * 100)
            st.metric("Avg Reserve Margin", f"{avg_reserve:.2f}%")
    
    # Time series plot
    if 'period' in data.columns and 'demand' in data.columns:
        st.subheader("Demand Over Time")
        time_series = data.groupby('period')['demand'].sum().reset_index()
        fig = px.line(time_series, x='period', y='demand', title='Total Electricity Demand Over Time')
        st.plotly_chart(fig, use_container_width=True)


def show_regional_analysis(data):
    """Show regional analysis"""
    st.header("🌍 Regional Analysis")
    
    if 'region' not in data.columns:
        st.error("Region column not found in data")
        return
    
    # Region selector
    regions = ['All'] + sorted(data['region'].unique().tolist())
    selected_region = st.selectbox("Select Region", regions)
    
    if selected_region == 'All':
        region_data = data.copy()
    else:
        region_data = data[data['region'] == selected_region].copy()
    
    # Initialize analyzer
    analyzer = ExploratoryAnalyzer(region_data)
    
    # Regional statistics
    if selected_region != 'All':
        st.subheader(f"Statistics for {selected_region}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_demand = region_data['demand'].mean() if 'demand' in region_data.columns else 0
            st.metric("Average Demand", f"{avg_demand:,.2f}")
        
        with col2:
            peak_demand = region_data['demand'].max() if 'demand' in region_data.columns else 0
            st.metric("Peak Demand", f"{peak_demand:,.2f}")
        
        with col3:
            if 'demand' in region_data.columns:
                peak_to_avg = analyzer.calculate_peak_to_average_ratio('demand')
                st.metric("Peak-to-Average Ratio", f"{peak_to_avg:.2f}")
    
    # Regional comparison chart
    if selected_region == 'All':
        st.subheader("Regional Comparison")
        regional_stats = data.groupby('region').agg({
            'demand': 'mean',
            'supply': 'mean'
        }).reset_index()
        
        fig = px.bar(
            regional_stats,
            x='region',
            y=['demand', 'supply'],
            title='Average Demand and Supply by Region',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_forecasting(data):
    """Show forecasting analysis"""
    st.header("📈 Forecasting")
    
    if 'region' not in data.columns:
        st.error("Region column not found in data")
        return
    
    # Region selector
    regions = sorted(data['region'].unique().tolist())
    selected_region = st.selectbox("Select Region for Forecasting", regions)
    
    region_data = data[data['region'] == selected_region].copy()
    
    if region_data.empty:
        st.error(f"No data found for region: {selected_region}")
        return
    
    # Initialize forecasting model
    model = ForecastingModel(region_data)
    
    # Prepare data
    train_data, test_data = model.prepare_data('demand', 'period')
    
    if train_data.empty or test_data.empty:
        st.error("Insufficient data for forecasting")
        return
    
    # Model selection
    model_type = st.selectbox("Select Forecasting Model", ["Ensemble", "ARIMA", "Prophet"])
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            if model_type == "Ensemble":
                results = model.ensemble_forecast(train_data, test_data, 'demand', 365)
            elif model_type == "ARIMA":
                results = model.arima_forecast(train_data, test_data, 'demand', 365)
            else:
                results = model.prophet_forecast(train_data, test_data, 'demand', 365)
            
            if results:
                # Display metrics
                st.subheader("Model Performance")
                metrics = results.get('metrics', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                with col3:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                
                # Plot forecast
                st.subheader("Forecast Results")
                forecast = results.get('forecast', pd.Series())
                long_term = results.get('long_term_forecast', pd.Series())
                
                if not forecast.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=train_data.index,
                        y=train_data['demand'],
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=test_data.index,
                        y=test_data['demand'],
                        name='Actual',
                        line=dict(color='green')
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast,
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    if not long_term.empty:
                        fig.add_trace(go.Scatter(
                            x=long_term.index,
                            y=long_term,
                            name='Long-term Forecast (5-10 years)',
                            line=dict(color='orange', dash='dot')
                        ))
                    
                    fig.update_layout(
                        title='Electricity Demand Forecast',
                        xaxis_title='Date',
                        yaxis_title='Demand',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)


def show_risk_analysis(data):
    """Show risk analysis"""
    st.header("⚠️ Risk Analysis")
    
    if 'region' not in data.columns:
        st.error("Region column not found in data")
        return
    
    # Initialize risk analyzer
    risk_analyzer = RiskAnalyzer(data)
    
    # Calculate reserve margins
    risk_df = risk_analyzer.calculate_reserve_margin('demand', 'supply', 'region')
    
    # Identify at-risk regions
    at_risk = risk_analyzer.identify_at_risk_regions('demand', 'supply', 'region')
    
    st.subheader("At-Risk Regions")
    
    if not at_risk.empty:
        st.dataframe(at_risk[['region', 'avg_reserve_margin', 'avg_demand', 'peak_demand']].style.format({
            'avg_reserve_margin': '{:.2f}%',
            'avg_demand': '{:,.2f}',
            'peak_demand': '{:,.2f}'
        }))
        
        # Risk visualization
        fig = px.bar(
            at_risk,
            x='region',
            y='avg_reserve_margin',
            title='Reserve Margin by Region (At-Risk Regions)',
            color='avg_reserve_margin',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Root cause analysis
        st.subheader("Root Cause Analysis")
        selected_region = st.selectbox("Select Region for Root Cause Analysis", at_risk['region'].tolist())
        
        if st.button("Analyze Root Causes"):
            root_causes = risk_analyzer.root_cause_analysis(selected_region, 'demand', 'supply', 'region')
            
            if root_causes:
                st.write(f"### Analysis for {selected_region}")
                
                # Root causes
                st.write("**Root Causes:**")
                for cause in root_causes.get('root_causes', []):
                    st.write(f"- **{cause['issue']}**: {cause['description']} (Impact: {cause['impact']})")
                
                # Recommendations
                st.write("**Recommendations:**")
                for rec in root_causes.get('recommendations', []):
                    st.write(f"- **{rec['action']}** (Priority: {rec['priority']}, Timeline: {rec['timeline']})")
    else:
        st.success("No at-risk regions identified. All regions have adequate reserve margins.")


def show_heatmap(data):
    """Show interactive heatmap"""
    st.header("🗺️ Interactive Geographical Heatmap")
    
    # Metric selection
    metric = st.selectbox("Select Metric", ["demand", "supply", "ratio", "reserve_margin"])
    
    # Region filter
    if 'region' in data.columns:
        regions = ['All'] + sorted(data['region'].unique().tolist())
        selected_region = st.selectbox("Filter by Region", regions)
    else:
        selected_region = 'All'
    
    # Date filter
    if 'period' in data.columns:
        dates = ['All'] + sorted(data['period'].unique().tolist())
        selected_date = st.selectbox("Filter by Date", dates)
    else:
        selected_date = 'All'
    
    # Initialize heatmap
    heatmap = ElectricityHeatmap(data)
    
    # Create plotly heatmap
    region_filter = None if selected_region == 'All' else selected_region
    date_filter = None if selected_date == 'All' else selected_date
    
    fig = heatmap.create_plotly_heatmap(region_filter, metric, date_filter)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Demand-Supply Comparison
    st.subheader("Demand vs Supply Comparison")
    comparison_fig = heatmap.create_demand_supply_comparison(region_filter, date_filter)
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)


if __name__ == "__main__":
    main()

