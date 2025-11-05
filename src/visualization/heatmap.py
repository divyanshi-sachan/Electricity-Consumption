"""
Interactive Geographical Heatmap Module
Creates interactive heatmaps for electricity demand/supply visualization
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectricityHeatmap:
    """
    Create interactive geographical heatmaps for electricity demand/supply
    """
    
    def __init__(self, data: pd.DataFrame, lat_column: str = 'latitude', lon_column: str = 'longitude'):
        """
        Initialize with electricity data
        
        Args:
            data: DataFrame with regional electricity data
            lat_column: Column name for latitude
            lon_column: Column name for longitude
        """
        self.data = data.copy()
        self.lat_column = lat_column
        self.lon_column = lon_column
    
    def create_interactive_map(self, 
                              region: Optional[str] = None,
                              metric: str = 'demand',
                              date: Optional[str] = None,
                              center_lat: float = 39.8283,
                              center_lon: float = -98.5795,
                              zoom_start: int = 4) -> folium.Map:
        """
        Create interactive folium map
        
        Args:
            region: Selected region (None for all regions)
            metric: 'demand', 'supply', 'ratio', or 'reserve_margin'
            date: Specific date to visualize (if None, uses latest or average)
            center_lat: Map center latitude (default: US center)
            center_lon: Map center longitude (default: US center)
            zoom_start: Initial zoom level
            
        Returns:
            folium.Map object
        """
        # Filter by region if specified
        if region and 'region' in self.data.columns:
            map_data = self.data[self.data['region'] == region].copy()
        else:
            map_data = self.data.copy()
        
        # Filter by date if specified
        if date and 'period' in map_data.columns:
            map_data = map_data[map_data['period'] == date].copy()
        elif 'period' in map_data.columns:
            # Use latest date or average
            if map_data['period'].dtype == 'datetime64[ns]':
                map_data = map_data[map_data['period'] == map_data['period'].max()].copy()
            else:
                # Average across all dates
                map_data = map_data.groupby('region').agg({
                    metric: 'mean',
                    self.lat_column: 'first',
                    self.lon_column: 'first'
                }).reset_index()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Check if we have lat/lon columns
        if self.lat_column not in map_data.columns or self.lon_column not in map_data.columns:
            logger.warning("Latitude/longitude columns not found. Using default locations.")
            # You may need to add geocoding here or use region names
            return m
        
        # Add metric column if it doesn't exist
        if metric not in map_data.columns:
            logger.warning(f"Metric '{metric}' not found in data")
            return m
        
        # Prepare data for heatmap
        heat_data = []
        for idx, row in map_data.iterrows():
            if pd.notna(row[self.lat_column]) and pd.notna(row[self.lon_column]):
                value = row[metric]
                if pd.notna(value) and value > 0:
                    # Normalize value for heatmap intensity (0-1 scale)
                    max_val = map_data[metric].max()
                    intensity = value / max_val if max_val > 0 else 0
                    heat_data.append([row[self.lat_column], row[self.lon_column], intensity])
        
        # Add heatmap layer
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={
                    0.2: 'blue',
                    0.4: 'cyan',
                    0.6: 'lime',
                    0.8: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)
        
        # Add markers for each region
        for idx, row in map_data.iterrows():
            if pd.notna(row[self.lat_column]) and pd.notna(row[self.lon_column]):
                popup_text = f"""
                <b>Region:</b> {row.get('region', 'N/A')}<br>
                <b>{metric.title()}:</b> {row[metric]:,.2f}<br>
                """
                
                if 'supply' in row and 'demand' in row:
                    popup_text += f"<b>Demand:</b> {row['demand']:,.2f}<br>"
                    popup_text += f"<b>Supply:</b> {row['supply']:,.2f}<br>"
                
                if 'reserve_margin' in row:
                    popup_text += f"<b>Reserve Margin:</b> {row['reserve_margin']:.2f}%<br>"
                
                folium.CircleMarker(
                    location=[row[self.lat_column], row[self.lon_column]],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.6
                ).add_to(m)
        
        logger.info(f"Interactive map created with {len(heat_data)} data points")
        
        return m
    
    def create_plotly_heatmap(self, 
                              region: Optional[str] = None,
                              metric: str = 'demand',
                              date: Optional[str] = None) -> go.Figure:
        """
        Create interactive plotly heatmap
        
        Args:
            region: Selected region (None for all regions)
            metric: 'demand', 'supply', 'ratio', or 'reserve_margin'
            date: Specific date to visualize
            
        Returns:
            plotly Figure object
        """
        # Filter by region if specified
        if region and 'region' in self.data.columns:
            map_data = self.data[self.data['region'] == region].copy()
        else:
            map_data = self.data.copy()
        
        # Filter by date if specified
        if date and 'period' in map_data.columns:
            map_data = map_data[map_data['period'] == date].copy()
        elif 'period' in map_data.columns:
            # Average across all dates
            if len(map_data['period'].unique()) > 1:
                map_data = map_data.groupby('region').agg({
                    metric: 'mean',
                    self.lat_column: 'first',
                    self.lon_column: 'first'
                }).reset_index()
        
        if metric not in map_data.columns:
            logger.error(f"Metric '{metric}' not found in data")
            return go.Figure()
        
        # Create scatter mapbox plot
        fig = go.Figure()
        
        # Check if we have lat/lon columns
        if self.lat_column not in map_data.columns or self.lon_column not in map_data.columns:
            # If no coordinates, create a bar chart by region
            if 'region' in map_data.columns:
                fig = px.bar(
                    map_data.groupby('region')[metric].mean().reset_index(),
                    x='region',
                    y=metric,
                    title=f'{metric.title()} by Region'
                )
                return fig
            else:
                logger.error("No location data available")
                return fig
        
        # Create map scatter
        fig.add_trace(
            go.Scattermapbox(
                lat=map_data[self.lat_column],
                lon=map_data[self.lon_column],
                mode='markers',
                marker=dict(
                    size=map_data[metric] / map_data[metric].max() * 50 + 10,
                    color=map_data[metric],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=metric.title())
                ),
                text=map_data.get('region', ''),
                hovertemplate=f'<b>%{{text}}</b><br>' +
                             f'{metric.title()}: %{{marker.color}}<br>' +
                             '<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=f'Electricity {metric.title()} Heatmap',
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=map_data[self.lat_column].mean(),
                    lon=map_data[self.lon_column].mean()
                ),
                zoom=4
            ),
            height=600
        )
        
        logger.info("Plotly heatmap created")
        
        return fig
    
    def create_demand_supply_comparison(self,
                                       region: Optional[str] = None,
                                       date: Optional[str] = None) -> go.Figure:
        """
        Create comparison chart of demand vs supply
        
        Args:
            region: Selected region (None for all regions)
            date: Specific date to visualize
            
        Returns:
            plotly Figure object
        """
        # Filter data
        if region and 'region' in self.data.columns:
            map_data = self.data[self.data['region'] == region].copy()
        else:
            map_data = self.data.copy()
        
        if date and 'period' in map_data.columns:
            map_data = map_data[map_data['period'] == date].copy()
        
        if 'demand' not in map_data.columns or 'supply' not in map_data.columns:
            logger.error("Demand or supply columns not found")
            return go.Figure()
        
        # Group by region
        if 'region' in map_data.columns:
            comparison_data = map_data.groupby('region').agg({
                'demand': 'mean',
                'supply': 'mean'
            }).reset_index()
            
            # Calculate reserve margin
            comparison_data['reserve_margin'] = (
                (comparison_data['supply'] - comparison_data['demand']) / 
                comparison_data['demand'] * 100
            )
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison_data['region'],
                y=comparison_data['demand'],
                name='Demand',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=comparison_data['region'],
                y=comparison_data['supply'],
                name='Supply',
                marker_color='green'
            ))
            
            fig.update_layout(
                title='Demand vs Supply by Region',
                xaxis_title='Region',
                yaxis_title='Value',
                barmode='group',
                height=600
            )
        else:
            # Single region or time series
            fig = go.Figure()
            
            if 'period' in map_data.columns:
                map_data = map_data.sort_values('period')
                fig.add_trace(go.Scatter(
                    x=map_data['period'],
                    y=map_data['demand'],
                    name='Demand',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=map_data['period'],
                    y=map_data['supply'],
                    name='Supply',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title='Demand vs Supply Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=600
                )
        
        logger.info("Demand-supply comparison chart created")
        
        return fig
    
    def save_map(self, map_obj: folium.Map, filename: str) -> str:
        """
        Save folium map to HTML file
        
        Args:
            map_obj: folium.Map object
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        from pathlib import Path
        
        output_dir = Path(__file__).parent.parent.parent / "figures"
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        map_obj.save(str(filepath))
        
        logger.info(f"Map saved to {filepath}")
        
        return str(filepath)
    
    def save_plotly_figure(self, fig: go.Figure, filename: str) -> str:
        """
        Save plotly figure to HTML file
        
        Args:
            fig: plotly Figure object
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        from pathlib import Path
        
        output_dir = Path(__file__).parent.parent.parent / "figures"
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        if filename.endswith('.html'):
            fig.write_html(str(filepath))
        elif filename.endswith('.png'):
            fig.write_image(str(filepath))
        else:
            filepath = filepath.with_suffix('.html')
            fig.write_html(str(filepath))
        
        logger.info(f"Figure saved to {filepath}")
        
        return str(filepath)

