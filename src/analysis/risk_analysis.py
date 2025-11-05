"""
Risk Analysis Module
Identifies areas at risk of under-supply and root causes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    Class for risk analysis of electricity supply and demand
    """
    
    def __init__(self, df: pd.DataFrame, forecast_data: Optional[pd.DataFrame] = None):
        """
        Initialize RiskAnalyzer
        
        Args:
            df: Historical electricity consumption data
            forecast_data: Forecasted future demand (optional)
        """
        self.df = df.copy()
        self.forecast_data = forecast_data.copy() if forecast_data is not None else None
    
    def calculate_reserve_margin(self,
                                demand_column: str = 'demand',
                                supply_column: str = 'supply',
                                region_column: Optional[str] = 'region') -> pd.DataFrame:
        """
        Calculate reserve margin (available capacity vs peak demand)
        
        Args:
            demand_column: Column with demand values
            supply_column: Column with supply/capacity values
            region_column: Column with region identifiers (optional)
            
        Returns:
            DataFrame with reserve margin calculations
        """
        if demand_column not in self.df.columns or supply_column not in self.df.columns:
            logger.error("Demand or supply columns not found")
            return pd.DataFrame()
        
        risk_df = self.df.copy()
        
        # Calculate reserve margin
        risk_df['reserve_margin'] = (
            (risk_df[supply_column] - risk_df[demand_column]) / 
            risk_df[demand_column] * 100
        )
        
        # Calculate demand-supply ratio
        risk_df['demand_supply_ratio'] = (
            risk_df[demand_column] / risk_df[supply_column]
        )
        
        # Risk levels (typical reserve margin should be >15%)
        risk_df['risk_level'] = risk_df['reserve_margin'].apply(
            lambda x: 'High' if x < 10 else ('Medium' if x < 15 else 'Low')
        )
        
        logger.info("Reserve margin calculated")
        
        return risk_df
    
    def identify_at_risk_regions(self,
                                 demand_column: str = 'demand',
                                 supply_column: str = 'supply',
                                 region_column: str = 'region',
                                 threshold: float = 15.0) -> pd.DataFrame:
        """
        Identify regions at risk of under-supply
        
        Args:
            demand_column: Column with demand values
            supply_column: Column with supply values
            region_column: Column with region identifiers
            threshold: Reserve margin threshold (default 15%)
            
        Returns:
            DataFrame with at-risk regions
        """
        risk_df = self.calculate_reserve_margin(demand_column, supply_column, region_column)
        
        if region_column not in risk_df.columns:
            # Single region analysis
            at_risk = risk_df[risk_df['reserve_margin'] < threshold].copy()
        else:
            # Regional analysis
            regional_risk = risk_df.groupby(region_column).agg({
                'reserve_margin': 'mean',
                demand_column: ['mean', 'max'],
                supply_column: 'mean'
            }).reset_index()
            
            regional_risk.columns = [
                region_column, 'avg_reserve_margin', 'avg_demand', 
                'peak_demand', 'avg_supply'
            ]
            
            at_risk = regional_risk[regional_risk['avg_reserve_margin'] < threshold].copy()
            at_risk = at_risk.sort_values('avg_reserve_margin')
        
        logger.info(f"Identified {len(at_risk)} at-risk regions")
        
        return at_risk
    
    def analyze_demand_growth(self,
                             value_column: str = 'demand',
                             date_column: str = 'period',
                             region_column: Optional[str] = 'region',
                             years: int = 5) -> pd.DataFrame:
        """
        Analyze demand growth trends
        
        Args:
            value_column: Column with demand values
            date_column: Column with dates
            region_column: Column with region identifiers (optional)
            years: Number of years to analyze
            
        Returns:
            DataFrame with growth analysis
        """
        if date_column not in self.df.columns:
            logger.error(f"Date column '{date_column}' not found")
            return pd.DataFrame()
        
        growth_df = self.df.copy()
        growth_df[date_column] = pd.to_datetime(growth_df[date_column])
        
        # Filter to recent years
        if 'year' in growth_df.columns:
            recent_years = growth_df['year'].max() - years
            growth_df = growth_df[growth_df['year'] >= recent_years]
        
        if region_column and region_column in growth_df.columns:
            # Regional growth rates
            growth_analysis = growth_df.groupby([region_column, 'year'])[value_column].sum().reset_index()
            growth_rates = growth_analysis.groupby(region_column).apply(
                lambda x: ((x[value_column].iloc[-1] - x[value_column].iloc[0]) / 
                          x[value_column].iloc[0] / years * 100) if len(x) > 1 else 0
            ).reset_index(name='annual_growth_rate')
            
            logger.info("Regional growth analysis complete")
            return growth_rates
        else:
            # Overall growth rate
            yearly_totals = growth_df.groupby('year')[value_column].sum()
            if len(yearly_totals) > 1:
                growth_rate = ((yearly_totals.iloc[-1] - yearly_totals.iloc[0]) / 
                              yearly_totals.iloc[0] / years * 100)
                logger.info(f"Overall annual growth rate: {growth_rate:.2f}%")
                return pd.DataFrame({'annual_growth_rate': [growth_rate]})
        
        return pd.DataFrame()
    
    def root_cause_analysis(self,
                           region: str,
                           demand_column: str = 'demand',
                           supply_column: str = 'supply',
                           region_column: str = 'region') -> Dict:
        """
        Perform root cause analysis for a specific region
        
        Args:
            region: Region identifier
            demand_column: Column with demand values
            supply_column: Column with supply values
            region_column: Column with region identifiers
            
        Returns:
            Dictionary with root cause analysis results
        """
        if region_column not in self.df.columns:
            logger.error("Region column not found")
            return {}
        
        region_data = self.df[self.df[region_column] == region].copy()
        
        if len(region_data) == 0:
            logger.error(f"No data found for region: {region}")
            return {}
        
        analysis = {
            'region': region,
            'root_causes': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Current status
        current_reserve = self.calculate_reserve_margin(demand_column, supply_column, region_column)
        region_reserve = current_reserve[current_reserve[region_column] == region]
        
        if len(region_reserve) > 0:
            avg_reserve = region_reserve['reserve_margin'].mean()
            analysis['metrics']['current_reserve_margin'] = avg_reserve
            
            # Identify root causes
            if avg_reserve < 15:
                # High demand growth
                growth_analysis = self.analyze_demand_growth(demand_column, 'period', region_column)
                if len(growth_analysis) > 0:
                    region_growth = growth_analysis[growth_analysis[region_column] == region]
                    if len(region_growth) > 0 and region_growth['annual_growth_rate'].iloc[0] > 3:
                        analysis['root_causes'].append({
                            'issue': 'High demand growth',
                            'description': f"Demand growing at {region_growth['annual_growth_rate'].iloc[0]:.2f}% annually",
                            'impact': 'High'
                        })
                        analysis['recommendations'].append({
                            'action': 'Expand generation capacity',
                            'priority': 'High',
                            'timeline': '1-3 years'
                        })
                
                # Supply constraints
                if supply_column in region_data.columns:
                    supply_growth = region_data.groupby('year')[supply_column].mean()
                    if len(supply_growth) > 1:
                        supply_change = ((supply_growth.iloc[-1] - supply_growth.iloc[0]) / 
                                        supply_growth.iloc[0] * 100)
                        if supply_change < 0:
                            analysis['root_causes'].append({
                                'issue': 'Declining supply capacity',
                                'description': f"Supply decreased by {abs(supply_change):.2f}%",
                                'impact': 'High'
                            })
                            analysis['recommendations'].append({
                                'action': 'Renew aging infrastructure',
                                'priority': 'High',
                                'timeline': '2-5 years'
                            })
                
                # Peak demand stress
                peak_demand = region_data[demand_column].max()
                avg_demand = region_data[demand_column].mean()
                peak_to_avg = peak_demand / avg_demand if avg_demand > 0 else 0
                
                if peak_to_avg > 1.5:
                    analysis['root_causes'].append({
                        'issue': 'High peak-to-average ratio',
                        'description': f"Peak demand is {peak_to_avg:.2f}x average demand",
                        'impact': 'Medium'
                    })
                    analysis['recommendations'].append({
                        'action': 'Implement demand response programs',
                        'priority': 'Medium',
                        'timeline': '6-12 months'
                    })
        
        logger.info(f"Root cause analysis complete for {region}")
        
        return analysis
    
    def generate_investment_recommendations(self,
                                          risk_df: pd.DataFrame,
                                          region_column: str = 'region') -> pd.DataFrame:
        """
        Generate prioritized investment recommendations
        
        Args:
            risk_df: DataFrame with risk analysis results
            region_column: Column with region identifiers
            
        Returns:
            DataFrame with investment recommendations
        """
        recommendations = []
        
        if region_column in risk_df.columns:
            for region in risk_df[region_column].unique():
                region_risk = risk_df[risk_df[region_column] == region].iloc[0]
                
                # Root cause analysis
                root_causes = self.root_cause_analysis(region)
                
                # Priority ranking
                priority_score = 0
                if region_risk.get('reserve_margin', 100) < 10:
                    priority_score += 10
                elif region_risk.get('reserve_margin', 100) < 15:
                    priority_score += 5
                
                if region_risk.get('demand', 0) > risk_df['demand'].quantile(0.75):
                    priority_score += 5
                
                recommendations.append({
                    'region': region,
                    'priority_score': priority_score,
                    'reserve_margin': region_risk.get('reserve_margin', 0),
                    'recommendations': root_causes.get('recommendations', []),
                    'root_causes': root_causes.get('root_causes', [])
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('priority_score', ascending=False)
        
        logger.info(f"Generated {len(recommendations)} investment recommendations")
        
        return recommendations_df
    
    def scenario_analysis(self,
                         forecast_data: pd.DataFrame,
                         supply_scenarios: Dict[str, float],
                         region_column: Optional[str] = 'region') -> Dict:
        """
        Analyze different supply scenarios
        
        Args:
            forecast_data: Forecasted demand data
            supply_scenarios: Dictionary of scenario names and supply multipliers
            region_column: Column with region identifiers (optional)
            
        Returns:
            Dictionary with scenario analysis results
        """
        scenarios = {}
        
        for scenario_name, supply_multiplier in supply_scenarios.items():
            scenario_df = forecast_data.copy()
            
            if 'supply' in scenario_df.columns:
                scenario_df['supply'] = scenario_df['supply'] * supply_multiplier
            else:
                logger.warning("Supply column not found in forecast data")
                continue
            
            # Calculate reserve margin for scenario
            risk_df = self.calculate_reserve_margin(
                'demand', 'supply', region_column
            )
            
            # Count at-risk regions
            at_risk_count = len(risk_df[risk_df['reserve_margin'] < 15])
            
            scenarios[scenario_name] = {
                'supply_multiplier': supply_multiplier,
                'at_risk_regions': at_risk_count,
                'avg_reserve_margin': risk_df['reserve_margin'].mean(),
                'min_reserve_margin': risk_df['reserve_margin'].min()
            }
        
        logger.info("Scenario analysis complete")
        
        return scenarios

