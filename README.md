# Mining Electricity Consumption Data for Resource Allocation and Supply Chain Enhancement

## Project Overview
This project analyzes electricity consumption data to provide data-driven insights for infrastructure investment, distribution optimization, and supply chain enhancement. The analysis focuses on identifying demand patterns, forecasting future needs, and recommending strategic investments in grid infrastructure.

## Team Members
- Divyanshi Sachan
- Soniya Malviya
- Aryan Soni

## Research Questions

1. **Temporal and Regional Variations**: How does electricity usage vary by region, time of day, and season?
2. **Demand Spikes**: Are there recurring demand spikes that could stress the grid (e.g., during summer heatwaves or festivals)?
3. **Sector Analysis**: How do residential vs. commercial vs. industrial users differ in consumption?
4. **Forecasting**: Can we forecast electricity demand for the next 5–10 years in specific regions?
5. **Risk Assessment**: Which areas are at risk of under-supply if current trends continue?

## Dataset
Primary data source: [EIA Open Data API](https://www.eia.gov/opendata/browser/electricity)

## Project Structure
```
data-mining/
├── data/
│   ├── raw/              # Raw data from EIA
│   ├── processed/        # Cleaned data
│   └── external/         # Additional data sources
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code modules
│   ├── data_collection/  # EIA API integration
│   ├── data_processing/  # Data cleaning and preprocessing
│   ├── analysis/         # Exploratory and predictive analysis
│   └── visualization/    # Heatmaps and visualizations
├── research/             # Research documentation (HIGH PRIORITY)
│   ├── literature_review/    # Comprehensive literature review
│   ├── hypotheses/            # Research hypotheses and testing
│   ├── methodology/           # Research methodology and theory
│   ├── references/            # Bibliography and citations
│   ├── research_questions/     # Research questions documentation
│   ├── innovation/            # Innovation and novel contributions
│   └── research_log/           # Weekly research notes
├── docs/                 # General documentation
├── tests/                # Unit tests
└── app/                  # Interactive dashboard
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-mining
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Add your EIA API key to .env file
   # Get API key from: https://www.eia.gov/opendata/register.php
   ```

5. **Run data collection**
   ```bash
   python src/data_collection/eia_api.py
   ```

6. **Start Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

## Methodology
See `docs/methodology.md` for detailed methodology and analysis philosophy.

## Research Documentation
See `research/README.md` for comprehensive research documentation including:
- **Literature Review**: 20-30+ academic papers and reports
- **Hypotheses**: Novel, well-founded research hypotheses
- **Methodology**: Theoretical foundation and research design
- **Innovation**: Novel contributions and insights
- **References**: Complete bibliography

**Note**: Research documentation has maximum weightage (40 marks combined) in evaluation criteria.

## Key Deliverables

- ✅ Interactive geographical heatmap showing demand/supply by region
- ✅ Predictive analysis with 5-10 year forecasts
- ✅ Risk assessment identifying under-supply areas
- ✅ Root cause analysis and realistic recommendations
- ✅ Comprehensive documentation of findings

## Timeline

- **Week 1**: Data collection and initial exploration
- **Week 2**: Data preprocessing and exploratory analysis
- **Week 3**: Predictive modeling and risk analysis
- **Week 4**: Visualization, documentation, and finalization

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Commit with clear messages
4. Push and create a pull request


