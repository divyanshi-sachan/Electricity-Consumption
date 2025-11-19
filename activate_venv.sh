#!/bin/bash
# Script to activate the virtual environment for Electricity Consumption project

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python: $(which python)"
echo "📦 Python version: $(python --version)"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""
echo "To start Jupyter Lab, run:"
echo "  jupyter lab"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

