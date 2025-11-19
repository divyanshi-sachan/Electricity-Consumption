# Jupyter Notebook Setup Guide

## ✅ Quick Fix for "ModuleNotFoundError: No module named 'pandas'"

The error occurs because the notebook is not using the virtual environment kernel.

## Solution Steps:

### 1. **Start Jupyter from the Virtual Environment**

```bash
cd /Users/divyanshisachan/Downloads/Electricity-Consumption
source venv/bin/activate
jupyter notebook
```

### 2. **Select the Correct Kernel in Jupyter**

**In Jupyter Notebook:**
- Open your notebook: `notebooks/01_data_collection.ipynb`
- Click on **"Kernel"** in the menu bar
- Select **"Change Kernel"**
- Choose **"Python (Electricity Consumption)"**

**In JupyterLab:**
- Open your notebook
- Look at the top right corner for the kernel name
- Click on it and select **"Python (Electricity Consumption)"**

### 3. **Restart the Kernel**

After selecting the kernel:
- Go to **"Kernel"** → **"Restart Kernel"** (or click the restart button)
- Re-run your cells

### 4. **Verify the Kernel is Correct**

Run this in a cell to verify:
```python
import sys
print("Python executable:", sys.executable)
# Should show: .../Electricity-Consumption/venv/bin/python
```

## Common Issues:

### Issue: Still getting ModuleNotFoundError
**Solution:** 
1. Make sure you started Jupyter from the activated venv
2. Restart the Jupyter server completely
3. Close and reopen the notebook

### Issue: Kernel not showing in the list
**Solution:**
```bash
source venv/bin/activate
python -m ipykernel install --user --name=electricity-consumption --display-name="Python (Electricity Consumption)"
```

### Issue: Wrong pip install command
**Correct command:**
```bash
pip install -r requirements.txt  # Note the -r flag!
```

**Wrong command:**
```bash
pip install requirements.txt  # This tries to install a package named "requirements.txt"
```

## Quick Commands Reference:

```bash
# Activate virtual environment
source venv/bin/activate

# Install packages (correct syntax)
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab
jupyter lab

# List available kernels
jupyter kernelspec list
```

