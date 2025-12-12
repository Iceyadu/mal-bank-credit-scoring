# Testing Guide for Credit Risk Modeling Project

## Quick Start Testing

### 1. Test Setup (Recommended First Step)

Run the setup test script to verify everything is ready:

```bash
cd /Users/mac/Documents/DigitalBank
python test_setup.py
```

This will check:
- ✓ All required packages are installed
- ✓ Data files exist and are accessible
- ✓ Source modules can be imported
- ✓ Data can be loaded successfully

### 2. Install Dependencies

If packages are missing:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap scipy jupyter
```

### 3. Test Individual Components

#### Test Data Loading
```python
# In Python or Jupyter
import sys
sys.path.append('src')
from data_loading import load_all_data

data = load_all_data("../malbank_case_data")
print(f"Loaded {len(data)} tables")
print(data['applications'].shape)
```

#### Test Feature Engineering
```python
from feature_engineering import build_feature_table
import pandas as pd

# Load a small sample
apps = pd.read_csv("../malbank_case_data/applications.csv", nrows=100)
prev_apps = pd.read_csv("../malbank_case_data/previous_applications.csv")

# Test aggregation
features = build_feature_table(apps, prev_apps=prev_apps)
print(f"Feature table shape: {features.shape}")
```

#### Test Modeling Pipeline
```python
from modeling import get_preprocessing_pipeline
import pandas as pd

# Create dummy data
X_train = pd.DataFrame({
    'numeric_col': [1, 2, 3, 4, 5],
    'categorical_col': ['A', 'B', 'A', 'C', 'B']
})

preprocessing = get_preprocessing_pipeline(
    X_train,
    categorical_cols=['categorical_col'],
    numerical_cols=['numeric_col']
)

print("Preprocessing pipeline created successfully!")
```

## Running the Full Notebook

### Option 1: Jupyter Notebook (Recommended)

1. **Start Jupyter**:
   ```bash
   cd /Users/mac/Documents/DigitalBank
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `notebooks/credit_risk_model.ipynb`
   - Click to open

3. **Run cells sequentially**:
   - Click "Run All" or run cells one by one (Shift+Enter)
   - The notebook is designed to run end-to-end

### Option 2: JupyterLab

```bash
jupyter lab notebooks/credit_risk_model.ipynb
```

### Option 3: VS Code / Cursor

1. Open the notebook file in VS Code/Cursor
2. Install Jupyter extension if needed
3. Run cells using the play button or Shift+Enter

## Expected Runtime

- **Data Loading**: ~10-30 seconds
- **Feature Engineering**: ~2-5 minutes (depending on data size)
- **Model Training**: 
  - Logistic Regression: ~1-2 minutes
  - LightGBM: ~2-5 minutes
- **Evaluation & Plots**: ~1-2 minutes
- **SHAP Analysis**: ~5-10 minutes (most time-consuming)

**Total**: ~15-25 minutes for full execution

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**:
```bash
# Make sure you're in the project root
cd /Users/mac/Documents/DigitalBank

# Add src to Python path in notebook
import sys
sys.path.append('../src')  # or 'src' if running from notebooks/
```

### Issue: Data File Not Found

**Solution**: Update the data directory path in the notebook:
```python
# Option 1: If data is in ../malbank_case_data/
data_dir = "../malbank_case_data"

# Option 2: If data is in ../data/
data_dir = "../data"

# Option 3: Absolute path
data_dir = "/Users/mac/Documents/DigitalBank/malbank_case_data"
```

### Issue: Memory Error

**Solution**: 
- Reduce data size for testing:
  ```python
  apps = apps.sample(n=10000, random_state=42)  # Use smaller sample
  ```
- Or increase system memory / use cloud environment

### Issue: SHAP Takes Too Long

**Solution**: Reduce sample size in notebook:
```python
plot_shap_summary(
    model_lgb,
    X_test.sample(100, random_state=42),  # Reduce from 500 to 100
    n_samples=50  # Reduce from 100 to 50
)
```

### Issue: Plot Style Error

**Solution**: The notebook already handles this with try/except, but if issues persist:
```python
# Use default style
plt.style.use('default')
```

### Issue: LightGBM Warnings

**Solution**: These are usually harmless. To suppress:
```python
import warnings
warnings.filterwarnings('ignore')
```

## Testing Specific Parts

### Test Only Data Loading
```python
# In notebook, run only cells 1-4
# Stop after data loading section
```

### Test Only Feature Engineering
```python
# Run cells 1-7
# Check feature_df shape and columns
```

### Test Only Model Training
```python
# Skip SHAP (time-consuming)
# Run cells 1-20 (stop before SHAP section)
```

### Test Only Evaluation
```python
# Load pre-trained models (if saved)
# Run evaluation cells only
```

## Verifying Outputs

After running the notebook, check:

1. **Plots Directory**: 
   ```bash
   ls -lh plots/
   ```
   Should contain:
   - `roc_logreg.png`, `roc_lgbm.png`
   - `prc_logreg.png`, `prc_lgbm.png`
   - `confusion_logreg.png`, `confusion_lgbm.png`
   - `coefficients_logreg.png`
   - `shap_summary_lgbm.png`
   - `shap_waterfall_high_risk.png`
   - `shap_waterfall_low_risk.png`

2. **Model Metrics**: Check printed output for:
   - AUC-ROC > 0.70 (good performance)
   - KS Statistic > 0.40 (good discrimination)
   - Reasonable precision/recall

3. **Feature Engineering**: 
   - Feature count should increase significantly
   - No excessive missing values in aggregated features

## Quick Smoke Test

Minimal test to verify everything works:

```python
# Quick smoke test script
import sys
sys.path.append('src')

from data_loading import load_all_data
from feature_engineering import build_feature_table
from modeling import get_preprocessing_pipeline, train_logistic_regression
from sklearn.model_selection import train_test_split

# Load data
data = load_all_data("../malbank_case_data")
apps = data['applications'].sample(1000, random_state=42)  # Small sample

# Build features (minimal)
feature_df = build_feature_table(apps, prev_apps=data['previous_applications'])

# Prepare
X = feature_df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = feature_df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
from utils import get_column_types
cat_cols, num_cols = get_column_types(X_train)
preprocessing = get_preprocessing_pipeline(X_train, cat_cols, num_cols)
model = train_logistic_regression(X_train, y_train, preprocessing)

# Evaluate
from evaluation import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
print("✓ Smoke test passed!")
```

## Next Steps

Once testing passes:
1. Run full notebook end-to-end
2. Review generated plots in `plots/` directory
3. Check model performance metrics
4. Review summary document in `docs/summary_draft.md`
5. Export notebook to PDF if needed for submission

