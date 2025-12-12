# Models Directory

This directory stores trained model artifacts.

## Model Files

After running the notebook, you should see:

- `model_logistic_regression.pkl` - Trained Logistic Regression model
- `model_lightgbm.pkl` - Trained LightGBM model
- `model_logistic_regression.metadata.pkl` - Metadata for Logistic Regression
- `model_lightgbm.metadata.pkl` - Metadata for LightGBM

## Loading Models

To load a saved model in Python:

```python
from src.model_persistence import load_model

# Load model
model = load_model('models/model_lightgbm.pkl')

# Use for predictions
predictions = model.predict_proba(X_new)[:, 1]
```

## Model Metadata

Each model includes metadata with:
- Model type and training date
- Number of features and samples
- Test set performance metrics (AUC-ROC, AUC-PR, KS, Brier)
- Hyperparameters (for LightGBM)

## Note

Models are automatically saved after training when you run the notebook. If the directory is empty, run the notebook first to train and save the models.

