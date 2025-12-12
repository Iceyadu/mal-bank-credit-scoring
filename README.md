# Credit Risk Modeling for Mal Bank (Sharia-Compliant)

A comprehensive credit scoring case study for Mal Bank, implementing predictive models, Islamic lending considerations, behavioral management, production architecture, and ethical bias mitigation.

## Project Structure

```
DigitalBank/
├── data/                          # Data files (CSV)
├── notebooks/
│   └── credit_risk_model.ipynb    # Main Jupyter notebook
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loading.py            # Data loading functions
│   ├── feature_engineering.py     # Feature aggregation
│   ├── modeling.py                # Model training
│   ├── evaluation.py              # Model evaluation & metrics
│   ├── explainability.py          # SHAP & coefficient analysis
│   └── utils.py                   # Utility functions
├── models/                        # Saved model artifacts
├── plots/                         # Generated plots
├── docs/
│   └── summary_draft.md           # 2-page summary
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Setup** (Recommended):
   ```bash
   python test_setup.py
   ```
   This verifies packages, data files, and modules are ready.

3. **Data Location**:
   - Data files are in `../malbank_case_data/` (already configured in notebook)
   - Or place data files in `data/` directory and update notebook path

4. **Run Notebook**:
   ```bash
   jupyter notebook notebooks/credit_risk_model.ipynb
   ```
   
   Or use JupyterLab:
   ```bash
   jupyter lab notebooks/credit_risk_model.ipynb
   ```

## Quick Testing

See `TESTING_GUIDE.md` for detailed testing instructions, or run:

```bash
# Quick setup test
python test_setup.py

# If all tests pass, run the notebook
jupyter notebook notebooks/credit_risk_model.ipynb
```

## Project Components

### Part 1: Credit Scoring Model Development
- Data loading and exploration
- Feature engineering (aggregations from supporting tables)
- Two models: Logistic Regression (baseline) and LightGBM (advanced)
- Comprehensive evaluation metrics (AUC-ROC, AUC-PR, KS, Brier)
- Model explainability (coefficients, SHAP values)

### Part 2: Islamic Lending Context
- Differences between conventional and Islamic lending
- PD/EAD/LGD considerations for Murabaha, Ijara, Mudarabah, Musharakah
- Murabaha-specific modeling details

### Part 3: Behavioural & Limit Management
- Behavioral variables from transactional history
- Early warning system for delinquency
- Limit management framework (increase/decrease conditions)
- Evaluation strategy with backtesting

### Part 4: Production & Monitoring
- Production architecture (batch and real-time scoring)
- Data drift detection (PSI)
- Performance monitoring
- Fairness monitoring
- Retraining triggers and process

### Part 5: Ethical & Bias Considerations
- Non-discriminatory model practices
- Proxy variable detection
- Group-wise evaluation
- Fairness vs. accuracy trade-offs
- Sharia-compliant ethical considerations

## Key Features

- **Modular Code**: Clean, production-like code in `src/` modules
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Explainability**: SHAP values and coefficient analysis
- **Islamic Finance Context**: Tailored for Sharia-compliant banking
- **Production-Ready**: Architecture and monitoring considerations

## Outputs

- Trained models (Logistic Regression, LightGBM)
- Evaluation plots (ROC, PR curves, confusion matrices)
- SHAP plots (summary, waterfall)
- Model coefficients analysis
- Summary document (in `docs/summary_draft.md`)

## Notes

- The notebook is designed to be run end-to-end
- All plots are saved to `plots/` directory
- Models can be saved to `models/` directory for production use
- The summary document (`docs/credit_risk_summary.md`) can be exported to PDF for submission

## Author

Credit Risk Modeling Case Study for Mal Bank

