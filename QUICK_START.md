# Quick Start Guide - Running the Notebook

## Step 1: Install Dependencies

Make sure you have all required packages installed:

```bash
cd /Users/mac/Documents/DigitalBank
pip install -r requirements.txt
```

Or install individually if needed:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap scipy jupyter
```

## Step 2: Start Jupyter

### Option A: Jupyter Notebook (Classic Interface)

```bash
cd /Users/mac/Documents/DigitalBank
jupyter notebook
```

This will:
- Open Jupyter in your web browser
- Show the project directory
- Click on `notebooks/credit_risk_model.ipynb` to open it

### Option B: JupyterLab (Modern Interface)

```bash
cd /Users/mac/Documents/DigitalBank
jupyter lab
```

Then navigate to `notebooks/credit_risk_model.ipynb`

### Option C: Direct Open

```bash
cd /Users/mac/Documents/DigitalBank
jupyter notebook notebooks/credit_risk_model.ipynb
```

## Step 3: Run the Notebook

Once the notebook is open:

### Method 1: Run All Cells (Recommended for first run)
- Go to menu: **Cell → Run All**
- Or use keyboard shortcut: **Shift + Enter** (runs current cell and moves to next)

### Method 2: Run Cells One by One
- Click on a cell
- Press **Shift + Enter** to run it
- Continue through all cells sequentially

### Method 3: Run Selected Cells
- Select multiple cells (Shift + Click)
- Press **Shift + Enter**

## Step 4: Monitor Progress

The notebook will:
1. Load data (~10-30 seconds)
2. Build features (~2-5 minutes)
3. Train models (~3-7 minutes total)
4. Generate plots (~1-2 minutes)
5. Compute SHAP values (~5-10 minutes)

**Total runtime: ~15-25 minutes**

## Step 5: Check Outputs

After completion, verify:

1. **Plots Directory**: Check `plots/` folder for generated visualizations
2. **Console Output**: Review printed metrics (AUC-ROC, KS, etc.)
3. **Notebook Cells**: All cells should show execution numbers (e.g., `[1]`, `[2]`)

## Troubleshooting

### Issue: Kernel Not Starting
```bash
# Install ipykernel
pip install ipykernel

# Or restart Jupyter
jupyter notebook --generate-config
```

### Issue: Module Not Found
- Make sure you're in the project root directory
- The notebook adds `../src` to the path automatically
- If issues persist, check the first cell imports

### Issue: Out of Memory
- Reduce data size in the notebook:
  ```python
  apps = apps.sample(n=10000, random_state=42)  # Use smaller sample
  ```

### Issue: SHAP Takes Too Long
- Reduce sample size in SHAP cells:
  ```python
  X_test.sample(100, random_state=42)  # Instead of 500
  n_samples=50  # Instead of 100
  ```

## Keyboard Shortcuts

- **Shift + Enter**: Run cell and move to next
- **Ctrl + Enter**: Run cell and stay
- **Esc**: Command mode
- **A**: Insert cell above
- **B**: Insert cell below
- **DD**: Delete cell (press D twice)

## Expected Results

After successful run, you should see:
- ✓ Model metrics printed (AUC-ROC > 0.70)
- ✓ 8+ plots in `plots/` directory
- ✓ Feature table with 100+ features
- ✓ Two trained models ready for use

## Next Steps

1. Review the generated plots in `plots/` directory
2. Check model performance metrics
3. Review the summary document: `docs/summary_draft.md`
4. Export notebook to PDF if needed for submission

