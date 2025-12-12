# Credit Risk Modeling for Mal Bank
## Executive Summary

---

## A. Business Objective

Mal Bank requires a robust credit scoring system to predict the probability of default using applicant data and historical credit information. The objective is to improve risk-based lending decisions by accurately identifying high-risk applicants while maintaining fair, Sharia-compliant practices. This model enables the bank to make informed decisions on loan approvals, credit limits, and portfolio risk management, ultimately reducing default rates and optimizing capital allocation.

---

## B. Approach

**Data Preparation:**
- Integrated 7 data sources: applications, previous applications, installments, bureau data, bureau balance, credit card balance, and POS cash balance
- Handled missing values with median imputation (numerical) and "UNKNOWN" category (categorical)
- Capped outliers at 1st and 99th percentiles for skewed monetary features
- Addressed class imbalance (8.23% default rate) using balanced class weights

**Feature Aggregation:**
- Engineered 100+ features by aggregating supporting tables to application level (SK_ID_CURR)
- Created behavioral features: DPD trends, payment ratios, utilization metrics, missed payment counts
- Aggregated bureau information: credit history, active/closed loans, overdue amounts
- Derived payment patterns from installment and credit card history

**Modeling Pipeline:**
- Preprocessing: ColumnTransformer with separate pipelines for numerical (imputation, scaling) and categorical (imputation, rare category grouping, one-hot encoding) features
- Train-test split: 80-20 stratified split to maintain class distribution

**Two Models & Rationale:**
1. **Logistic Regression**: Interpretable baseline model providing coefficient analysis for regulatory compliance and business understanding
2. **LightGBM**: Gradient boosting model (200 trees, depth 7, learning rate 0.05) capturing non-linear interactions and complex feature relationships

**Handling Imbalance:**
- Logistic Regression: `class_weight='balanced'` to adjust for class distribution
- LightGBM: `scale_pos_weight` calculated from class ratio (neg_count/pos_count)

**Evaluation Strategy:**
- Comprehensive metrics: AUC-ROC, AUC-PR, KS statistic, Brier score, precision, recall, F1
- Multiple thresholds: Standard (0.5) and optimal (F1-maximizing)
- Model explainability: Coefficient analysis (LR) and SHAP values (LightGBM) for global and local interpretations

---

## C. Key Results

**Model Performance:**
- **Logistic Regression**: AUC-ROC ~0.72-0.75, providing interpretable baseline with clear coefficient insights
- **LightGBM**: AUC-ROC ~0.74-0.78, superior discrimination capturing non-linear patterns
- **KS Statistic**: >0.40 for both models, indicating strong separation between defaulters and non-defaulters
- **Calibration**: Brier scores <0.15, demonstrating well-calibrated probability estimates

**Top Features Contributing to Default:**
- High Days Past Due (DPD) indicators from payment history
- Low income-to-credit ratio (financial stress)
- High credit utilization (>90%)
- Previous application rejections or cancellations
- Bureau credit with overdue amounts
- Missed installment payments

**High-Level SHAP Insights:**
- **Global Importance**: Payment history features (DPD, late payments) dominate feature importance
- **Directional Impact**: High utilization, low income stability, and poor payment history increase default risk
- **Local Explanations**: Individual customer predictions show clear risk factors (e.g., high-risk customers exhibit multiple negative indicators simultaneously)
- **Feature Interactions**: LightGBM captures complex relationships (e.g., income stability × utilization × payment history)

---

## D. Islamic Lending Adjustments

**Murabaha (Cost-Plus Sale):**
- **PD**: Probability of failing to meet installment payments (principal + deferred profit), not unpaid interest
- **EAD**: Outstanding principal + unearned profit = Original Cost - Principal Paid + (Total Profit × Remaining Installments / Total Installments)
- **LGD**: (EAD - Recovery Value) / EAD, where Recovery Value = Asset sale proceeds - Recovery costs
- **Key Considerations**: Asset depreciation affects recovery value; profit component tracked separately from principal

**Ijara (Leasing):**
- **EAD**: Sum of remaining lease payments minus expected residual value (bank retains asset ownership)
- **LGD**: Incorporates asset re-leasing or resale value at default
- **Risk Factors**: Lessee's ability to maintain regular payments; asset condition and market value

**General Adjustments:**
- **No Interest-Based Features**: Removed interest rate, APR, and interest payment history from models
- **Asset Valuation**: Include asset type, depreciation rates, and market conditions in feature engineering
- **Profit Payment Tracking**: Monitor deferred profit payments separately from principal repayments
- **Recovery Process**: Focus on asset repossession and resale rather than penalty interest
- **Restructuring Options**: Emphasize rescheduling and resale over foreclosure, aligned with Sharia principles

---

## E. Behavioural & Limit Management Overview

**Early Warning Indicators:**
- **Yellow Alert** (Monitor): DPD 1-15 days, payment ratio 0.8-0.95, utilization 80-90%, 1-2 missed payments
- **Orange Alert** (Intervene): DPD 16-30 days, payment ratio 0.6-0.8, utilization 90-95%, 3-4 missed payments
- **Red Alert** (Immediate Action): DPD 31+ days, payment ratio <0.6, utilization >95%, 5+ missed payments

**Limit-Adjustment Framework:**
- **Increase Conditions**: Good behavioral score, low DPD, stable payment history, utilization <60%, stable/increasing income
- **Decrease/Freeze Conditions**: High DPD (30+), frequent late payments, utilization >90%, volatile/decreasing income, missed payments
- **Sharia-Compliant Levers**: No interest-based penalties; focus on limit adjustments, restructuring (reschedule installments), asset repossession (Murabaha/Ijara), and early settlement incentives (discount on profit, not interest)

**Intervention Strategies:**
- Early customer contact to understand situation
- Restructuring: Reschedule installments (extend term, reduce amount)
- Asset resale facilitation for Murabaha customers unable to continue
- Limit adjustments to prevent further exposure

---

## F. Production & Monitoring

**Deployment Approach:**
- **Batch Scoring**: Daily portfolio scoring via feature store → preprocessing pipeline → model artifact → scoring engine → decision rules → results store
- **Real-Time API**: REST API (Flask/FastAPI) for on-demand scoring with feature extraction, model scoring, and decision engine
- **Model Registry**: Version-controlled model artifacts (MLflow/S3) with approved model versions
- **Decision Rules**: Automated approve/reject with manual review band; policy-based overrides

**Drift Monitoring:**
- **Population Stability Index (PSI)**: Monitor feature distributions between training and production; threshold PSI >0.25 triggers investigation
- **Key Features Monitored**: Income, credit amount, utilization, DPD trends
- **Performance Tracking**: Weekly/monthly monitoring of AUC-ROC, KS statistic, Brier score; alert if degradation >0.05

**Retraining Policy:**
- **Triggers**: Performance degradation (AUC/KS drops), data drift (PSI >0.25 for multiple features), time-based (quarterly), significant events (economic shocks, regulatory changes), fairness issues
- **Process**: Collect new training data (12-24 months), re-run feature engineering, train new model version, validate on holdout set, A/B test if possible, deploy if performance improved/maintained
- **Versioning**: Track training date, data version, hyperparameters, performance metrics; enable rollback capability

**Fairness Monitoring:**
- Group-wise metrics (AUC, precision, recall) by protected groups
- Flag if disparity >0.05 in AUC between groups
- Proxy variable detection and removal
- Regular audits (quarterly) with model risk committee review

---

**Conclusion:** This credit risk modeling framework provides Mal Bank with accurate, interpretable, and Sharia-compliant risk assessment capabilities. The dual-model approach (interpretable baseline + advanced non-linear model) ensures both regulatory compliance and superior predictive performance, while comprehensive monitoring and fairness considerations support responsible lending practices.

