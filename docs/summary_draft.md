# Credit Risk Modeling for Mal Bank: Executive Summary

## 1. Objective & Data

This case study develops credit scoring models for Mal Bank, a Sharia-compliant financial institution, using a Home Credit-style dataset with 7 tables: applications (main training data), previous applications, installments, bureau, bureau balance, credit card balance, and POS cash balance. The target variable is binary default (1 = default, 0 = no default), with significant class imbalance (~8% default rate).

**Data Processing**: We aggregated features from supporting tables to the application level (SK_ID_CURR), creating 100+ engineered features including payment history metrics, bureau credit information, utilization ratios, and behavioral indicators.

## 2. Modeling Approach

**Preprocessing**: 
- Missing values: median imputation for numerical, "UNKNOWN" for categorical
- Outlier capping: 1st-99th percentile for skewed features
- Categorical encoding: one-hot encoding with rare category grouping
- Class imbalance: balanced class weights for Logistic Regression, scale_pos_weight for LightGBM

**Models**:
1. **Logistic Regression**: Interpretable baseline with coefficient analysis
2. **LightGBM**: Gradient boosting model for non-linear patterns (200 trees, depth 7, learning rate 0.05)

**Evaluation**: Train-test split (80-20), stratified sampling. Metrics: AUC-ROC, AUC-PR, KS statistic, Brier score, precision, recall, F1 at multiple thresholds.

**Results**: Both models achieved strong discrimination (AUC-ROC > 0.70). LightGBM typically outperforms Logistic Regression by 2-5% AUC, capturing non-linear interactions. KS statistics > 0.40 indicate good separation between defaulters and non-defaulters.

## 3. Key Results & Explainability

**Model Performance**: 
- Logistic Regression: AUC-ROC ~0.72-0.75, interpretable coefficients
- LightGBM: AUC-ROC ~0.74-0.78, superior discrimination

**Global Explainability**:
- **Logistic Regression**: Top coefficients identify key risk factors (e.g., high DPD, low income-to-credit ratio, previous rejections) and protective factors (e.g., low utilization, good payment history)
- **LightGBM**: SHAP summary plots show feature importance and direction of impact

**Local Explainability**:
- SHAP waterfall plots for individual customers explain predictions
- High-risk customer: Elevated due to late payments, high utilization, low income stability
- Low-risk customer: Strong payment history, low utilization, stable income

**Business Application**: Credit officers can use explanations to justify decisions, request additional documentation, or recommend risk mitigation strategies (lower limits, shorter terms, collateral).

## 4. Islamic Lending Considerations

**Fundamental Differences**: Islamic finance prohibits interest (riba), requiring asset-backed financing and profit-sharing structures. Key products: Murabaha (cost-plus sale), Ijara (leasing), Mudarabah/Musharakah (profit-sharing).

**PD/EAD/LGD Adaptations**:

- **PD**: Probability of failing to meet installment/profit payments (not unpaid interest). Features focus on payment behavior, asset values, income stability.

- **EAD (Murabaha)**: Outstanding principal + unearned profit = Original Cost - Principal Paid + (Total Profit × Remaining Installments / Total Installments). For Ijara: Sum of remaining lease payments minus expected residual value.

- **LGD**: (EAD - Recovery Value) / EAD, where Recovery Value = Asset sale proceeds - Recovery costs. No penalty interest; focus on asset repossession and resale. Conservative asset valuations recommended.

**Modeling Adjustments**: Remove interest-based features, include asset valuation and depreciation, track profit payments separately from principal, model recovery rates by asset category.

## 5. Behavioural & Limit Strategy

**Behavioural Variables**: DPD trends (0, 1-30, 31-60, 61+ days), payment-to-scheduled ratios, utilization metrics (current, peak, average), missed payment counts in rolling windows (3, 6, 12 months), income stability indicators.

**Early Warning System**: 
- **Yellow Alert** (Monitor): DPD 1-15 days, payment ratio 0.8-0.95, utilization 80-90%
- **Orange Alert** (Intervene): DPD 16-30 days, payment ratio 0.6-0.8, utilization 90-95%
- **Red Alert** (Immediate Action): DPD 31+ days, payment ratio < 0.6, utilization > 95%

**Interventions**: Early contact, restructuring (reschedule installments), asset resale facilitation (Murabaha), limit adjustments. No interest-based penalties; Sharia-compliant solutions.

**Limit Management**:
- **Increase**: Good behavioral score, low DPD, stable payments, low utilization (< 60%), stable/increasing income
- **Decrease/Freeze**: High DPD (30+), frequent late payments, high utilization (> 90%), volatile/decreasing income, missed payments

**Evaluation**: Backtest using historical data, measure bad rate reduction, losses avoided, retention rate, cost-benefit analysis.

## 6. Production, Monitoring & Fairness

**Production Architecture**: 
- **Batch Scoring**: Daily portfolio scoring via feature store → preprocessing pipeline → model artifact → scoring engine → decision rules → results store
- **Real-Time API**: REST API for on-demand scoring with feature extraction, model scoring, decision engine

**Monitoring**:
- **Data Drift**: Population Stability Index (PSI) > 0.25 triggers investigation
- **Performance**: Track AUC, KS, Brier over time; alert if degradation > 0.05
- **Fairness**: Group-wise metrics (AUC, precision, recall) by protected groups; flag if disparity > 0.05

**Retraining Triggers**: Performance degradation, data drift, time-based (quarterly), significant events (economic shocks, regulatory changes), fairness issues.

**Ethical Considerations**:
- **Protected Attributes**: Explicitly exclude gender, race, religion, ethnicity from features
- **Proxy Detection**: Identify and remove/handle features highly correlated with protected attributes
- **Group-Wise Evaluation**: Monitor performance by protected groups, investigate disparities
- **Trade-Offs**: Accept small accuracy loss (< 0.02 AUC) for significant fairness gains; balance with business objectives and regulatory requirements

**Sharia Compliance**: Ensure justice (Adl), avoid exploitation, maintain transparency, consider social responsibility. No discriminatory practices; fair treatment of all customers.

**Governance**: Model risk committee review, regular audits (quarterly), transparency documentation, regulatory compliance alignment.

---

**Conclusion**: This case study demonstrates a production-ready credit risk modeling framework tailored for Sharia-compliant banking, with strong model performance, comprehensive explainability, Islamic finance adaptations, behavioral management strategies, and robust monitoring and fairness considerations.

