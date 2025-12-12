"""
Feature engineering functions for credit risk modeling.
Aggregates supporting tables to application level.
"""

import pandas as pd
import numpy as np
from typing import Optional


def aggregate_previous_applications(prev_apps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous applications to application level.
    
    Args:
        prev_apps: DataFrame with previous applications
        
    Returns:
        Aggregated features per SK_ID_CURR
    """
    features = []
    
    # Count of previous applications
    features.append(prev_apps.groupby('SK_ID_CURR').size().rename('PREV_APP_COUNT'))
    
    # Contract status aggregations
    if 'NAME_CONTRACT_STATUS' in prev_apps.columns:
        status_counts = prev_apps.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack(fill_value=0)
        status_counts.columns = [f'PREV_STATUS_{col}' for col in status_counts.columns]
        features.append(status_counts)
        
        # Ratio of approved vs refused
        if 'Approved' in status_counts.columns and 'Refused' in status_counts.columns:
            features.append(
                (status_counts['PREV_STATUS_Approved'] / 
                 (status_counts['PREV_STATUS_Refused'] + 1)).rename('PREV_APPROVED_RATIO')
            )
    
    # Average previous loan amounts
    if 'AMT_CREDIT' in prev_apps.columns:
        features.append(prev_apps.groupby('SK_ID_CURR')['AMT_CREDIT'].mean().rename('PREV_AMT_CREDIT_MEAN'))
        features.append(prev_apps.groupby('SK_ID_CURR')['AMT_CREDIT'].sum().rename('PREV_AMT_CREDIT_SUM'))
    
    if 'AMT_ANNUITY' in prev_apps.columns:
        features.append(prev_apps.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean().rename('PREV_AMT_ANNUITY_MEAN'))
    
    # Average term (if available)
    if 'CNT_PAYMENT' in prev_apps.columns:
        features.append(prev_apps.groupby('SK_ID_CURR')['CNT_PAYMENT'].mean().rename('PREV_CNT_PAYMENT_MEAN'))
    
    # Days decision (time to decision)
    if 'DAYS_DECISION' in prev_apps.columns:
        features.append(prev_apps.groupby('SK_ID_CURR')['DAYS_DECISION'].mean().rename('PREV_DAYS_DECISION_MEAN'))
    
    # Combine all features
    result = pd.concat(features, axis=1)
    result = result.fillna(0)
    
    return result


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installments payment history to application level.
    
    Args:
        inst: DataFrame with installment payments
        
    Returns:
        Aggregated features per SK_ID_CURR
    """
    features = []
    
    # Calculate Days Past Due (DPD)
    if 'DAYS_INSTALMENT' in inst.columns and 'DAYS_ENTRY_PAYMENT' in inst.columns:
        inst = inst.copy()
        inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
        inst['DPD'] = inst['DPD'].clip(lower=0)  # Only positive delays
        
        # Share of installments paid late
        late_payments = (inst['DPD'] > 0).groupby(inst['SK_ID_CURR']).sum()
        total_payments = inst.groupby('SK_ID_CURR').size()
        features.append((late_payments / (total_payments + 1)).rename('INST_LATE_PAYMENT_RATIO'))
        
        # Maximum DPD
        features.append(inst.groupby('SK_ID_CURR')['DPD'].max().rename('INST_MAX_DPD'))
        
        # Average DPD
        features.append(inst.groupby('SK_ID_CURR')['DPD'].mean().rename('INST_AVG_DPD'))
        
        # Std dev of delays
        features.append(inst.groupby('SK_ID_CURR')['DPD'].std().fillna(0).rename('INST_DPD_STD'))
    
    # Number of missed installments (DPD > 30)
    if 'DPD' in inst.columns:
        features.append((inst['DPD'] > 30).groupby(inst['SK_ID_CURR']).sum().rename('INST_MISSED_COUNT'))
    
    # Payment amounts
    if 'AMT_PAYMENT' in inst.columns:
        features.append(inst.groupby('SK_ID_CURR')['AMT_PAYMENT'].mean().rename('INST_AMT_PAYMENT_MEAN'))
        features.append(inst.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum().rename('INST_AMT_PAYMENT_SUM'))
    
    if 'AMT_INSTALMENT' in inst.columns:
        features.append(inst.groupby('SK_ID_CURR')['AMT_INSTALMENT'].mean().rename('INST_AMT_INSTALMENT_MEAN'))
        
        # Payment ratio
        if 'AMT_PAYMENT' in inst.columns:
            payment_ratio = (inst.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum() / 
                           (inst.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum() + 1))
            features.append(payment_ratio.rename('INST_PAYMENT_RATIO'))
    
    # Combine all features
    if features:
        result = pd.concat(features, axis=1)
        result = result.fillna(0)
    else:
        result = pd.DataFrame(index=inst['SK_ID_CURR'].unique())
    
    return result


def aggregate_bureau(bureau: pd.DataFrame, bureau_bal: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate credit bureau data to application level.
    
    Args:
        bureau: DataFrame with bureau credit records
        bureau_bal: Optional DataFrame with bureau balance history
        
    Returns:
        Aggregated features per SK_ID_CURR
    """
    features = []
    
    # Number of bureau loans
    features.append(bureau.groupby('SK_ID_CURR').size().rename('BUREAU_LOAN_COUNT'))
    
    # Credit active status
    if 'CREDIT_ACTIVE' in bureau.columns:
        active_counts = bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).size().unstack(fill_value=0)
        active_counts.columns = [f'BUREAU_STATUS_{col}' for col in active_counts.columns]
        features.append(active_counts)
    
    # Credit amounts
    if 'AMT_CREDIT_SUM' in bureau.columns:
        features.append(bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum().rename('BUREAU_AMT_CREDIT_SUM'))
        features.append(bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean().rename('BUREAU_AMT_CREDIT_MEAN'))
    
    if 'AMT_CREDIT_SUM_DEBT' in bureau.columns:
        features.append(bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum().rename('BUREAU_AMT_DEBT_SUM'))
    
    if 'AMT_CREDIT_SUM_OVERDUE' in bureau.columns:
        features.append(bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum().rename('BUREAU_AMT_OVERDUE_SUM'))
        # Flag for any overdue
        features.append((bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum() > 0)
                       .astype(int).rename('BUREAU_HAS_OVERDUE'))
    
    if 'CREDIT_DAY_OVERDUE' in bureau.columns:
        features.append(bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max().rename('BUREAU_MAX_OVERDUE_DAYS'))
        features.append((bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max() > 0)
                       .astype(int).rename('BUREAU_HAS_OVERDUE_DAYS'))
    
    # Days credit (how old is the credit)
    if 'DAYS_CREDIT' in bureau.columns:
        features.append(bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].min().rename('BUREAU_OLDEST_CREDIT'))
        features.append(bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].mean().rename('BUREAU_AVG_CREDIT_AGE'))
    
    # Credit type
    if 'CREDIT_TYPE' in bureau.columns:
        credit_type_counts = bureau.groupby(['SK_ID_CURR', 'CREDIT_TYPE']).size().unstack(fill_value=0)
        credit_type_counts.columns = [f'BUREAU_TYPE_{col}' for col in credit_type_counts.columns]
        features.append(credit_type_counts)
    
    # Bureau balance aggregations
    if bureau_bal is not None and not bureau_bal.empty:
        # Merge with bureau to get SK_ID_CURR
        if 'SK_ID_BUREAU' in bureau_bal.columns and 'SK_ID_BUREAU' in bureau.columns:
            bureau_bal_with_curr = bureau_bal.merge(
                bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates(),
                on='SK_ID_BUREAU',
                how='left'
            )
            
            if 'STATUS' in bureau_bal_with_curr.columns:
                # Count by status
                status_counts = bureau_bal_with_curr.groupby(['SK_ID_CURR', 'STATUS']).size().unstack(fill_value=0)
                status_counts.columns = [f'BUREAU_BAL_STATUS_{col}' for col in status_counts.columns]
                features.append(status_counts)
    
    # Combine all features
    if features:
        result = pd.concat(features, axis=1)
        result = result.fillna(0)
    else:
        result = pd.DataFrame(index=bureau['SK_ID_CURR'].unique())
    
    return result


def aggregate_credit_card_balance(cc_bal: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit card balance history to application level.
    
    Args:
        cc_bal: DataFrame with credit card balance history
        
    Returns:
        Aggregated features per SK_ID_CURR
    """
    features = []
    
    # Average balance
    if 'AMT_BALANCE' in cc_bal.columns:
        features.append(cc_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].mean().rename('CC_AVG_BALANCE'))
        features.append(cc_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].max().rename('CC_MAX_BALANCE'))
    
    # Credit limit utilization
    if 'AMT_CREDIT_LIMIT_ACTUAL' in cc_bal.columns and 'AMT_BALANCE' in cc_bal.columns:
        cc_bal = cc_bal.copy()
        cc_bal['UTILIZATION'] = cc_bal['AMT_BALANCE'] / (cc_bal['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        features.append(cc_bal.groupby('SK_ID_CURR')['UTILIZATION'].mean().rename('CC_AVG_UTILIZATION'))
        features.append(cc_bal.groupby('SK_ID_CURR')['UTILIZATION'].max().rename('CC_MAX_UTILIZATION'))
    
    # Overdue amounts
    if 'AMT_RECEIVABLE_PRINCIPAL' in cc_bal.columns:
        features.append(cc_bal.groupby('SK_ID_CURR')['AMT_RECEIVABLE_PRINCIPAL'].sum()
                       .rename('CC_TOTAL_RECEIVABLE'))
    
    # Balance volatility (std dev)
    if 'AMT_BALANCE' in cc_bal.columns:
        features.append(cc_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].std().fillna(0)
                       .rename('CC_BALANCE_STD'))
    
    # Combine all features
    if features:
        result = pd.concat(features, axis=1)
        result = result.fillna(0)
    else:
        result = pd.DataFrame(index=cc_bal['SK_ID_CURR'].unique())
    
    return result


def aggregate_pos_cash_balance(pos_bal: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate POS cash balance history to application level.
    
    Args:
        pos_bal: DataFrame with POS cash balance history
        
    Returns:
        Aggregated features per SK_ID_CURR
    """
    features = []
    
    # Count of POS loans
    if 'SK_ID_PREV' in pos_bal.columns:
        features.append(pos_bal.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().rename('POS_LOAN_COUNT'))
    
    # Name contract status
    if 'NAME_CONTRACT_STATUS' in pos_bal.columns:
        status_counts = pos_bal.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack(fill_value=0)
        status_counts.columns = [f'POS_STATUS_{col}' for col in status_counts.columns]
        features.append(status_counts)
    
    # DPD indicators
    if 'SK_DPD' in pos_bal.columns:
        features.append(pos_bal.groupby('SK_ID_CURR')['SK_DPD'].max().rename('POS_MAX_DPD'))
        features.append(pos_bal.groupby('SK_ID_CURR')['SK_DPD'].mean().rename('POS_AVG_DPD'))
        features.append((pos_bal.groupby('SK_ID_CURR')['SK_DPD'].max() > 0)
                       .astype(int).rename('POS_HAS_DPD'))
    
    # Remaining installments
    if 'CNT_INSTALMENT' in pos_bal.columns and 'CNT_INSTALMENT_FUTURE' in pos_bal.columns:
        pos_bal = pos_bal.copy()
        pos_bal['REMAINING'] = pos_bal['CNT_INSTALMENT'] - pos_bal['CNT_INSTALMENT_FUTURE']
        features.append(pos_bal.groupby('SK_ID_CURR')['REMAINING'].mean().rename('POS_AVG_REMAINING'))
    
    # Combine all features
    if features:
        result = pd.concat(features, axis=1)
        result = result.fillna(0)
    else:
        result = pd.DataFrame(index=pos_bal['SK_ID_CURR'].unique())
    
    return result


def build_feature_table(
    apps: pd.DataFrame,
    prev_apps: Optional[pd.DataFrame] = None,
    inst: Optional[pd.DataFrame] = None,
    bureau: Optional[pd.DataFrame] = None,
    bureau_bal: Optional[pd.DataFrame] = None,
    cc_bal: Optional[pd.DataFrame] = None,
    pos_bal: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build complete feature table by aggregating all supporting tables.
    
    Args:
        apps: Main applications dataframe
        prev_apps: Previous applications dataframe
        inst: Installments dataframe
        bureau: Bureau dataframe
        bureau_bal: Bureau balance dataframe
        cc_bal: Credit card balance dataframe
        pos_bal: POS cash balance dataframe
        
    Returns:
        Complete feature table keyed by SK_ID_CURR
    """
    # Start with applications
    feature_df = apps.copy()
    
    # Aggregate and merge previous applications
    if prev_apps is not None and not prev_apps.empty:
        prev_features = aggregate_previous_applications(prev_apps)
        feature_df = feature_df.merge(prev_features, left_on='SK_ID_CURR', right_index=True, how='left')
    
    # Aggregate and merge installments
    if inst is not None and not inst.empty:
        inst_features = aggregate_installments(inst)
        feature_df = feature_df.merge(inst_features, left_on='SK_ID_CURR', right_index=True, how='left')
    
    # Aggregate and merge bureau
    if bureau is not None and not bureau.empty:
        bureau_features = aggregate_bureau(bureau, bureau_bal)
        feature_df = feature_df.merge(bureau_features, left_on='SK_ID_CURR', right_index=True, how='left')
    
    # Aggregate and merge credit card balance
    if cc_bal is not None and not cc_bal.empty:
        cc_features = aggregate_credit_card_balance(cc_bal)
        feature_df = feature_df.merge(cc_features, left_on='SK_ID_CURR', right_index=True, how='left')
    
    # Aggregate and merge POS cash balance
    if pos_bal is not None and not pos_bal.empty:
        pos_features = aggregate_pos_cash_balance(pos_bal)
        feature_df = feature_df.merge(pos_features, left_on='SK_ID_CURR', right_index=True, how='left')
    
    # Fill NaN values from aggregations with 0 (no history = 0)
    agg_cols = [col for col in feature_df.columns if col not in apps.columns]
    feature_df[agg_cols] = feature_df[agg_cols].fillna(0)
    
    return feature_df

