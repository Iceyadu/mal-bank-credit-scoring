"""
Utility functions for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def identify_target_column(df: pd.DataFrame, possible_names: List[str] = None) -> Optional[str]:
    """
    Identify the target column in the applications dataframe.
    
    Args:
        df: DataFrame to search
        possible_names: List of possible target column names
        
    Returns:
        Name of target column or None if not found
    """
    if possible_names is None:
        possible_names = ["TARGET", "target", "default", "Default", "y"]
    
    for col in possible_names:
        if col in df.columns:
            return col
    
    return None


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and ID columns from numerical if present
    id_cols = [col for col in numerical if 'ID' in col.upper() or col.upper() == 'SK_ID_CURR']
    numerical = [col for col in numerical if col not in id_cols]
    
    return categorical, numerical


def cap_outliers(series: pd.Series, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> pd.Series:
    """
    Cap outliers at specified percentiles.
    
    Args:
        series: Series to cap
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold
        
    Returns:
        Series with capped values
    """
    lower = series.quantile(lower_percentile / 100.0)
    upper = series.quantile(upper_percentile / 100.0)
    return series.clip(lower=lower, upper=upper)


def calculate_class_weights(y: pd.Series) -> dict:
    """
    Calculate class weights for imbalanced classification.
    
    Args:
        y: Target series
        
    Returns:
        Dictionary with class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

