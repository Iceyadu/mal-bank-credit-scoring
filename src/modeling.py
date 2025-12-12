"""
Model training functions for credit risk modeling.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from typing import List, Tuple, Optional


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at specified percentiles."""
    
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for i, col in enumerate(X.columns):
                X[col] = X[col].clip(lower=self.lower_bounds_[i], upper=self.upper_bounds_[i])
            return X
        else:
            X = X.copy()
            for i in range(X.shape[1]):
                X[:, i] = np.clip(X[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
            return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categories into 'OTHER'."""
    
    def __init__(self, min_frequency: float = 0.01):
        self.min_frequency = min_frequency
        self.category_mapping_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                value_counts = X[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < self.min_frequency].index.tolist()
                if rare_categories:
                    self.category_mapping_[col] = rare_categories
        else:
            # For array input, treat each column separately
            X_df = pd.DataFrame(X)
            for col in X_df.columns:
                value_counts = X_df[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < self.min_frequency].index.tolist()
                if rare_categories:
                    self.category_mapping_[col] = rare_categories
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col, rare_cats in self.category_mapping_.items():
                if col in X.columns:
                    X[col] = X[col].replace(rare_cats, 'OTHER')
            return X
        else:
            X_df = pd.DataFrame(X)
            for col, rare_cats in self.category_mapping_.items():
                if col in X_df.columns:
                    X_df[col] = X_df[col].replace(rare_cats, 'OTHER')
            return X_df.values


def get_preprocessing_pipeline(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    cap_outliers: bool = True
) -> ColumnTransformer:
    """
    Build preprocessing pipeline using ColumnTransformer.
    
    Args:
        df: DataFrame to analyze (for column detection)
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        cap_outliers: Whether to cap outliers for numerical features
        
    Returns:
        ColumnTransformer pipeline
    """
    # Numerical preprocessing
    numerical_steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    
    if cap_outliers:
        numerical_steps.append(('outlier_capper', OutlierCapper()))
    
    numerical_steps.append(('scaler', StandardScaler()))
    
    numerical_transformer = Pipeline(numerical_steps)
    
    # Categorical preprocessing
    categorical_steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('rare_grouper', RareCategoryGrouper(min_frequency=0.01)),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ]
    
    categorical_transformer = Pipeline(categorical_steps)
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing: ColumnTransformer,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> Pipeline:
    """
    Train logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessing: Preprocessing pipeline
        class_weight: Class weight strategy
        random_state: Random state for reproducibility
        
    Returns:
        Trained Pipeline (preprocessing + model)
    """
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing: ColumnTransformer,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 7
) -> Pipeline:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessing: Preprocessing pipeline
        scale_pos_weight: Weight for positive class (if None, calculated from data)
        random_state: Random state for reproducibility
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        
    Returns:
        Trained Pipeline (preprocessing + model)
    """
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / (pos_count + 1e-6)
    
    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbose=-1,
        force_col_wise=True
    )
    
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    id_col: str = 'SK_ID_CURR',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from dataframe.
    
    Args:
        df: Full dataframe
        target_col: Name of target column
        id_col: Name of ID column to exclude
        exclude_cols: Additional columns to exclude
        
    Returns:
        Tuple of (X, y)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude_cols = exclude_cols + [id_col, target_col]
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    
    return X, y

