"""
Model explainability functions using SHAP and coefficient analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional
import os


def extract_logistic_coefficients(
    model: Pipeline,
    feature_names: list,
    top_n: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and analyze logistic regression coefficients.
    
    Args:
        model: Trained logistic regression pipeline
        feature_names: Original feature names before preprocessing
        top_n: Number of top features to return
        
    Returns:
        Tuple of (top_positive_coefs, top_negative_coefs) DataFrames
    """
    # Get the classifier from pipeline
    classifier = model.named_steps['classifier']
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessing']
    
    # Extract feature names from one-hot encoding
    if hasattr(preprocessor, 'transformers_'):
        cat_transformer = None
        num_features = []
        cat_features = []
        
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                num_features = list(cols)
            elif name == 'cat':
                cat_transformer = transformer
                cat_cols = list(cols)
        
        # Get one-hot encoded feature names
        all_feature_names = num_features.copy()
        if cat_transformer is not None:
            onehot = cat_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_feature_names = onehot.get_feature_names_out(cat_cols)
                all_feature_names.extend(cat_feature_names)
            else:
                # Fallback for older sklearn
                n_cat_features = len(onehot.categories_)
                cat_feature_names = [f'CAT_{i}' for i in range(n_cat_features)]
                all_feature_names.extend(cat_feature_names)
    else:
        # Fallback: use generic names
        n_features = len(classifier.coef_[0])
        all_feature_names = [f'FEATURE_{i}' for i in range(n_features)]
    
    # Get coefficients
    coefs = classifier.coef_[0]
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': coefs
    })
    
    # Sort by coefficient value
    coef_df = coef_df.sort_values('coefficient', ascending=False)
    
    # Top positive and negative
    top_positive = coef_df.head(top_n).copy()
    top_negative = coef_df.tail(top_n).copy()
    
    return top_positive, top_negative


def plot_logistic_coefficients(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    model_name: str = "Logistic Regression",
    save_path: Optional[str] = None
):
    """
    Plot top positive and negative coefficients.
    
    Args:
        top_positive: DataFrame with top positive coefficients
        top_negative: DataFrame with top negative coefficients
        model_name: Name of model
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top positive
    ax1.barh(range(len(top_positive)), top_positive['coefficient'].values)
    ax1.set_yticks(range(len(top_positive)))
    ax1.set_yticklabels(top_positive['feature'].values, fontsize=9)
    ax1.set_xlabel('Coefficient Value', fontsize=11)
    ax1.set_title(f'Top 15 Positive Coefficients - {model_name}', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Top negative
    ax2.barh(range(len(top_negative)), top_negative['coefficient'].values, color='coral')
    ax2.set_yticks(range(len(top_negative)))
    ax2.set_yticklabels(top_negative['feature'].values, fontsize=9)
    ax2.set_xlabel('Coefficient Value', fontsize=11)
    ax2.set_title(f'Top 15 Negative Coefficients - {model_name}', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_shap_values(
    model: Pipeline,
    X_sample: pd.DataFrame,
    n_samples: int = 100
) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Compute SHAP values for tree-based model.
    
    Args:
        model: Trained model pipeline
        X_sample: Sample data to explain
        n_samples: Number of samples to use for explanation
        
    Returns:
        Tuple of (shap_values, explainer)
    """
    # Get the classifier from pipeline
    classifier = model.named_steps['classifier']
    
    # Preprocess data
    X_preprocessed = model.named_steps['preprocessing'].transform(X_sample)
    
    # Convert to DataFrame if needed
    if isinstance(X_preprocessed, np.ndarray):
        X_preprocessed = pd.DataFrame(X_preprocessed)
    
    # Sample for faster computation
    if len(X_preprocessed) > n_samples:
        sample_idx = np.random.choice(len(X_preprocessed), n_samples, replace=False)
        X_shap = X_preprocessed.iloc[sample_idx]
    else:
        X_shap = X_preprocessed
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(classifier)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap)
    
    # For binary classification, use positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, explainer


def plot_shap_summary(
    model: Pipeline,
    X_sample: pd.DataFrame,
    model_name: str = "LightGBM",
    save_path: Optional[str] = None,
    n_samples: int = 100
):
    """
    Plot SHAP summary (bar + beeswarm).
    
    Args:
        model: Trained model pipeline
        X_sample: Sample data to explain
        model_name: Name of model
        save_path: Path to save plot
        n_samples: Number of samples to use
    """
    shap_values, explainer = compute_shap_values(model, X_sample, n_samples)
    
    # Preprocess for feature names
    X_preprocessed = model.named_steps['preprocessing'].transform(X_sample)
    if isinstance(X_preprocessed, np.ndarray):
        X_preprocessed = pd.DataFrame(X_preprocessed)
    
    if len(X_preprocessed) > n_samples:
        sample_idx = np.random.choice(len(X_preprocessed), n_samples, replace=False)
        X_shap = X_preprocessed.iloc[sample_idx]
    else:
        X_shap = X_preprocessed
    
    # Get feature names
    feature_names = [f'Feature_{i}' for i in range(X_shap.shape[1])]
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_shap.values,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Also create bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_shap.values,
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Feature Importance (Mean |SHAP|) - {model_name}', 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        bar_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_shap_waterfall(
    model: Pipeline,
    X_instance: pd.DataFrame,
    instance_idx: int,
    model_name: str = "LightGBM",
    save_path: Optional[str] = None
):
    """
    Plot SHAP waterfall for a single instance.
    
    Args:
        model: Trained model pipeline
        X_instance: Instance data
        instance_idx: Index of instance to explain
        model_name: Name of model
        save_path: Path to save plot
    """
    # Get single instance
    if isinstance(X_instance, pd.DataFrame):
        instance = X_instance.iloc[[instance_idx]]
    else:
        instance = X_instance[[instance_idx]]
    
    # Preprocess
    X_preprocessed = model.named_steps['preprocessing'].transform(instance)
    if isinstance(X_preprocessed, np.ndarray):
        X_preprocessed = pd.DataFrame(X_preprocessed)
    
    # Get classifier
    classifier = model.named_steps['classifier']
    
    # Create explainer and compute SHAP
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_preprocessed)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get feature names
    feature_names = [f'Feature_{i}' for i in range(X_preprocessed.shape[1])]
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_preprocessed.iloc[0].values,
            feature_names=feature_names
        ),
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Waterfall - {model_name} (Instance {instance_idx})', 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return shap_values[0], explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

