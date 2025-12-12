"""
Model evaluation functions and plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_score, recall_score
)
from scipy.stats import ks_2samp
from typing import Tuple, Optional
import os


def calculate_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        KS statistic
    """
    scores_0 = y_pred_proba[y_true == 0]
    scores_1 = y_pred_proba[y_true == 1]
    ks_stat, _ = ks_2samp(scores_1, scores_0)
    return ks_stat


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "model"
) -> dict:
    """
    Evaluate model and return comprehensive metrics.
    
    Args:
        model: Trained model (Pipeline)
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold
        model_name: Name of model for logging
        
    Returns:
        Dictionary with metrics
    """
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    ks = calculate_ks_statistic(y_test.values, y_pred_proba)
    
    # Classification metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Find optimal threshold for F1
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold
    optimal_f1 = f1_scores[optimal_idx]
    
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier,
        'ks_statistic': ks,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'confusion_matrix_optimal': cm_optimal,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'y_pred_optimal': y_pred_optimal
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of model
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of model
        save_path: Path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        model_name: Name of model
        threshold: Threshold used
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['No Default', 'Default'],
        yticklabels=['No Default', 'Default']
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name} (threshold={threshold:.2f})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_metrics(metrics: dict, model_name: str):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metrics from evaluate_model
        model_name: Name of model
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Metrics - {model_name}")
    print(f"{'='*60}")
    print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:            {metrics['auc_pr']:.4f}")
    print(f"KS Statistic:      {metrics['ks_statistic']:.4f}")
    print(f"Brier Score:       {metrics['brier_score']:.4f}")
    print(f"\nAt threshold = 0.5:")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1 Score:        {metrics['f1_score']:.4f}")
    print(f"\nAt optimal threshold = {metrics['optimal_threshold']:.4f}:")
    print(f"  Optimal F1:      {metrics['optimal_f1']:.4f}")
    print(f"\nConfusion Matrix (threshold=0.5):")
    print(metrics['confusion_matrix'])
    print(f"{'='*60}\n")

