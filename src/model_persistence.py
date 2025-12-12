"""
Model persistence functions for saving and loading trained models.
"""

import pickle
import joblib
from pathlib import Path
from typing import Optional
from sklearn.pipeline import Pipeline


def save_model(
    model: Pipeline,
    filepath: str,
    use_joblib: bool = True
) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model (Pipeline)
        filepath: Path to save the model (e.g., 'models/model_lr.pkl')
        use_joblib: If True, use joblib (better for sklearn), else use pickle
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if use_joblib:
        joblib.dump(model, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"✓ Model saved to: {filepath}")


def load_model(
    filepath: str,
    use_joblib: bool = True
) -> Pipeline:
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model
        use_joblib: If True, use joblib, else use pickle
        
    Returns:
        Loaded model (Pipeline)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if use_joblib:
        model = joblib.load(filepath)
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"✓ Model loaded from: {filepath}")
    return model


def save_model_with_metadata(
    model: Pipeline,
    filepath: str,
    metadata: dict,
    use_joblib: bool = True
) -> None:
    """
    Save model with metadata (metrics, training date, etc.).
    
    Args:
        model: Trained model
        filepath: Path to save model
        metadata: Dictionary with metadata (e.g., {'auc_roc': 0.75, 'training_date': '2024-01-01'})
        use_joblib: If True, use joblib
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if use_joblib:
        joblib.dump(model, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata
    metadata_path = filepath.with_suffix('.metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Model saved to: {filepath}")
    print(f"✓ Metadata saved to: {metadata_path}")

