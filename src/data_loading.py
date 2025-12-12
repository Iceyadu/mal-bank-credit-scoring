"""
Data loading functions for credit risk modeling.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_applications(path: str) -> pd.DataFrame:
    """Load main applications table."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try alternative encodings
        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        # Last resort: use latin-1 with error replacement
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_previous_applications(path: str) -> pd.DataFrame:
    """Load previous applications table."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_installments(path: str) -> pd.DataFrame:
    """Load installments payment history."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_bureau(path: str) -> pd.DataFrame:
    """Load credit bureau data."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_bureau_balance(path: str) -> pd.DataFrame:
    """Load credit bureau balance history."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_credit_card_balance(path: str) -> pd.DataFrame:
    """Load credit card balance history."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_pos_cash_balance(path: str) -> pd.DataFrame:
    """Load POS cash balance history."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_columns_description(path: str) -> pd.DataFrame:
    """Load column descriptions."""
    # Try different encodings in case of encoding issues
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # If all encodings fail, try with error handling
    return pd.read_csv(path, encoding='latin-1', errors='replace')


def load_all_data(data_dir: str = "data") -> dict:
    """
    Load all data files from the data directory.
    
    Returns:
        Dictionary with keys: applications, previous_applications, installments,
        bureau, bureau_balance, credit_card_balance, pos_cash_balance, columns_description
    """
    data_dir = Path(data_dir)
    
    return {
        "applications": load_applications(str(data_dir / "applications.csv")),
        "previous_applications": load_previous_applications(str(data_dir / "previous_applications.csv")),
        "installments": load_installments(str(data_dir / "installments.csv")),
        "bureau": load_bureau(str(data_dir / "bureau.csv")),
        "bureau_balance": load_bureau_balance(str(data_dir / "bureau_balance.csv")),
        "credit_card_balance": load_credit_card_balance(str(data_dir / "credit_card_balance.csv")),
        "pos_cash_balance": load_pos_cash_balance(str(data_dir / "pos_cash_balance.csv")),
        "columns_description": load_columns_description(str(data_dir / "HomeCredit_columns_description.csv")),
    }

