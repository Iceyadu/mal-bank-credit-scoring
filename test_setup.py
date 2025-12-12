"""
Quick setup test script to verify the environment and data are ready.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import lightgbm as lgb
        import matplotlib
        import seaborn as sns
        import shap
        import scipy
        print("✓ All packages imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False

def test_data_files():
    """Test if data files exist."""
    print("\nTesting data files...")
    data_dir = Path("../malbank_case_data")
    if not data_dir.exists():
        data_dir = Path("data")
    
    required_files = [
        "applications.csv",
        "previous_applications.csv",
        "installments.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "pos_cash_balance.csv",
        "HomeCredit_columns_description.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file} - NOT FOUND")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print(f"Data directory checked: {data_dir.absolute()}")
        return False
    return True

def test_src_modules():
    """Test if src modules can be imported."""
    print("\nTesting src modules...")
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    modules = [
        "data_loading",
        "feature_engineering",
        "modeling",
        "evaluation",
        "explainability",
        "utils"
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except Exception as e:
            print(f"✗ {module}.py - Error: {e}")
            failed.append(module)
    
    if failed:
        return False
    return True

def test_data_loading():
    """Test if data can be loaded."""
    print("\nTesting data loading...")
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        from data_loading import load_all_data
        
        # Try to find data directory
        data_dir = "../malbank_case_data"
        if not Path(data_dir).exists():
            data_dir = "data"
        
        if not Path(data_dir).exists():
            print(f"✗ Data directory not found: {data_dir}")
            return False
        
        print(f"Loading data from: {data_dir}")
        data = load_all_data(data_dir)
        
        print(f"✓ Loaded {len(data)} tables:")
        for key, df in data.items():
            print(f"  - {key}: {df.shape}")
        
        # Quick check on applications
        if 'applications' in data:
            apps = data['applications']
            if 'TARGET' in apps.columns:
                print(f"✓ Target column found: TARGET")
                print(f"  Default rate: {apps['TARGET'].mean():.2%}")
            else:
                print("⚠ TARGET column not found (may have different name)")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Credit Risk Model - Setup Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Source Modules", test_src_modules()))
    results.append(("Data Loading", test_data_loading()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to run the notebook.")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

