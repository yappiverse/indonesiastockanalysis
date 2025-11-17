#!/usr/bin/env python3
"""
Debug script to investigate why 'HighVol' class is missing from training data.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from regime_detection import RegimeDetector

def analyze_rule_based_detection():
    """Analyze the rule-based detection logic for HighVol class"""
    print("=" * 60)
    print("ANALYZING RULE-BASED DETECTION LOGIC")
    print("=" * 60)
    
    # Create test data with all 6 classes
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    df = pd.DataFrame({
        'trend_diff': np.random.randn(n_samples),
        'ret_1': np.random.randn(n_samples) * 0.01,
        'ret_5': np.random.randn(n_samples) * 0.02,
        'ret_20': np.random.randn(n_samples) * 0.05,
        'vol_20': np.random.randn(n_samples) * 0.1 + 0.02,
        'vol_50': np.random.randn(n_samples) * 0.1 + 0.02,
        'vol_z': np.random.randn(n_samples),
        'atr_14': np.random.randn(n_samples) * 0.1 + 0.01
    }, index=dates)

    # Add all 6 regime classes
    classes = ['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways']
    df['regime'] = np.random.choice(classes, n_samples)
    
    print("Original data class distribution:")
    print(df['regime'].value_counts())
    print()
    
    # Test rule-based detection
    detector = RegimeDetector()
    df_rb = detector.detect_rule_based(df)
    
    print("After rule-based detection class distribution:")
    print(df_rb['regime'].value_counts())
    print()
    
    # Analyze HighVol detection conditions
    rb = detector.config.get("rule_based", {})
    win = rb.get("high_vol_window", 60)
    q = rb.get("high_vol_quantile", 0.8)
    
    high_vol = (
        df["vol_20"]
        > df["vol_20"].rolling(win).quantile(q).shift(1)
    )
    
    crash = df["ret_5"] < rb.get("crash_threshold", -0.08)
    
    print(f"HighVol detection conditions:")
    print(f"  - High volatility condition met: {high_vol.sum()} samples")
    print(f"  - Crash condition met: {crash.sum()} samples")
    print(f"  - HighVol & ~Crash: {(high_vol & ~crash).sum()} samples")
    print()
    
    return df, df_rb

def analyze_ml_data_preparation(df):
    """Analyze the ML data preparation pipeline"""
    print("=" * 60)
    print("ANALYZING ML DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    detector = RegimeDetector()
    
    # Step 1 - rule-based detection
    df_rb = detector.detect_rule_based(df)
    
    # Step 2 - data extraction
    X = df_rb[detector._features]
    y = df_rb["regime"]
    
    print("Before data preparation:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y unique classes: {y.unique()}")
    print(f"  - y class distribution:")
    print(y.value_counts())
    print()
    
    # Step 3 - data preparation
    X_clean, y_clean, mask_valid = detector._prepare_ml_data(X, y)
    
    print("After data preparation:")
    print(f"  - X_clean shape: {X_clean.shape}")
    print(f"  - y_clean unique classes: {y_clean.unique()}")
    print(f"  - y_clean class distribution:")
    print(y_clean.value_counts())
    print()
    
    # Check for missing values
    print("Missing values analysis:")
    print(f"  - X missing values: {X.isnull().sum().sum()}")
    print(f"  - y missing values: {y.isnull().sum()}")
    print(f"  - mask_valid True count: {mask_valid.sum()}")
    print()
    
    return X_clean, y_clean

def analyze_training_data_distribution():
    """Analyze the training data distribution throughout the pipeline"""
    print("=" * 60)
    print("ANALYZING TRAINING DATA DISTRIBUTION")
    print("=" * 60)
    
    # Create test data with all 6 classes
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    df = pd.DataFrame({
        'trend_diff': np.random.randn(n_samples),
        'ret_1': np.random.randn(n_samples) * 0.01,
        'ret_5': np.random.randn(n_samples) * 0.02,
        'ret_20': np.random.randn(n_samples) * 0.05,
        'vol_20': np.random.randn(n_samples) * 0.1 + 0.02,
        'vol_50': np.random.randn(n_samples) * 0.1 + 0.02,
        'vol_z': np.random.randn(n_samples),
        'atr_14': np.random.randn(n_samples) * 0.1 + 0.01
    }, index=dates)

    # Add all 6 regime classes
    classes = ['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways']
    df['regime'] = np.random.choice(classes, n_samples)
    
    detector = RegimeDetector()
    
    # Run full ML detection pipeline
    result = detector.detect_ml(df, retrain=True)
    
    print("Final model info:")
    model_info = detector.get_model_info()
    print(f"  - Status: {model_info['status']}")
    print(f"  - ML metrics: {model_info['ml_metrics']}")
    print()
    
    # Check if HighVol is in the final predictions
    print("Final prediction class distribution:")
    print(result['regime_ml'].value_counts())
    print()
    
    return result, model_info

def main():
    """Main diagnostic function"""
    print("DEBUGGING HIGHVOL CLASS MISSING FROM TRAINING DATA")
    print("=" * 60)
    
    # Analyze rule-based detection
    df, df_rb = analyze_rule_based_detection()
    
    # Analyze ML data preparation
    X_clean, y_clean = analyze_ml_data_preparation(df)
    
    # Analyze training data distribution
    result, model_info = analyze_training_data_distribution()
    
    print("=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    
    if 'HighVol' not in y_clean.unique():
        print("❌ ISSUE FOUND: HighVol class is missing from training data after preprocessing")
        print("   This suggests the issue is in the data preparation pipeline")
    else:
        print("✅ HighVol class is present in training data after preprocessing")
        print("   The issue may be in the cross-validation or model training logic")
    
    if model_info['ml_metrics'].get('missing_classes'):
        print(f"❌ Model reports missing classes: {model_info['ml_metrics']['missing_classes']}")
    else:
        print("✅ No missing classes reported by model")

if __name__ == "__main__":
    main()