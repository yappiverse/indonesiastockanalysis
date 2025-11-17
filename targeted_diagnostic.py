#!/usr/bin/env python3
"""
Targeted diagnostic for HighVol class missing from training data.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from regime_detection import RegimeDetector

def analyze_highvol_issue():
    print("=" * 70)
    print("TARGETED DIAGNOSTIC: HIGHVOL CLASS MISSING FROM TRAINING DATA")
    print("=" * 70)
    
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
    
    print("\n1. ORIGINAL DATA ANALYSIS")
    print("-" * 40)
    print("Original class distribution:")
    print(df['regime'].value_counts().sort_index())
    
    print("\n2. RULE-BASED DETECTION ANALYSIS")
    print("-" * 40)
    df_rb = detector.detect_rule_based(df)
    print("After rule-based detection:")
    print(df_rb['regime'].value_counts().sort_index())
    
    # Analyze the HighVol detection logic
    rb = detector.config.get("rule_based", {})
    win = rb.get("high_vol_window", 60)
    q = rb.get("high_vol_quantile", 0.8)
    
    high_vol = (
        df["vol_20"]
        > df["vol_20"].rolling(win).quantile(q).shift(1)
    )
    
    crash = df["ret_5"] < rb.get("crash_threshold", -0.08)
    
    print(f"\nHighVol detection conditions:")
    print(f"  - High volatility condition met: {high_vol.sum()} samples")
    print(f"  - Crash condition met: {crash.sum()} samples")
    print(f"  - HighVol & ~Crash: {(high_vol & ~crash).sum()} samples")
    
    print("\n3. ML DATA PREPARATION ANALYSIS")
    print("-" * 40)
    
    # Extract features and labels
    X = df_rb[detector._features]
    y = df_rb["regime"]
    
    print("Before data preparation:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y unique classes: {sorted(y.unique())}")
    
    # Apply data preparation
    X_clean, y_clean, mask_valid = detector._prepare_ml_data(X, y)
    
    print("After data preparation:")
    print(f"  - X_clean shape: {X_clean.shape}")
    print(f"  - y_clean unique classes: {sorted(y_clean.unique())}")
    print(f"  - y_clean class distribution:")
    print(y_clean.value_counts().sort_index())
    
    print("\n4. CROSS-VALIDATION FOLD ANALYSIS")
    print("-" * 40)
    
    # Simulate the cross-validation logic
    y_enc = detector._encode_labels(y_clean)
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"TimeSeriesSplit with {n_splits} splits:")
    for fold_idx, (tr, te) in enumerate(tscv.split(X_clean)):
        y_tr = y_enc.iloc[tr]
        fold_unique_classes = set(y_tr.unique())
        print(f"  Fold {fold_idx + 1}: {len(fold_unique_classes)} classes present")
        if len(fold_unique_classes) < len(detector.FIXED_CLASSES):
            missing_classes = set(range(len(detector.FIXED_CLASSES))) - fold_unique_classes
            missing_names = [detector._int_to_label[c] for c in missing_classes]
            print(f"    Missing classes: {missing_names}")
    
    print("\n5. ROOT CAUSE ANALYSIS")
    print("-" * 40)
    
    # Check the key problematic code sections
    print("Key issues identified:")
    
    # Issue 1: Rule-based detection overwrites original labels
    print("1. Rule-based detection overwrites original 'regime' column")
    print("   - Original 'HighVol' labels are replaced by rule-based logic")
    
    # Issue 2: Cross-validation fold skipping
    print("2. Cross-validation folds skip when not all classes are present")
    print("   - If a fold doesn't have all 6 classes, it's skipped entirely")
    print("   - This can lead to no cross-validation being performed")
    
    # Issue 3: Dummy data addition logic
    print("3. Dummy data addition in _train_ml_model method")
    print("   - Lines 210-214: Missing classes get dummy data added")
    print("   - This creates synthetic samples but may not be effective")
    
    print("\n6. RECOMMENDED FIXES")
    print("-" * 40)
    print("1. Preserve original labels for ML training")
    print("   - Don't overwrite the 'regime' column with rule-based results")
    print("   - Use rule-based only for fallback, not for training labels")
    
    print("2. Improve cross-validation robustness")
    print("   - Don't skip folds when classes are missing")
    print("   - Use stratified sampling or class weights")
    
    print("3. Better handling of class imbalance")
    print("   - Use class weights in XGBoost")
    print("   - Implement proper oversampling/undersampling")
    
    return df, detector

if __name__ == "__main__":
    from sklearn.model_selection import TimeSeriesSplit
    analyze_highvol_issue()