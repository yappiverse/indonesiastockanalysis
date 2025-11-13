#!/usr/bin/env python3
"""
Test to confirm ML model sharing between RegimeDetector instances
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from regime_detection import RegimeDetector

def test_ml_model_sharing():
    """Test if ML models are shared between instances"""
    print("ML MODEL SHARING TEST")
    print("=" * 50)
    
    # Create multiple instances
    detectors = [RegimeDetector() for _ in range(3)]
    
    print("Initial state:")
    for i, detector in enumerate(detectors):
        print(f"Detector {i}: id={id(detector)}, ML pipeline id={id(detector._ml_pipeline)}")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.cumsum(np.random.randn(300) * 0.01 + 0.001) + 100,
        'High': np.cumsum(np.random.randn(300) * 0.01 + 0.002) + 105,
        'Low': np.cumsum(np.random.randn(300) * 0.01 - 0.001) + 95,
        'Close': np.cumsum(np.random.randn(300) * 0.01 + 0.001) + 100,
        'Volume': np.random.randint(1000, 10000, 300)
    }, index=dates)
    
    # Add features
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    sample_data = fe.add_daily_features(sample_data)
    sample_data = sample_data.dropna()
    
    print(f"\nSample data shape: {sample_data.shape}")
    
    # Train first detector
    print("\nTraining detector 0...")
    result0 = detectors[0].detect_ml(sample_data.copy())
    
    print("After training detector 0:")
    for i, detector in enumerate(detectors):
        print(f"Detector {i}: ML pipeline id={id(detector._ml_pipeline)}, is_fitted={detector._is_fitted}")
    
    # Train second detector with different data
    print("\nTraining detector 1 with different data...")
    sample_data2 = sample_data.copy()
    sample_data2['Close'] = sample_data2['Close'] * 0.8  # Different price pattern
    result1 = detectors[1].detect_ml(sample_data2.copy())
    
    print("After training detector 1:")
    for i, detector in enumerate(detectors):
        print(f"Detector {i}: ML pipeline id={id(detector._ml_pipeline)}, is_fitted={detector._is_fitted}")
    
    # Check if predictions are identical (indicating shared model)
    test_point = sample_data.iloc[-1:].copy()
    pred0 = detectors[0].predict_single(test_point.iloc[0].to_dict())
    pred1 = detectors[1].predict_single(test_point.iloc[0].to_dict())
    
    print(f"\nPrediction comparison:")
    print(f"Detector 0: {pred0}")
    print(f"Detector 1: {pred1}")
    
    if pred0 == pred1:
        print("⚠️ WARNING: Different detectors making identical predictions!")
        print("This indicates ML model sharing between instances")
    else:
        print("✓ Different detectors making different predictions")

def test_class_attribute_sharing():
    """Test if the issue is with class-level attributes"""
    print("\n" + "=" * 50)
    print("CLASS ATTRIBUTE SHARING TEST")
    print("=" * 50)
    
    # Check if _ml_pipeline is defined at class level
    import inspect
    
    print("RegimeDetector class attributes:")
    for attr_name, attr_value in RegimeDetector.__dict__.items():
        if not attr_name.startswith('__'):
            print(f"  {attr_name}: {type(attr_value)}")
    
    print("\nInstance attributes after creation:")
    detector1 = RegimeDetector()
    detector2 = RegimeDetector()
    
    print(f"detector1._ml_pipeline: {detector1._ml_pipeline}")
    print(f"detector2._ml_pipeline: {detector2._ml_pipeline}")
    print(f"Same object? {detector1._ml_pipeline is detector2._ml_pipeline}")

if __name__ == "__main__":
    test_ml_model_sharing()
    test_class_attribute_sharing()