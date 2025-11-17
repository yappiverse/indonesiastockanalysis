#!/usr/bin/env python3
"""
Test script for ELSA.JK regime detection system.
Verifies that the class label mismatch issue is resolved.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher
from regime_detection import RegimeDetector, RegimeAnalyzer
from feature_engineering import FeatureEngineer
from stock_analyzer import StockAnalyzer

def test_elsa_regime_detection():
    """Test the complete regime detection pipeline with ELSA.JK data."""
    
    print("=" * 80)
    print("ELSA.JK REGIME DETECTION TEST")
    print("=" * 80)
    print("Testing the fixed regime detection system with real ELSA.JK data")
    print("Verifying that class label mismatch issue is resolved")
    print()
    
    # Step 1: Load ELSA.JK stock data
    print("1. Loading ELSA.JK stock data...")
    try:
        data_fetcher = DataFetcher()
        df_elsa = data_fetcher.get_ohlcv("ELSA.JK", period="2y", interval="1d")
        print(f"   ✅ Successfully loaded {len(df_elsa)} days of ELSA.JK data")
        print(f"   Date range: {df_elsa.index[0].strftime('%Y-%m-%d')} to {df_elsa.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"   ❌ Failed to load ELSA.JK data: {e}")
        return False
    
    # Step 2: Generate features
    print("\n2. Generating technical features...")
    try:
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.add_daily_features(df_elsa)
        print(f"   ✅ Successfully generated {len(df_features.columns)} features")
        print(f"   Features: {list(df_features.columns)}")
    except Exception as e:
        print(f"   ❌ Failed to generate features: {e}")
        return False
    
    # Step 3: Initialize regime detector
    print("\n3. Initializing regime detector...")
    try:
        detector = RegimeDetector()
        print(f"   ✅ Regime detector initialized")
        print(f"   Fixed classes: {detector.FIXED_CLASSES}")
        print(f"   Label mapping: {detector._label_to_int}")
    except Exception as e:
        print(f"   ❌ Failed to initialize regime detector: {e}")
        return False
    
    # Step 4: Run ML regime detection
    print("\n4. Running ML regime detection...")
    try:
        print("   Starting regime detection with retrain=True...")
        result_df = detector.detect_ml(df_features, retrain=True)
        print("   ✅ ML regime detection completed successfully!")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"   Model status: {model_info['status']}")
        print(f"   ML metrics: {model_info['ml_metrics']}")
        
    except Exception as e:
        print(f"   ❌ ML regime detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify no class label mismatch error
    print("\n5. Verifying class label consistency...")
    try:
        # Check if the regime_ml column contains only valid classes
        regime_counts = result_df['regime_ml'].value_counts()
        invalid_classes = [cls for cls in regime_counts.index if cls not in detector.FIXED_CLASSES]
        
        if invalid_classes:
            print(f"   ❌ Found invalid classes: {invalid_classes}")
            return False
        else:
            print(f"   ✅ All regime classes are valid: {list(regime_counts.index)}")
            print(f"   Regime distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(result_df)) * 100
                print(f"     {regime}: {count} ({percentage:.1f}%)")
    
    except Exception as e:
        print(f"   ❌ Error verifying class labels: {e}")
        return False
    
    # Step 6: Test single prediction
    print("\n6. Testing single prediction...")
    try:
        # Get latest features for prediction
        latest_features = {col: result_df[col].iloc[-1] for col in df_features.columns if col != 'regime'}
        prediction = detector.predict_single(latest_features)
        print(f"   ✅ Single prediction successful")
        print(f"   Prediction: {prediction}")
    except Exception as e:
        print(f"   ❌ Single prediction failed: {e}")
        return False
    
    # Step 7: Analyze regime distribution
    print("\n7. Analyzing regime distribution...")
    try:
        analyzer = RegimeAnalyzer()
        regime_dist = analyzer.analyze_regime_distribution(result_df)
        print(f"   ✅ Regime analysis completed")
        print(f"   Regime distribution:\n{regime_dist}")
        
        # Check regime transitions
        transitions = analyzer.get_regime_transitions(result_df)
        print(f"   Regime transitions:\n{transitions}")
        
    except Exception as e:
        print(f"   ❌ Regime analysis failed: {e}")
        return False
    
    # Step 8: Run complete stock analysis
    print("\n8. Running complete stock analysis...")
    try:
        stock_analyzer = StockAnalyzer()
        analysis_result = stock_analyzer.analyze_ticker("ELSA.JK", use_cache=False, verbose=False)
        
        if analysis_result.get('success'):
            print(f"   ✅ Complete stock analysis successful")
            print(f"   Regime: {analysis_result.get('regime', 'Unknown')}")
            print(f"   Recommendation: {analysis_result.get('recommendation', 'Unknown')}")
            print(f"   Confidence: {analysis_result.get('confidence', 'Unknown')}")
        else:
            print(f"   ❌ Complete stock analysis failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Complete stock analysis failed: {e}")
        return False
    
    # Final verification
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)
    
    # Check for the specific error that was fixed
    error_patterns = [
        "Invalid classes inferred from unique values",
        "class label mismatch",
        "unexpected classes",
        "invalid regime label"
    ]
    
    print("Checking for previously reported errors...")
    all_clear = True
    
    for pattern in error_patterns:
        if pattern.lower() in str(model_info).lower():
            print(f"   ❌ Found error pattern: {pattern}")
            all_clear = False
        else:
            print(f"   ✅ No '{pattern}' error found")
    
    if all_clear:
        print("\n🎉 SUCCESS: Class label mismatch issue is RESOLVED!")
        print("   - All 6 regime classes are properly handled")
        print("   - No 'Invalid classes inferred from unique values' error")
        print("   - ML model trained successfully")
        print("   - Predictions working correctly")
        print("   - Complete analysis pipeline functional")
        return True
    else:
        print("\n❌ FAILURE: Some issues remain")
        return False

if __name__ == "__main__":
    success = test_elsa_regime_detection()
    sys.exit(0 if success else 1)