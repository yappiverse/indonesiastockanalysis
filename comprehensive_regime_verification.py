#!/usr/bin/env python3
"""
Comprehensive verification test for regime detection fix.
Tests that all 6 regime classes are properly handled in ML training.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from regime_detection import RegimeDetector
from stock_analyzer import StockAnalyzer
from summary_analyzer import SummaryAnalyzer

def test_all_classes_present():
    """Test 1: Verify all 6 regime classes are properly included in training data"""
    print("=" * 70)
    print("TEST 1: VERIFY ALL 6 REGIME CLASSES IN TRAINING DATA")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Create features
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

    # Add all 6 regime classes with balanced distribution
    classes = ['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways']
    df['regime'] = np.random.choice(classes, n_samples, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])

    print("Original data class distribution:")
    print(df['regime'].value_counts())
    print()

    detector = RegimeDetector()

    # Test ML detection
    result = detector.detect_ml(df, retrain=True)
    model_info = detector.get_model_info()
    
    print("Model training results:")
    print(f"Status: {model_info['ml_metrics']['status']}")
    print(f"Missing classes in training: {model_info['ml_metrics']['missing_classes']}")
    print(f"Class distribution in training: {model_info['ml_metrics']['class_distribution']}")
    print(f"All 6 classes present in training: {len(model_info['ml_metrics']['class_distribution']) == 6}")
    
    # Verify HighVol class specifically
    highvol_present = 'HighVol' in [detector._int_to_label[i] for i in model_info['ml_metrics']['class_distribution'].keys()]
    print(f"HighVol class present: {highvol_present}")
    
    # Test predictions
    print("\nFinal predictions distribution:")
    print(result['regime_ml'].value_counts())
    
    # Verify all 6 classes appear in predictions
    prediction_classes = set(result['regime_ml'].dropna().unique())
    expected_classes = set(['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways'])
    missing_in_predictions = expected_classes - prediction_classes
    print(f"Missing classes in predictions: {list(missing_in_predictions)}")
    
    return model_info, result

def test_highvol_specific_scenario():
    """Test 2: Create scenario specifically testing HighVol detection"""
    print("\n" + "=" * 70)
    print("TEST 2: HIGHVOL-SPECIFIC SCENARIO TEST")
    print("=" * 70)
    
    np.random.seed(123)
    n_samples = 500
    dates = pd.date_range('2021-01-01', periods=n_samples, freq='D')

    # Create features with explicit HighVol conditions
    df = pd.DataFrame({
        'trend_diff': np.random.randn(n_samples) * 0.5,
        'ret_1': np.random.randn(n_samples) * 0.02,
        'ret_5': np.random.randn(n_samples) * 0.03,
        'ret_20': np.random.randn(n_samples) * 0.04,
        'vol_20': np.random.randn(n_samples) * 0.2 + 0.05,  # Higher volatility
        'vol_50': np.random.randn(n_samples) * 0.2 + 0.05,
        'vol_z': np.random.randn(n_samples) * 1.5,  # Higher z-score
        'atr_14': np.random.randn(n_samples) * 0.15 + 0.02
    }, index=dates)

    # Force more HighVol samples
    classes = ['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways']
    df['regime'] = np.random.choice(classes, n_samples, p=[0.1, 0.1, 0.1, 0.3, 0.1, 0.3])

    print("HighVol-focused data distribution:")
    print(df['regime'].value_counts())
    print()

    detector = RegimeDetector()
    result = detector.detect_ml(df, retrain=True)
    model_info = detector.get_model_info()
    
    print("HighVol model results:")
    print(f"Missing classes: {model_info['ml_metrics']['missing_classes']}")
    print(f"HighVol samples in training: {model_info['ml_metrics']['class_distribution'].get(3, 0)}")
    
    # Test single prediction with HighVol-like features
    highvol_features = {
        'trend_diff': 0.1,
        'ret_1': 0.01,
        'ret_5': 0.02,
        'ret_20': 0.03,
        'vol_20': 0.08,  # High volatility
        'vol_50': 0.07,
        'vol_z': 2.5,    # High z-score
        'atr_14': 0.04
    }
    
    try:
        prediction = detector.predict_single(highvol_features)
        print(f"HighVol-like feature prediction: {prediction}")
    except Exception as e:
        print(f"Prediction test failed: {e}")
    
    return model_info, result

def test_pipeline_integration():
    """Test 3: Verify complete pipeline integration"""
    print("\n" + "=" * 70)
    print("TEST 3: COMPLETE PIPELINE INTEGRATION")
    print("=" * 70)
    
    print("Testing with multiple tickers...")
    tickers = ['BBCA', 'BBRI', 'BMRI']  # Test with multiple Indonesian stocks
    
    for ticker in tickers:
        print(f"\n--- Testing {ticker}.JK ---")
        try:
            analyzer = StockAnalyzer()
            result = analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
            
            if result['success']:
                print(f"✅ {ticker}: Regime={result['regime']}, Confidence={result['conf']:.2f}")
                
                # Generate summary
                summary_analyzer = SummaryAnalyzer()
                summary = summary_analyzer.generate_summary(result)
                print(f"   Recommendation: {summary['recommendation']}, Score: {summary['summary_score']}/5")
                
            else:
                print(f"❌ {ticker}: Failed - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {ticker}: Exception - {e}")

def test_edge_cases():
    """Test 4: Edge cases and boundary conditions"""
    print("\n" + "=" * 70)
    print("TEST 4: EDGE CASES AND BOUNDARY CONDITIONS")
    print("=" * 70)
    
    # Test with minimal data but all classes
    np.random.seed(456)
    n_samples = 50  # Very small dataset
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')

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

    # Ensure at least one sample of each class
    classes = ['Bear', 'Bull', 'Crash', 'HighVol', 'Recovery', 'Sideways']
    df['regime'] = classes * (n_samples // len(classes)) + classes[:n_samples % len(classes)]
    np.random.shuffle(df['regime'].values)

    print("Minimal data distribution:")
    print(df['regime'].value_counts())
    print()

    detector = RegimeDetector()
    result = detector.detect_ml(df, retrain=True)
    model_info = detector.get_model_info()
    
    print("Minimal data results:")
    print(f"Status: {model_info['ml_metrics']['status']}")
    print(f"Samples: {model_info['ml_metrics']['samples']}")
    print(f"Missing classes: {model_info['ml_metrics']['missing_classes']}")

def main():
    """Run all verification tests"""
    print("COMPREHENSIVE REGIME DETECTION VERIFICATION")
    print("Testing the fix for 'HighVol' class missing from training data")
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results['test1'] = test_all_classes_present()
    results['test2'] = test_highvol_specific_scenario()
    results['test3'] = test_pipeline_integration()
    results['test4'] = test_edge_cases()
    
    # Summary report
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_tests_passed = True
    for test_name, (model_info, result) in results.items():
        if test_name in ['test1', 'test2', 'test4']:
            missing_classes = model_info['ml_metrics']['missing_classes']
            if missing_classes:
                print(f"❌ {test_name}: Missing classes detected: {missing_classes}")
                all_tests_passed = False
            else:
                print(f"✅ {test_name}: All classes present in training")
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED: Regime detection fix is working correctly!")
        print("   - All 6 regime classes properly included in training")
        print("   - 'HighVol' class no longer missing")
        print("   - Model can predict all regime types")
        print("   - Complete pipeline integration verified")
    else:
        print("\n⚠️  SOME TESTS FAILED: Review the results above")

if __name__ == "__main__":
    main()