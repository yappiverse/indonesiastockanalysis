#!/usr/bin/env python3
"""
Comprehensive diagnostic script for ELSA.JK regime detection failure.
This script will trace the exact failure point in the analysis pipeline.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import traceback
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher
from feature_engineering import FeatureEngineer
from regime_detection import RegimeDetector
from stock_analyzer import StockAnalyzer

def diagnose_elsa_failure():
    """Comprehensive diagnostic for ELSA.JK regime detection failure."""
    
    print("=" * 80)
    print("ELSA.JK COMPREHENSIVE DIAGNOSTIC")
    print("=" * 80)
    print("Tracing the exact failure point in the analysis pipeline...")
    print()
    
    diagnostic_results = {}
    
    # Step 1: Test data loading
    print("1. TESTING DATA LOADING...")
    try:
        data_fetcher = DataFetcher()
        df_elsa = data_fetcher.get_ohlcv("ELSA.JK", period="2y", interval="1d", use_cache=False)
        print(f"   ✅ Data loading successful")
        print(f"   Data shape: {df_elsa.shape}")
        print(f"   Columns: {list(df_elsa.columns)}")
        print(f"   Date range: {df_elsa.index[0]} to {df_elsa.index[-1]}")
        print(f"   Sample data:")
        print(f"   {df_elsa.head(3)}")
        diagnostic_results['data_loading'] = {'success': True, 'data': df_elsa}
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        diagnostic_results['data_loading'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
        return diagnostic_results
    
    # Step 2: Test feature engineering
    print("\n2. TESTING FEATURE ENGINEERING...")
    try:
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.add_daily_features(df_elsa.copy())
        print(f"   ✅ Feature engineering successful")
        print(f"   Features shape: {df_features.shape}")
        print(f"   Feature columns: {list(df_features.columns)}")
        print(f"   Missing values per column:")
        for col in df_features.columns:
            missing = df_features[col].isna().sum()
            if missing > 0:
                print(f"     {col}: {missing} missing values")
        diagnostic_results['feature_engineering'] = {'success': True, 'data': df_features}
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
        diagnostic_results['feature_engineering'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
        return diagnostic_results
    
    # Step 3: Test regime detector initialization
    print("\n3. TESTING REGIME DETECTOR INITIALIZATION...")
    try:
        detector = RegimeDetector()
        print(f"   ✅ Regime detector initialized")
        print(f"   Fixed classes: {detector.FIXED_CLASSES}")
        print(f"   Features config: {detector._features}")
        print(f"   Model fitted: {detector._is_fitted}")
        diagnostic_results['detector_init'] = {'success': True, 'detector': detector}
    except Exception as e:
        print(f"   ❌ Regime detector initialization failed: {e}")
        diagnostic_results['detector_init'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
        return diagnostic_results
    
    # Step 4: Test rule-based detection
    print("\n4. TESTING RULE-BASED DETECTION...")
    try:
        df_rb = detector.detect_rule_based(df_features.copy())
        print(f"   ✅ Rule-based detection successful")
        print(f"   Regime column added: {'regime' in df_rb.columns}")
        if 'regime' in df_rb.columns:
            regime_counts = df_rb['regime'].value_counts()
            print(f"   Rule-based regime distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(df_rb)) * 100
                print(f"     {regime}: {count} ({percentage:.1f}%)")
        diagnostic_results['rule_based'] = {'success': True, 'data': df_rb}
    except Exception as e:
        print(f"   ❌ Rule-based detection failed: {e}")
        diagnostic_results['rule_based'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    # Step 5: Test ML regime detection
    print("\n5. TESTING ML REGIME DETECTION...")
    try:
        # First, check if we have the required 'regime' column for ML training
        if 'regime' not in df_features.columns:
            print("   ⚠️  No 'regime' column found, adding rule-based regimes first...")
            df_features = detector.detect_rule_based(df_features)
        
        print(f"   Starting ML detection with retrain=True...")
        df_ml = detector.detect_ml(df_features.copy(), retrain=True)
        print(f"   ✅ ML regime detection successful")
        
        # Check results
        if 'regime_ml' in df_ml.columns:
            regime_counts = df_ml['regime_ml'].value_counts()
            print(f"   ML regime distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(df_ml)) * 100
                print(f"     {regime}: {count} ({percentage:.1f}%)")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"   Model status: {model_info['status']}")
        if 'ml_metrics' in model_info:
            print(f"   ML metrics: {model_info['ml_metrics']}")
        
        diagnostic_results['ml_detection'] = {'success': True, 'data': df_ml, 'model_info': model_info}
    except Exception as e:
        print(f"   ❌ ML regime detection failed: {e}")
        print(f"   Full traceback:")
        traceback.print_exc()
        diagnostic_results['ml_detection'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    # Step 6: Test single prediction
    print("\n6. TESTING SINGLE PREDICTION...")
    try:
        if diagnostic_results['ml_detection']['success']:
            df_ml = diagnostic_results['ml_detection']['data']
            latest_features = {}
            for col in detector._features:
                if col in df_ml.columns:
                    latest_features[col] = df_ml[col].iloc[-1]
            
            print(f"   Latest features for prediction: {latest_features}")
            prediction = detector.predict_single(latest_features)
            print(f"   ✅ Single prediction successful: {prediction}")
            diagnostic_results['single_prediction'] = {'success': True, 'prediction': prediction}
        else:
            print(f"   ⚠️  Skipping single prediction due to previous ML failure")
            diagnostic_results['single_prediction'] = {'success': False, 'error': 'ML detection failed'}
    except Exception as e:
        print(f"   ❌ Single prediction failed: {e}")
        diagnostic_results['single_prediction'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    # Step 7: Test complete stock analyzer
    print("\n7. TESTING COMPLETE STOCK ANALYZER...")
    try:
        stock_analyzer = StockAnalyzer()
        print(f"   Running complete analysis pipeline...")
        result = stock_analyzer.analyze_ticker("ELSA.JK", use_cache=False, verbose=False)
        
        if result.get('success'):
            print(f"   ✅ Complete analysis successful")
            print(f"   Regime: {result.get('regime', 'Unknown')}")
            print(f"   Confidence: {result.get('conf', 'Unknown')}")
            diagnostic_results['complete_analysis'] = {'success': True, 'result': result}
        else:
            print(f"   ❌ Complete analysis failed: {result.get('error', 'Unknown error')}")
            diagnostic_results['complete_analysis'] = {'success': False, 'error': result.get('error', 'Unknown error')}
    except Exception as e:
        print(f"   ❌ Complete analysis failed: {e}")
        diagnostic_results['complete_analysis'] = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    failed_steps = []
    for step_name, result in diagnostic_results.items():
        if not result.get('success', False):
            failed_steps.append(step_name)
            print(f"❌ {step_name}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            print(f"✅ {step_name}: SUCCESS")
    
    if failed_steps:
        print(f"\n❌ FAILURE DETECTED in steps: {failed_steps}")
        print(f"Primary failure point: {failed_steps[0]}")
    else:
        print(f"\n🎉 ALL STEPS SUCCESSFUL - No issues detected")
    
    return diagnostic_results

if __name__ == "__main__":
    results = diagnose_elsa_failure()
    
    # Save detailed results for further analysis
    import json
    with open('elsa_diagnostic_results.json', 'w') as f:
        # Convert DataFrames to string representations for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (pd.DataFrame, pd.Series)):
                        serializable_results[key][subkey] = str(subvalue)
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = str(value)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to 'elsa_diagnostic_results.json'")