#!/usr/bin/env python3
"""
Final Pipeline Verification Test
Comprehensive test of the complete stock analysis pipeline to verify:
1. Class label fix works end-to-end
2. ELSA.JK analysis produces valid recommendations
3. All components work together
4. System produces valid metrics
5. "Invalid classes inferred from unique values" error is eliminated
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher
from regime_detection import RegimeDetector, RegimeAnalyzer
from feature_engineering import FeatureEngineer
from stock_analyzer import StockAnalyzer
from summary_analyzer import SummaryAnalyzer
from backtesting import Backtester, PerformanceReporter

def test_complete_pipeline():
    """Test the complete stock analysis pipeline end-to-end."""
    
    print("=" * 80)
    print("FINAL PIPELINE VERIFICATION TEST")
    print("=" * 80)
    print("Testing complete workflow from stock symbol input to final analysis output")
    print()
    
    test_results = {
        'regime_detection': False,
        'feature_engineering': False,
        'backtesting': False,
        'summary_generation': False,
        'error_handling': False,
        'class_label_fix': False
    }
    
    # Test 1: Regime Detection with Class Label Fix
    print("1. TESTING REGIME DETECTION WITH CLASS LABEL FIX")
    print("-" * 50)
    
    try:
        # Test with synthetic data containing all 6 classes
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
        result_df = detector.detect_ml(df, retrain=True)
        
        # Verify no class label mismatch
        regime_counts = result_df['regime_ml'].value_counts()
        invalid_classes = [cls for cls in regime_counts.index if cls not in detector.FIXED_CLASSES]
        
        if invalid_classes:
            print(f"   ❌ Found invalid classes: {invalid_classes}")
        else:
            print(f"   ✅ All regime classes are valid: {list(regime_counts.index)}")
            test_results['regime_detection'] = True
            test_results['class_label_fix'] = True
            
        model_info = detector.get_model_info()
        print(f"   Model status: {model_info['status']}")
        print(f"   Classes handled: {list(model_info['classes'].values())}")
        
    except Exception as e:
        print(f"   ❌ Regime detection test failed: {e}")
    
    # Test 2: Feature Engineering
    print("\n2. TESTING FEATURE ENGINEERING")
    print("-" * 50)
    
    try:
        data_fetcher = DataFetcher()
        df_stock = data_fetcher.get_ohlcv("BBCA.JK", period="1y", interval="1d")
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.add_daily_features(df_stock)
        
        required_features = ['trend_diff', 'ret_1', 'ret_5', 'ret_20', 'vol_20', 'vol_50', 'vol_z', 'atr_14']
        missing_features = [f for f in required_features if f not in df_features.columns]
        
        if missing_features:
            print(f"   ❌ Missing features: {missing_features}")
        else:
            print(f"   ✅ All required features generated: {len(df_features.columns)} total features")
            test_results['feature_engineering'] = True
            
    except Exception as e:
        print(f"   ❌ Feature engineering test failed: {e}")
    
    # Test 3: Complete Stock Analysis Pipeline
    print("\n3. TESTING COMPLETE STOCK ANALYSIS PIPELINE")
    print("-" * 50)
    
    try:
        analyzer = StockAnalyzer()
        result = analyzer.analyze_ticker("BBCA", use_cache=False, verbose=False)
        
        if result['success']:
            print(f"   ✅ Complete analysis successful for {result['ticker']}")
            print(f"   Regime: {result['regime']}")
            print(f"   Confidence: {result['conf']:.2f}")
            
            # Verify metrics exist
            stats = result['stats']
            required_metrics = ['Total Return [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Max Drawdown [%]']
            missing_metrics = [m for m in required_metrics if m not in stats]
            
            if missing_metrics:
                print(f"   ⚠️  Missing metrics: {missing_metrics}")
            else:
                print(f"   ✅ All key metrics present")
                test_results['backtesting'] = True
                
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Complete pipeline test failed: {e}")
    
    # Test 4: Summary Generation
    print("\n4. TESTING SUMMARY GENERATION")
    print("-" * 50)
    
    try:
        analyzer = StockAnalyzer()
        result = analyzer.analyze_ticker("BBCA", use_cache=False, verbose=False)
        
        if result['success']:
            summary_analyzer = SummaryAnalyzer()
            summary = summary_analyzer.generate_summary(result)
            
            required_summary_fields = ['recommendation', 'confidence', 'risk', 'rationale', 'key_metrics', 'summary_score']
            missing_fields = [f for f in required_summary_fields if f not in summary]
            
            if missing_fields:
                print(f"   ❌ Missing summary fields: {missing_fields}")
            else:
                print(f"   ✅ Summary generated successfully")
                print(f"   Recommendation: {summary['recommendation']}")
                print(f"   Risk: {summary['risk']}")
                print(f"   Score: {summary['summary_score']}/5")
                test_results['summary_generation'] = True
                
    except Exception as e:
        print(f"   ❌ Summary generation test failed: {e}")
    
    # Test 5: Error Handling
    print("\n5. TESTING ERROR HANDLING")
    print("-" * 50)
    
    try:
        analyzer = StockAnalyzer()
        result = analyzer.analyze_ticker("INVALID_TICKER", use_cache=False, verbose=False)
        
        if not result['success']:
            print(f"   ✅ Error handling working - invalid ticker properly handled")
            print(f"   Error message: {result.get('error', 'No error message')}")
            test_results['error_handling'] = True
        else:
            print(f"   ⚠️  Invalid ticker didn't trigger error")
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
    
    # Final Verification
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 80)
    
    all_passed = all(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("🎉 SUCCESS: Complete pipeline verification PASSED!")
        print("   - Class label mismatch issue is RESOLVED")
        print("   - All 6 regime classes are properly handled")
        print("   - No 'Invalid classes inferred from unique values' error")
        print("   - Complete analysis pipeline functional")
        print("   - All components working together")
        return True
    else:
        print("❌ FAILURE: Some pipeline components need attention")
        failed_tests = [k for k, v in test_results.items() if not v]
        print(f"   Issues in: {', '.join(failed_tests)}")
        return False

def test_elsa_specific():
    """Test ELSA.JK specifically to verify the fix works for this stock."""
    
    print("\n" + "=" * 80)
    print("ELSA.JK SPECIFIC VERIFICATION")
    print("=" * 80)
    
    try:
        analyzer = StockAnalyzer()
        result = analyzer.analyze_ticker("ELSA", use_cache=False, verbose=False)
        
        if result['success']:
            print("✅ ELSA.JK analysis successful!")
            print(f"   Regime: {result['regime']}")
            print(f"   Confidence: {result['conf']:.2f}")
            
            summary_analyzer = SummaryAnalyzer()
            summary = summary_analyzer.generate_summary(result)
            
            print(f"   Recommendation: {summary['recommendation']}")
            print(f"   Risk: {summary['risk']}")
            print(f"   Score: {summary['summary_score']}/5")
            
            # Check for the specific error that was fixed
            error_patterns = [
                "Invalid classes inferred from unique values",
                "class label mismatch", 
                "unexpected classes",
                "invalid regime label"
            ]
            
            all_clear = True
            for pattern in error_patterns:
                if pattern.lower() in str(result).lower():
                    print(f"   ❌ Found error pattern: {pattern}")
                    all_clear = False
            
            if all_clear:
                print("   ✅ No class label mismatch errors detected")
                return True
            else:
                return False
                
        else:
            print(f"❌ ELSA.JK analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ ELSA.JK test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Final Pipeline Verification...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run comprehensive pipeline test
    pipeline_success = test_complete_pipeline()
    
    # Run ELSA-specific test
    elsa_success = test_elsa_specific()
    
    print("\n" + "=" * 80)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 80)
    
    if pipeline_success and elsa_success:
        print("🎉 ALL TESTS PASSED - PIPELINE VERIFICATION COMPLETE!")
        print("   The class label fix is working correctly end-to-end")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - NEEDS ATTENTION")
        sys.exit(1)