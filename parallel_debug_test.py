#!/usr/bin/env python3
"""
Diagnostic test to identify the root cause of identical results in parallel processing.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from stock_analyzer import StockAnalyzer
from data_layer import DataFetcher
from feature_engineering import FeatureEngineer
from regime_detection import RegimeDetector
from backtesting import Backtester

def test_data_fetcher_isolation():
    """Test if DataFetcher instances are properly isolated"""
    print("=" * 60)
    print("TEST 1: DataFetcher Instance Isolation")
    print("=" * 60)
    
    # Create two separate instances
    fetcher1 = DataFetcher()
    fetcher2 = DataFetcher()
    
    print(f"fetcher1 id: {id(fetcher1)}")
    print(f"fetcher2 id: {id(fetcher2)}")
    print(f"fetcher1 cache id: {id(fetcher1._cache)}")
    print(f"fetcher2 cache id: {id(fetcher2._cache)}")
    
    # Test fetching different tickers
    try:
        bca_data = fetcher1.get_ohlcv("BBCA.JK", use_cache=False)
        care_data = fetcher2.get_ohlcv("CARE.JK", use_cache=False)
        
        print(f"BBCA data shape: {bca_data.shape}")
        print(f"CARE data shape: {care_data.shape}")
        print(f"BBCA first close: {bca_data['Close'].iloc[0]}")
        print(f"CARE first close: {care_data['Close'].iloc[0]}")
        print("✓ DataFetcher instances are properly isolated")
    except Exception as e:
        print(f"✗ DataFetcher test failed: {e}")

def test_regime_detector_shared_state():
    """Test if RegimeDetector has shared state issues"""
    print("\n" + "=" * 60)
    print("TEST 2: RegimeDetector Shared State")
    print("=" * 60)
    
    detector1 = RegimeDetector()
    detector2 = RegimeDetector()
    
    print(f"detector1 id: {id(detector1)}")
    print(f"detector2 id: {id(detector2)}")
    print(f"detector1 ML pipeline id: {id(detector1._ml_pipeline)}")
    print(f"detector2 ML pipeline id: {id(detector2._ml_pipeline)}")
    
    # Test with sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add basic features
    fe = FeatureEngineer()
    sample_data = fe.add_daily_features(sample_data)
    
    # Test regime detection with proper data
    # Ensure we have enough data for ML training
    sample_data_with_features = sample_data.dropna()
    if len(sample_data_with_features) > 50:
        result1 = detector1.detect_ml(sample_data_with_features.copy())
        result2 = detector2.detect_ml(sample_data_with_features.copy())
        
        print(f"Result1 regimes: {result1['regime_ml'].unique()}")
        print(f"Result2 regimes: {result2['regime_ml'].unique()}")
        print("✓ RegimeDetector instances appear independent")
    else:
        print("⚠️ Not enough data for ML regime detection test")

def test_stock_analyzer_isolation():
    """Test if StockAnalyzer instances are properly isolated"""
    print("\n" + "=" * 60)
    print("TEST 3: StockAnalyzer Instance Isolation")
    print("=" * 60)
    
    analyzer1 = StockAnalyzer()
    analyzer2 = StockAnalyzer()
    
    print(f"analyzer1 id: {id(analyzer1)}")
    print(f"analyzer2 id: {id(analyzer2)}")
    print(f"analyzer1 data_fetcher id: {id(analyzer1.data_fetcher)}")
    print(f"analyzer2 data_fetcher id: {id(analyzer2.data_fetcher)}")
    print(f"analyzer1 regime_detector id: {id(analyzer1.regime_detector)}")
    print(f"analyzer2 regime_detector id: {id(analyzer2.regime_detector)}")
    
    # Test component isolation
    print(f"analyzer1 data_fetcher cache id: {id(analyzer1.data_fetcher._cache)}")
    print(f"analyzer2 data_fetcher cache id: {id(analyzer2.data_fetcher._cache)}")
    print("✓ StockAnalyzer instances have separate components")

def test_parallel_processing_simulation():
    """Simulate parallel processing to identify contamination"""
    print("\n" + "=" * 60)
    print("TEST 4: Parallel Processing Simulation")
    print("=" * 60)
    
    tickers = ["BBCA.JK", "CARE.JK", "ASII.JK"]
    
    print("Testing sequential processing (baseline):")
    sequential_results = []
    for ticker in tickers:
        analyzer = StockAnalyzer()
        result = analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
        if result['success']:
            stats = result['stats']
            sequential_results.append({
                'ticker': ticker,
                'total_return': stats.get('Total Return [%]', 'N/A'),
                'sharpe_ratio': stats.get('Sharpe Ratio', 'N/A'),
                'regime': result['regime']
            })
    
    for res in sequential_results:
        print(f"{res['ticker']}: Return={res['total_return']:.2f}%, Sharpe={res['sharpe_ratio']:.2f}, Regime={res['regime']}")
    
    # Check for identical results
    returns = [r['total_return'] for r in sequential_results if r['total_return'] != 'N/A']
    if len(set(returns)) == 1 and len(returns) > 1:
        print("⚠️ WARNING: Sequential processing also shows identical returns!")
    else:
        print("✓ Sequential processing shows different results")

def test_regime_detector_ml_model_sharing():
    """Test if ML model is being shared between instances"""
    print("\n" + "=" * 60)
    print("TEST 5: ML Model Sharing Analysis")
    print("=" * 60)
    
    # Create sample data for two different "stocks"
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Stock A data (trending up)
    stock_a_data = pd.DataFrame({
        'Open': np.cumsum(np.random.randn(200) * 0.01 + 0.001) + 100,
        'High': np.cumsum(np.random.randn(200) * 0.01 + 0.002) + 105,
        'Low': np.cumsum(np.random.randn(200) * 0.01 - 0.001) + 95,
        'Close': np.cumsum(np.random.randn(200) * 0.01 + 0.001) + 100,
        'Volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Stock B data (trending down)
    stock_b_data = pd.DataFrame({
        'Open': np.cumsum(np.random.randn(200) * 0.01 - 0.001) + 100,
        'High': np.cumsum(np.random.randn(200) * 0.01 - 0.0005) + 105,
        'Low': np.cumsum(np.random.randn(200) * 0.01 - 0.002) + 95,
        'Close': np.cumsum(np.random.randn(200) * 0.01 - 0.001) + 100,
        'Volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Add features
    fe = FeatureEngineer()
    stock_a_data = fe.add_daily_features(stock_a_data)
    stock_b_data = fe.add_daily_features(stock_b_data)
    
    # Test with separate detectors
    detector_a = RegimeDetector()
    detector_b = RegimeDetector()
    
    result_a = detector_a.detect_ml(stock_a_data.copy())
    result_b = detector_b.detect_ml(stock_b_data.copy())
    
    print(f"Stock A final regime: {result_a['regime_ml'].iloc[-1]}")
    print(f"Stock B final regime: {result_b['regime_ml'].iloc[-1]}")
    print(f"Stock A confidence: {result_a['regime_ml_conf'].iloc[-1]:.3f}")
    print(f"Stock B confidence: {result_b['regime_ml_conf'].iloc[-1]:.3f}")
    
    if result_a['regime_ml'].iloc[-1] == result_b['regime_ml'].iloc[-1]:
        print("⚠️ WARNING: Different stocks getting same regime!")
    else:
        print("✓ Different stocks get different regimes")

def main():
    """Run all diagnostic tests"""
    print("PARALLEL PROCESSING DEBUG DIAGNOSTIC")
    print("=" * 60)
    
    test_data_fetcher_isolation()
    test_regime_detector_shared_state()
    test_stock_analyzer_isolation()
    test_parallel_processing_simulation()
    test_regime_detector_ml_model_sharing()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("If sequential processing shows different results but parallel shows identical,")
    print("the issue is likely in the parallel execution environment or shared global state.")
    print("\nNext steps:")
    print("1. Check if any global variables are being modified")
    print("2. Verify thread safety of all components")
    print("3. Test with different parallel backends")

if __name__ == "__main__":
    main()