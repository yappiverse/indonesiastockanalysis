#!/usr/bin/env python3
"""
Deep diagnostic to identify the exact source of data contamination
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from stock_analyzer import StockAnalyzer
from data_layer import DataFetcher
import threading

# Global variable to track contamination
analysis_data = {}

def process_ticker_with_deep_logging(ticker: str, use_cache: bool = True) -> dict:
    """
    Wrapper function with deep logging to track all state changes
    """
    thread_name = threading.current_thread().name
    print(f"\n=== Thread {thread_name}: Starting analysis for {ticker} ===")
    
    analyzer = StockAnalyzer()
    
    # Log initial state in detail
    print(f"  Thread {thread_name} - Analyzer ID: {id(analyzer)}")
    print(f"  Thread {thread_name} - RegimeDetector ID: {id(analyzer.regime_detector)}")
    print(f"  Thread {thread_name} - ML Pipeline ID: {id(analyzer.regime_detector._ml_pipeline)}")
    print(f"  Thread {thread_name} - ML Pipeline is None: {analyzer.regime_detector._ml_pipeline is None}")
    print(f"  Thread {thread_name} - Is fitted: {analyzer.regime_detector._is_fitted}")
    
    # Store initial state for comparison
    initial_state = {
        'analyzer_id': id(analyzer),
        'detector_id': id(analyzer.regime_detector),
        'pipeline_id': id(analyzer.regime_detector._ml_pipeline),
        'is_fitted': analyzer.regime_detector._is_fitted
    }
    
    # Track data fetching
    print(f"  Thread {thread_name} - Fetching data for {ticker}...")
    try:
        # Get data directly to check if data is the issue
        data_fetcher = DataFetcher()
        daily_data = data_fetcher.get_ohlcv(ticker + ".JK", use_cache=False)
        print(f"  Thread {thread_name} - Data shape: {daily_data.shape}")
        print(f"  Thread {thread_name} - First close: {daily_data['Close'].iloc[0]}")
        print(f"  Thread {thread_name} - Last close: {daily_data['Close'].iloc[-1]}")
        
        # Store data for comparison
        analysis_data[f"{thread_name}_{ticker}"] = {
            'data_shape': daily_data.shape,
            'first_close': daily_data['Close'].iloc[0],
            'last_close': daily_data['Close'].iloc[-1]
        }
    except Exception as e:
        print(f"  Thread {thread_name} - Data fetch error: {e}")
    
    # Run analysis
    result = analyzer.analyze_ticker(ticker, use_cache=use_cache, verbose=False)
    
    # Log final state
    final_state = {
        'analyzer_id': id(analyzer),
        'detector_id': id(analyzer.regime_detector),
        'pipeline_id': id(analyzer.regime_detector._ml_pipeline),
        'is_fitted': analyzer.regime_detector._is_fitted
    }
    
    print(f"  Thread {thread_name} - Final ML Pipeline ID: {id(analyzer.regime_detector._ml_pipeline)}")
    print(f"  Thread {thread_name} - Final is fitted: {analyzer.regime_detector._is_fitted}")
    
    # Check if state changed unexpectedly
    if initial_state['pipeline_id'] == final_state['pipeline_id'] and initial_state['pipeline_id'] != id(None):
        print(f"  ⚠️ Thread {thread_name} - WARNING: ML Pipeline ID did not change!")
    
    if result['success']:
        stats = result['stats']
        print(f"  Thread {thread_name} - Result - Return: {stats.get('Total Return [%]', 'N/A')}, Sharpe: {stats.get('Sharpe Ratio', 'N/A')}, Regime: {result['regime']}")
    else:
        print(f"  Thread {thread_name} - Result - Error: {result['error']}")
    
    print(f"=== Thread {thread_name}: Finished analysis for {ticker} ===")
    
    return result

def test_different_parallel_backends():
    """Test with different parallel backends to isolate the issue"""
    tickers = ["BBCA", "CARE", "ASII"]
    
    print("TESTING DIFFERENT PARALLEL BACKENDS")
    print("=" * 60)
    
    # Test 1: Threading backend (current problematic one)
    print("\n1. THREADING BACKEND:")
    results_threading = Parallel(n_jobs=3, backend="threading")(
        delayed(process_ticker_with_deep_logging)(ticker, use_cache=False)
        for ticker in tickers
    )
    check_identical_results(results_threading, "Threading")
    
    # Clear analysis data
    analysis_data.clear()
    
    # Test 2: Multiprocessing backend (should be safer)
    print("\n2. MULTIPROCESSING BACKEND:")
    try:
        results_multiprocessing = Parallel(n_jobs=3, backend="multiprocessing")(
            delayed(process_ticker_with_deep_logging)(ticker, use_cache=False)
            for ticker in tickers
        )
        check_identical_results(results_multiprocessing, "Multiprocessing")
    except Exception as e:
        print(f"  Multiprocessing failed: {e}")
    
    # Clear analysis data
    analysis_data.clear()
    
    # Test 3: Sequential (baseline)
    print("\n3. SEQUENTIAL (BASELINE):")
    results_sequential = []
    for ticker in tickers:
        results_sequential.append(process_ticker_with_deep_logging(ticker, use_cache=False))
    check_identical_results(results_sequential, "Sequential")

def check_identical_results(results, backend_name):
    """Check if results are identical and print analysis"""
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
        regimes = [r['regime'] for r in successful_results]
        
        print(f"  {backend_name} - Returns: {returns}")
        print(f"  {backend_name} - Regimes: {regimes}")
        
        if len(set(returns)) == 1 and len(returns) > 1:
            print(f"  ⚠️ {backend_name} - WARNING: Identical returns!")
        else:
            print(f"  ✓ {backend_name} - Different returns")
        
        if len(set(regimes)) == 1 and len(regimes) > 1:
            print(f"  ⚠️ {backend_name} - WARNING: Identical regimes!")
        else:
            print(f"  ✓ {backend_name} - Different regimes")
        
        # Check data contamination
        print(f"  {backend_name} - Data analysis:")
        for key, data in analysis_data.items():
            print(f"    {key}: shape={data['data_shape']}, first_close={data['first_close']}, last_close={data['last_close']}")

def test_regime_detector_race_condition():
    """Test if there's a race condition in RegimeDetector initialization"""
    print("\n" + "=" * 60)
    print("RACE CONDITION TEST")
    print("=" * 60)
    
    from regime_detection import RegimeDetector
    
    # Create multiple detectors simultaneously
    detectors = []
    for i in range(5):
        detector = RegimeDetector()
        detectors.append(detector)
        print(f"Detector {i}: id={id(detector)}, pipeline_id={id(detector._ml_pipeline)}")
    
    # Check if they all share the same None
    pipeline_ids = [id(d._ml_pipeline) for d in detectors]
    unique_pipeline_ids = set(pipeline_ids)
    
    print(f"Unique pipeline IDs: {unique_pipeline_ids}")
    print(f"All share same None? {len(unique_pipeline_ids) == 1}")
    
    if len(unique_pipeline_ids) == 1:
        print("⚠️ RACE CONDITION CONFIRMED: All detectors share the same initial None object!")

def main():
    """Run all diagnostic tests"""
    print("DEEP DIAGNOSTIC FOR PARALLEL PROCESSING ISSUE")
    print("=" * 60)
    
    test_regime_detector_race_condition()
    test_different_parallel_backends()

if __name__ == "__main__":
    main()