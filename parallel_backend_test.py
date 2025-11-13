#!/usr/bin/env python3
"""
Test different joblib backends to identify the best solution for parallel processing.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from stock_analyzer import StockAnalyzer
import time

def process_ticker_isolated(ticker: str, use_cache: bool = True) -> dict:
    """
    Completely isolated wrapper function for parallel processing.
    Creates fresh instances and imports to ensure no shared state.
    """
    # Import inside function to ensure fresh imports in each worker
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    from stock_analyzer import StockAnalyzer
    from data_layer import DataFetcher
    
    # Create completely fresh instances
    analyzer = StockAnalyzer()
    
    # Clear any existing cache
    analyzer.data_fetcher.clear_cache()
    
    result = analyzer.analyze_ticker(ticker, use_cache=use_cache, verbose=False)
    
    # Return simplified result
    if result['success']:
        stats = result['stats']
        return {
            'ticker': result['ticker'],
            'regime': result['regime'],
            'conf': result['conf'],
            'stats': {
                'Total Return [%]': stats.get('Total Return [%]', 'N/A'),
                'Sharpe Ratio': stats.get('Sharpe Ratio', 'N/A'),
                'Max Drawdown [%]': stats.get('Max Drawdown [%]', 'N/A'),
                'Win Rate [%]': stats.get('Win Rate [%]', 'N/A'),
                'Total Trades': stats.get('Total Trades', 'N/A')
            },
            'success': True
        }
    else:
        return {
            'ticker': result['ticker'],
            'error': result['error'],
            'success': False
        }

def test_sequential_baseline():
    """Test sequential processing as baseline."""
    print("SEQUENTIAL PROCESSING (BASELINE)")
    print("=" * 50)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    start_time = time.time()
    results = []
    for ticker in tickers:
        result = process_ticker_isolated(ticker, use_cache=False)
        results.append(result)
    sequential_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
    regimes = [r['regime'] for r in successful_results]
    
    print(f"Time: {sequential_time:.2f}s")
    print(f"Returns: {returns}")
    print(f"Regimes: {regimes}")
    
    if len(set(returns)) == 1 and len(returns) > 1:
        print("⚠️ WARNING: Sequential processing shows identical returns!")
    else:
        print("✓ Sequential processing shows different returns")
    
    return sequential_time, results

def test_threading_backend():
    """Test threading backend (current problematic configuration)."""
    print("\nTHREADING BACKEND")
    print("=" * 50)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    start_time = time.time()
    results = Parallel(n_jobs=3, backend="threading")(
        delayed(process_ticker_isolated)(ticker, use_cache=False)
        for ticker in tickers
    )
    threading_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
    regimes = [r['regime'] for r in successful_results]
    
    print(f"Time: {threading_time:.2f}s")
    print(f"Returns: {returns}")
    print(f"Regimes: {regimes}")
    
    if len(set(returns)) == 1 and len(returns) > 1:
        print("⚠️ WARNING: Threading backend shows identical returns!")
    else:
        print("✓ Threading backend shows different returns")
    
    return threading_time, results

def test_multiprocessing_backend():
    """Test multiprocessing backend (potential solution)."""
    print("\nMULTIPROCESSING BACKEND")
    print("=" * 50)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    start_time = time.time()
    results = Parallel(n_jobs=3, backend="multiprocessing")(
        delayed(process_ticker_isolated)(ticker, use_cache=False)
        for ticker in tickers
    )
    multiprocessing_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
    regimes = [r['regime'] for r in successful_results]
    
    print(f"Time: {multiprocessing_time:.2f}s")
    print(f"Returns: {returns}")
    print(f"Regimes: {regimes}")
    
    if len(set(returns)) == 1 and len(returns) > 1:
        print("⚠️ WARNING: Multiprocessing backend shows identical returns!")
    else:
        print("✓ Multiprocessing backend shows different returns")
    
    return multiprocessing_time, results

def test_loky_backend():
    """Test loky backend (modern multiprocessing alternative)."""
    print("\nLOKY BACKEND")
    print("=" * 50)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    start_time = time.time()
    results = Parallel(n_jobs=3, backend="loky")(
        delayed(process_ticker_isolated)(ticker, use_cache=False)
        for ticker in tickers
    )
    loky_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
    regimes = [r['regime'] for r in successful_results]
    
    print(f"Time: {loky_time:.2f}s")
    print(f"Returns: {returns}")
    print(f"Regimes: {regimes}")
    
    if len(set(returns)) == 1 and len(returns) > 1:
        print("⚠️ WARNING: Loky backend shows identical returns!")
    else:
        print("✓ Loky backend shows different returns")
    
    return loky_time, results

def main():
    """Run all backend tests and compare results."""
    print("PARALLEL BACKEND COMPARISON TEST")
    print("=" * 60)
    
    # Run all tests
    sequential_time, seq_results = test_sequential_baseline()
    threading_time, thread_results = test_threading_backend()
    multiprocessing_time, mp_results = test_multiprocessing_backend()
    loky_time, loky_results = test_loky_backend()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Sequential:      {sequential_time:.2f}s")
    print(f"Threading:       {threading_time:.2f}s")
    print(f"Multiprocessing: {multiprocessing_time:.2f}s")
    print(f"Loky:            {loky_time:.2f}s")
    
    # Check for data consistency
    print("\n" + "=" * 60)
    print("DATA CONSISTENCY CHECK")
    print("=" * 60)
    
    def extract_key_results(results):
        key_data = {}
        for result in results:
            if result['success']:
                key_data[result['ticker']] = {
                    'return': result['stats'].get('Total Return [%]', 'N/A'),
                    'regime': result['regime']
                }
        return key_data
    
    seq_data = extract_key_results(seq_results)
    thread_data = extract_key_results(thread_results)
    mp_data = extract_key_results(mp_results)
    loky_data = extract_key_results(loky_results)
    
    print("Sequential baseline:")
    for ticker, data in seq_data.items():
        print(f"  {ticker}: Return={data['return']:.2f}%, Regime={data['regime']}")
    
    print("\nThreading vs Sequential:")
    for ticker in seq_data:
        if ticker in thread_data:
            seq_return = seq_data[ticker]['return']
            thread_return = thread_data[ticker]['return']
            if seq_return != thread_return:
                print(f"  {ticker}: MISMATCH - Seq={seq_return:.2f}%, Thread={thread_return:.2f}%")
            else:
                print(f"  {ticker}: MATCH - {seq_return:.2f}%")
    
    print("\nMultiprocessing vs Sequential:")
    for ticker in seq_data:
        if ticker in mp_data:
            seq_return = seq_data[ticker]['return']
            mp_return = mp_data[ticker]['return']
            if seq_return != mp_return:
                print(f"  {ticker}: MISMATCH - Seq={seq_return:.2f}%, MP={mp_return:.2f}%")
            else:
                print(f"  {ticker}: MATCH - {seq_return:.2f}%")

if __name__ == "__main__":
    main()