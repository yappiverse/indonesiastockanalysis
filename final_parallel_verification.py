import warnings
warnings.filterwarnings('ignore')
import time
from joblib import Parallel, delayed
from stock_analyzer import StockAnalyzer

def process_ticker_isolated(ticker: str, use_cache: bool = True) -> dict:
    """Completely isolated wrapper function."""
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    from stock_analyzer import StockAnalyzer
    from data_layer import DataFetcher
    
    analyzer = StockAnalyzer()
    analyzer.data_fetcher.clear_cache()
    
    result = analyzer.analyze_ticker(ticker, use_cache=use_cache, verbose=False)
    
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
            },
            'success': True
        }
    else:
        return {
            'ticker': result['ticker'],
            'error': result['error'],
            'success': False
        }

def test_parallel_correctness():
    """Test that parallel processing produces correct, non-contaminated results."""
    print("FINAL PARALLEL PROCESSING VERIFICATION")
    print("=" * 60)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    # Test sequential processing (baseline)
    print("\n1. SEQUENTIAL PROCESSING (Baseline)")
    print("-" * 40)
    start_time = time.time()
    sequential_results = []
    for ticker in tickers:
        result = process_ticker_isolated(ticker, use_cache=False)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Test parallel processing
    print("\n2. PARALLEL PROCESSING (Loky Backend)")
    print("-" * 40)
    start_time = time.time()
    parallel_results = Parallel(n_jobs=3, backend="loky")(
        delayed(process_ticker_isolated)(ticker, use_cache=False)
        for ticker in tickers
    )
    parallel_time = time.time() - start_time
    
    # Analyze results
    print("\n3. RESULTS ANALYSIS")
    print("-" * 40)
    
    def extract_key_data(results):
        key_data = {}
        for result in results:
            if result['success']:
                key_data[result['ticker']] = {
                    'return': result['stats'].get('Total Return [%]', 'N/A'),
                    'regime': result['regime'],
                    'sharpe': result['stats'].get('Sharpe Ratio', 'N/A')
                }
        return key_data
    
    seq_data = extract_key_data(sequential_results)
    par_data = extract_key_data(parallel_results)
    
    print("Sequential Results:")
    for ticker, data in seq_data.items():
        print(f"  {ticker}: Return={data['return']:.2f}%, Regime={data['regime']}, Sharpe={data['sharpe']:.2f}")
    
    print("\nParallel Results:")
    for ticker, data in par_data.items():
        print(f"  {ticker}: Return={data['return']:.2f}%, Regime={data['regime']}, Sharpe={data['sharpe']:.2f}")
    
    # Check for data contamination
    print("\n4. DATA CONTAMINATION CHECK")
    print("-" * 40)
    
    contamination_found = False
    for ticker in seq_data:
        if ticker in par_data:
            seq_return = seq_data[ticker]['return']
            par_return = par_data[ticker]['return']
            seq_regime = seq_data[ticker]['regime']
            par_regime = par_data[ticker]['regime']
            
            if seq_return == par_return and seq_regime == par_regime:
                print(f"  ✓ {ticker}: Results match between sequential and parallel")
            else:
                print(f"  ✗ {ticker}: MISMATCH - Seq: {seq_return:.2f}%/{seq_regime}, Par: {par_return:.2f}%/{par_regime}")
                contamination_found = True
    
    # Check for identical results in parallel (indicating contamination)
    parallel_returns = [data['return'] for data in par_data.values()]
    parallel_regimes = [data['regime'] for data in par_data.values()]
    
    if len(set(parallel_returns)) == 1 and len(parallel_returns) > 1:
        print(f"  ✗ PARALLEL CONTAMINATION: All tickers have identical returns: {parallel_returns[0]:.2f}%")
        contamination_found = True
    else:
        print(f"  ✓ Parallel returns are different: {[f'{r:.2f}%' for r in parallel_returns]}")
    
    if len(set(parallel_regimes)) == 1 and len(parallel_regimes) > 1:
        print(f"  ✗ PARALLEL CONTAMINATION: All tickers have identical regimes: {parallel_regimes[0]}")
        contamination_found = True
    else:
        print(f"  ✓ Parallel regimes are different: {parallel_regimes}")
    
    # Performance comparison
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Speedup:         {sequential_time/parallel_time:.2f}x")
    
    # Final verdict
    print("\n6. FINAL VERDICT")
    print("-" * 40)
    if contamination_found:
        print("❌ PARALLEL PROCESSING FAILED: Data contamination detected")
    else:
        print("✅ PARALLEL PROCESSING SUCCESSFUL: No data contamination, correct results")
        print(f"   Performance improvement: {sequential_time/parallel_time:.2f}x speedup")

if __name__ == "__main__":
    test_parallel_correctness()