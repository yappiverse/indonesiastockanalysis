import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from stock_analyzer import StockAnalyzer

def process_ticker_with_logging(ticker: str, use_cache: bool = True) -> dict:
    """
    Wrapper function with detailed logging to track state
    """
    print(f"=== Starting analysis for {ticker} ===")
    
    analyzer = StockAnalyzer()
    
    # Log initial state
    print(f"  Analyzer ID: {id(analyzer)}")
    print(f"  RegimeDetector ID: {id(analyzer.regime_detector)}")
    print(f"  ML Pipeline ID: {id(analyzer.regime_detector._ml_pipeline)}")
    print(f"  ML Pipeline is None: {analyzer.regime_detector._ml_pipeline is None}")
    print(f"  Is fitted: {analyzer.regime_detector._is_fitted}")
    
    result = analyzer.analyze_ticker(ticker, use_cache=use_cache, verbose=False)
    
    # Log final state
    print(f"  Final ML Pipeline ID: {id(analyzer.regime_detector._ml_pipeline)}")
    print(f"  Final is fitted: {analyzer.regime_detector._is_fitted}")
    
    if result['success']:
        stats = result['stats']
        print(f"  Result - Return: {stats.get('Total Return [%]', 'N/A')}, Sharpe: {stats.get('Sharpe Ratio', 'N/A')}, Regime: {result['regime']}")
    else:
        print(f"  Result - Error: {result['error']}")
    
    print(f"=== Finished analysis for {ticker} ===\n")
    
    return result

def test_sequential_processing():
    """Test sequential processing as baseline"""
    print("SEQUENTIAL PROCESSING TEST")
    print("=" * 60)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    results = []
    for ticker in tickers:
        result = process_ticker_with_logging(ticker, use_cache=False)
        results.append(result)
    
    # Check for identical results
    successful_results = [r for r in results if r['success']]
    if successful_results:
        returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
        regimes = [r['regime'] for r in successful_results]
        
        print(f"Returns: {returns}")
        print(f"Regimes: {regimes}")
        
        if len(set(returns)) == 1 and len(returns) > 1:
            print("⚠️ WARNING: Sequential processing shows identical returns!")
        else:
            print("✓ Sequential processing shows different returns")
        
        if len(set(regimes)) == 1 and len(regimes) > 1:
            print("⚠️ WARNING: Sequential processing shows identical regimes!")
        else:
            print("✓ Sequential processing shows different regimes")

def test_parallel_processing():
    """Test parallel processing to reproduce the issue"""
    print("\nPARALLEL PROCESSING TEST")
    print("=" * 60)
    
    tickers = ["BBCA", "CARE", "ASII"]
    
    print("Running parallel processing with threading backend...")
    results = Parallel(n_jobs=3, backend="threading")(
        delayed(process_ticker_with_logging)(ticker, use_cache=False)
        for ticker in tickers
    )
    
    # Check for identical results
    successful_results = [r for r in results if r['success']]
    if successful_results:
        returns = [r['stats'].get('Total Return [%]', 'N/A') for r in successful_results]
        regimes = [r['regime'] for r in successful_results]
        
        print(f"Returns: {returns}")
        print(f"Regimes: {regimes}")
        
        if len(set(returns)) == 1 and len(returns) > 1:
            print("⚠️ WARNING: Parallel processing shows identical returns!")
        else:
            print("✓ Parallel processing shows different returns")
        
        if len(set(regimes)) == 1 and len(regimes) > 1:
            print("⚠️ WARNING: Parallel processing shows identical regimes!")
        else:
            print("✓ Parallel processing shows different regimes")

def test_regime_detector_initialization():
    """Test if RegimeDetector instances are properly initialized"""
    print("\nREGIME DETECTOR INITIALIZATION TEST")
    print("=" * 60)
    
    from regime_detection import RegimeDetector
    
    detectors = [RegimeDetector() for _ in range(3)]
    
    print("Initial state of multiple detectors:")
    for i, detector in enumerate(detectors):
        print(f"  Detector {i}: id={id(detector)}, pipeline_id={id(detector._ml_pipeline)}, is_fitted={detector._is_fitted}")
    
    # Check if they share the same None object
    pipeline_ids = [id(d._ml_pipeline) for d in detectors]
    if len(set(pipeline_ids)) == 1:
        print("⚠️ WARNING: All detectors share the same initial _ml_pipeline object!")
    else:
        print("✓ Detectors have different initial _ml_pipeline objects")

def main():
    """Run all tests"""
    print("PARALLEL PROCESSING REPRODUCTION TEST")
    print("=" * 60)
    
    test_regime_detector_initialization()
    test_sequential_processing()
    test_parallel_processing()

if __name__ == "__main__":
    main()