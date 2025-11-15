import argparse
import os
from datetime import datetime
from joblib import Parallel, delayed
from stock_analyzer import StockAnalyzer
from summary_analyzer import SummaryAnalyzer
from data_layer import CSVExporter
from config import PARALLEL_CONFIG


def process_ticker_wrapper(ticker: str, use_cache: bool = True) -> dict:
    """
    Wrapper function for parallel processing of tickers.
    This function creates a completely isolated environment for each analysis.
    
    Args:
        ticker: Stock ticker symbol
        use_cache: Whether to use cached data
        
    Returns:
        Analysis results dictionary with summary information and serializable data
    """
    # Import inside function to ensure fresh imports in each thread
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    from stock_analyzer import StockAnalyzer
    from summary_analyzer import SummaryAnalyzer
    from data_layer import DataFetcher
    from backtesting import PerformanceReporter
    
    # Create completely fresh instances to avoid thread contamination
    analyzer = StockAnalyzer()
    summary_analyzer = SummaryAnalyzer()
    
    # Clear any existing cache to ensure fresh data
    analyzer.data_fetcher.clear_cache()
    
    result = analyzer.analyze_ticker(ticker, use_cache=use_cache, verbose=False)
    
    # Generate comprehensive summary
    summary = summary_analyzer.generate_summary(result)
    
    # Generate backtest report for detailed output if needed
    backtest_report = None
    if result.get('portfolio'):
        performance_reporter = PerformanceReporter()
        portfolio = result['portfolio']
        backtest_report = performance_reporter.generate_summary_report(
            portfolio, ticker, summary['regime'], summary['regime_confidence']
        )
    
    # Get latest price data including opening price
    price_data = None
    try:
        # Import inside function to ensure fresh imports in each thread
        from data_layer import DataFetcher
        data_fetcher = DataFetcher()
        price_data = data_fetcher.get_latest_price(summary['ticker'])
    except Exception as e:
        price_data = {
            'ticker': summary['ticker'],
            'opening_price': None,
            'success': False,
            'error': str(e)
        }

    # Return only serializable data for parallel processing
    return {
        'ticker': summary['ticker'],
        'recommendation': summary['recommendation'],
        'confidence': summary['confidence'],
        'risk': summary['risk'],
        'rationale': summary['rationale'],
        'key_metrics': summary['key_metrics'],
        'regime': summary['regime'],
        'regime_confidence': summary['regime_confidence'],
        'next_day_action': summary['next_day_action'],
        'summary_score': summary['summary_score'],
        'success': summary.get('success', True),
        'backtest_report': backtest_report,  # Store pre-generated report instead of reporter object
        'has_portfolio': bool(result.get('portfolio')),  # Flag to indicate if portfolio data exists
        'price_data': price_data  # Include latest price data
    }


def main():
    """Main entry point for the stock analysis system."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis System - Regime Detection and Backtesting"
    )
    parser.add_argument(
        "tickers", 
        nargs="+", 
        help="List of ticker symbols (e.g., BBCA CARE ASII)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=PARALLEL_CONFIG["default_jobs"],
        help="Number of parallel workers (default: -1 for all cores)"
    )
    parser.add_argument(
        "--backend",
        choices=["MP", "LOCKY"],
        default="LOCKY",
        help="Parallel processing backend: MP (multiprocessing) or LOCKY (loky) (default: MP)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable data caching"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed backtest table for each ticker"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV file with automatic filename (output/ddmmyyyy - hhmmss.csv)"
    )
    
    args = parser.parse_args()
    
    print("\n Starting Multi-Ticker Analysis")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Parallel workers: {args.jobs}")
    print(f"Backend: {args.backend}")
    print(f"Data caching: {'enabled' if not args.no_cache else 'disabled'}")
    print(f"Detailed output: {'enabled' if args.detailed else 'disabled'}")
    
    # Use multiprocessing backend for parallel processing to ensure data isolation
    # Threading backend causes data contamination due to shared ML pipeline state
    if args.jobs != 1:
        # Map backend argument to actual backend names
        backend_map = {
            "MP": "multiprocessing",
            "LOCKY": "loky"
        }
        actual_backend = backend_map[args.backend]
        
        print(f"\nProcessing {len(args.tickers)} tickers in parallel using {args.jobs} workers...")
        print(f"Backend: {args.backend} ({actual_backend})")
        
        # Use selected backend for parallel processing
        results = Parallel(n_jobs=args.jobs, backend=actual_backend)(
            delayed(process_ticker_wrapper)(ticker, use_cache=not args.no_cache)
            for ticker in args.tickers
        )
    else:
        print("\nProcessing tickers sequentially...")
        results = []
        for i, ticker in enumerate(args.tickers, 1):
            print(f"Processing {ticker} ({i}/{len(args.tickers)})...")
            result = process_ticker_wrapper(ticker, use_cache=not args.no_cache)
            results.append(result)
    
    # Generate comprehensive summary report
    print("\n\n" + "="*50 + " COMPREHENSIVE SUMMARY " + "="*50)
    
    successful_results = []
    failed_results = []
    
    for result in results:
        if result.get("success", False):
            successful_results.append(result)
        else:
            failed_results.append(result)
    
    # Print successful results with formatted summaries
    if successful_results:
        print(f"\n📊 SUCCESSFUL ANALYSES ({len(successful_results)} stocks):")
        print("="*80)
        
        # Sort by summary score (highest first)
        successful_results.sort(key=lambda x: x.get('summary_score', 0), reverse=True)
        
        for result in successful_results:
            # Recommendation emoji mapping
            rec_emoji = {
                "STRONG BUY": "🚀",
                "BUY": "📈",
                "HOLD": "⚖️",
                "SELL": "📉",
                "STRONG SELL": "💥"
            }
            
            # Risk emoji mapping
            risk_emoji = {
                "VERY LOW": "🟢",
                "LOW": "🟡",
                "MODERATE": "🟠",
                "HIGH": "🔴",
                "VERY HIGH": "💀"
            }
            
            emoji = rec_emoji.get(result["recommendation"], "📊")
            risk_icon = risk_emoji.get(result["risk"], "⚫")
            
            print(f"\n{emoji} {result['ticker']} - {result['recommendation']} {emoji}")
            print(f"   Next Day: {result['next_day_action']}")
            print(f"   Confidence: {result['confidence']} | {risk_icon} Risk: {result['risk']}")
            print(f"   Score: {'★' * result['summary_score']}{'☆' * (5 - result['summary_score'])}")
            print(f"   Regime: {result['regime']} (confidence: {result['regime_confidence']:.2f})")
            
            # Print key metrics
            metrics = result['key_metrics']
            total_return = metrics.get('total_return', 'N/A')
            sharpe_ratio = metrics.get('sharpe_ratio', 'N/A')
            win_rate = metrics.get('win_rate', 'N/A')
            
            # Format numeric values, leave strings as-is
            returns_str = f"{total_return:.2f}%" if isinstance(total_return, (int, float)) else str(total_return)
            sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else str(sharpe_ratio)
            win_rate_str = f"{win_rate:.1f}%" if isinstance(win_rate, (int, float)) else str(win_rate)
            
            print(f"   Returns: {returns_str} | "
                  f"Sharpe: {sharpe_str} | "
                  f"Win Rate: {win_rate_str}")
            
            # Print full rationale without truncation
            rationale = result['rationale']
            print(f"   Rationale: {rationale}")
            
            # Print full backtest results if --detailed flag is set
            if args.detailed and result.get('backtest_report') and result['success']:
                print(f"\n   📊 FULL BACKTEST RESULTS:")
                print(f"   " + "="*40)
                
                # Display the pre-generated backtest report
                backtest_report = result['backtest_report']
                for line in backtest_report.strip().split('\n'):
                    print(f"   {line}")
                
                print(f"   " + "="*40)
    
    # Print failed results
    if failed_results:
        print(f"\n❌ FAILED ANALYSES ({len(failed_results)} stocks):")
        for result in failed_results:
            print(f"   {result['ticker']}: {result.get('rationale', 'Unknown error')}")
    
    # Print overall statistics
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total tickers processed: {len(results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(failed_results)}")
    print(f"   Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    # Calculate recommendation distribution
    if successful_results:
        rec_counts = {}
        for result in successful_results:
            rec = result['recommendation']
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        print(f"\n🎯 RECOMMENDATION DISTRIBUTION:")
        for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_results)) * 100
            print(f"   {rec}: {count} stocks ({percentage:.1f}%)")
    
    # Print system information
    print(f"\n⚙️  SYSTEM INFORMATION:")
    print(f"   Parallel workers: {args.jobs}")
    if args.jobs != 1:
        backend_map = {
            "MP": "multiprocessing",
            "LOCKY": "loky"
        }
        actual_backend = backend_map[args.backend]
        print(f"   Backend: {args.backend} ({actual_backend})")
    else:
        print(f"   Backend: sequential")
    print(f"   Data isolation: {'✓ ensured' if args.jobs != 1 else '✓ sequential processing'}")
    
    # Export to CSV if requested
    if args.csv:
        # Generate automatic filename with current date/time
        current_time = datetime.now()
        filename = current_time.strftime("output/%y%m%d - %H%M%S.csv")
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        print(f"\n💾 EXPORTING TO CSV: {filename}")
        try:
            csv_exporter = CSVExporter()
            csv_path = csv_exporter.export_to_csv(results, filename)
            print(f"   ✅ Successfully exported {len(results)} results to: {csv_path}")
            print(f"   📊 Columns: no, ticker, current_price, regime, confidence, action, risk, score")
        except Exception as e:
            print(f"   ❌ Failed to export CSV: {e}")


if __name__ == "__main__":
    main()