import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from stock_analyzer import StockAnalyzer

def analyze_data_quality():
    """Analyze data quality and structure for multiple tickers."""
    print("=== DATA QUALITY ANALYSIS ===")
    
    tickers = ["BBCA", "CARE", "ASII"]
    analyzer = StockAnalyzer()
    
    for ticker in tickers:
        print(f"\n--- Analyzing {ticker} ---")
        
        try:
            result = analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
            
            if result["success"]:
                df_intraday = result["intraday_data"]
                stats = result["stats"]
                
                print(f"Data shape: {df_intraday.shape}")
                print(f"Date range: {df_intraday.index.min()} to {df_intraday.index.max()}")
                
                # Check for data gaps
                time_diff = df_intraday.index.to_series().diff()
                gaps = time_diff[time_diff > pd.Timedelta('2h')]
                print(f"Data gaps (>2h): {len(gaps)}")
                if len(gaps) > 0:
                    print(f"Sample gaps: {gaps.head(3)}")
                
                # Check Close price statistics
                close_prices = df_intraday["Close"]
                print(f"Close price stats:")
                print(f"  Min: {close_prices.min():.2f}")
                print(f"  Max: {close_prices.max():.2f}")
                print(f"  Mean: {close_prices.mean():.2f}")
                print(f"  Std: {close_prices.std():.2f}")
                
                # Check regime distribution
                regime_counts = df_intraday["regime_ml"].value_counts()
                print(f"Regime distribution: {dict(regime_counts)}")
                
                print(f"Total Return: {stats.get('Total Return [%]', 'N/A')}")
                print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
                
            else:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

def test_backtesting_with_frequency():
    """Test backtesting with explicit frequency parameter."""
    print("\n=== BACKTESTING WITH FREQUENCY TEST ===")
    
    from backtesting import Backtester
    
    backtester = Backtester()
    
    # Test with BBCA data
    analyzer = StockAnalyzer()
    result = analyzer.analyze_ticker("BBCA", use_cache=False, verbose=False)
    
    if result["success"]:
        df_intraday = result["intraday_data"]
        
        # Test without frequency (current behavior)
        portfolio_no_freq = backtester.backtest_regime_strategy(df_intraday)
        stats_no_freq = portfolio_no_freq.stats()
        
        print("Without explicit frequency:")
        print(f"  Total Return: {stats_no_freq.get('Total Return [%]', 'N/A')}")
        print(f"  Sharpe Ratio: {stats_no_freq.get('Sharpe Ratio', 'N/A')}")
        print(f"  Calmar Ratio: {stats_no_freq.get('Calmar Ratio', 'N/A')}")
        
        # Test with frequency parameter
        portfolio_with_freq = backtester.backtest_regime_strategy(
            df_intraday, 
            frequency='1h'  # Explicit frequency
        )
        stats_with_freq = portfolio_with_freq.stats()
        
        print("With explicit frequency:")
        print(f"  Total Return: {stats_with_freq.get('Total Return [%]', 'N/A')}")
        print(f"  Sharpe Ratio: {stats_with_freq.get('Sharpe Ratio', 'N/A')}")
        print(f"  Calmar Ratio: {stats_with_freq.get('Calmar Ratio', 'N/A')}")

def check_data_consistency():
    """Check if data is consistent across tickers."""
    print("\n=== DATA CONSISTENCY CHECK ===")
    
    tickers = ["BBCA", "CARE", "ASII"]
    analyzer = StockAnalyzer()
    
    all_data = {}
    
    for ticker in tickers:
        result = analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
        
        if result["success"]:
            df_intraday = result["intraday_data"]
            all_data[ticker] = {
                'data': df_intraday,
                'stats': result["stats"],
                'date_range': (df_intraday.index.min(), df_intraday.index.max())
            }
    
    # Compare date ranges
    print("Date ranges comparison:")
    for ticker, data in all_data.items():
        start, end = data['date_range']
        print(f"  {ticker}: {start.date()} to {end.date()}")
    
    # Compare returns
    print("\nReturns comparison:")
    for ticker, data in all_data.items():
        stats = data['stats']
        print(f"  {ticker}: {stats.get('Total Return [%]', 'N/A')}%")

def main():
    """Run detailed diagnostics."""
    print("DETAILED FINANCIAL METRICS DIAGNOSTIC")
    print("=" * 60)
    
    analyze_data_quality()
    test_backtesting_with_frequency()
    check_data_consistency()
    
    print("\n" + "=" * 60)
    print("DETAILED DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()