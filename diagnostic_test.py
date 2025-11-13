#!/usr/bin/env python3
"""
Diagnostic test to identify financial metrics calculation issues.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import vectorbt as vbt
from stock_analyzer import StockAnalyzer
from backtesting import Backtester

def test_backtesting_frequency():
    """Test backtesting with frequency configuration."""
    print("=== BACKTESTING FREQUENCY DIAGNOSTIC ===")
    
    # Create backtester instance
    backtester = Backtester()
    
    # Test with sample data
    print("\n1. Testing backtester configuration:")
    print(f"   Config: {backtester.config}")
    
    # Check if frequency is properly set
    frequency = backtester.config.get("frequency")
    print(f"   Frequency in config: {frequency}")
    
    return backtester

def test_data_structure():
    """Test data structure and datetime indexing."""
    print("\n2. Testing data structure:")
    
    analyzer = StockAnalyzer()
    
    # Test with a single ticker
    try:
        result = analyzer.analyze_ticker("BBCA", use_cache=False, verbose=False)
        
        if result["success"]:
            df_intraday = result["intraday_data"]
            portfolio = result["portfolio"]
            stats = result["stats"]
            
            print(f"   Intraday data shape: {df_intraday.shape}")
            print(f"   Index type: {type(df_intraday.index)}")
            print(f"   Index frequency: {pd.infer_freq(df_intraday.index)}")
            print(f"   Index sample: {df_intraday.index[:5]}")
            
            print(f"\n   Portfolio stats keys: {list(stats.keys())[:10]}...")
            print(f"   Total Return: {stats.get('Total Return [%]', 'N/A')}")
            print(f"   Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
            print(f"   Max Drawdown: {stats.get('Max Drawdown [%]', 'N/A')}")
            
            # Check portfolio object
            print(f"\n   Portfolio type: {type(portfolio)}")
            print(f"   Portfolio frequency: {getattr(portfolio, 'freq', 'Not set')}")
            
        else:
            print(f"   Error: {result['error']}")
            
    except Exception as e:
        print(f"   Error in diagnostic: {e}")

def test_vectorbt_frequency():
    """Test vectorbt frequency requirements directly."""
    print("\n3. Testing vectorbt frequency requirements:")
    
    # Create sample data with proper datetime index
    dates = pd.date_range('2020-01-01', periods=1000, freq='1h')
    prices = pd.Series(range(100, 1100), index=dates)
    
    # Test portfolio without frequency
    portfolio_no_freq = vbt.Portfolio.from_holding(prices, init_cash=1000000)
    stats_no_freq = portfolio_no_freq.stats()
    
    print(f"   Without frequency setting:")
    print(f"     Sharpe Ratio: {stats_no_freq.get('Sharpe Ratio', 'N/A')}")
    print(f"     Calmar Ratio: {stats_no_freq.get('Calmar Ratio', 'N/A')}")
    
    # Test portfolio with frequency
    portfolio_with_freq = vbt.Portfolio.from_holding(prices, init_cash=1000000, freq='1h')
    stats_with_freq = portfolio_with_freq.stats()
    
    print(f"   With frequency setting:")
    print(f"     Sharpe Ratio: {stats_with_freq.get('Sharpe Ratio', 'N/A')}")
    print(f"     Calmar Ratio: {stats_with_freq.get('Calmar Ratio', 'N/A')}")

def main():
    """Run all diagnostic tests."""
    print("FINANCIAL METRICS DIAGNOSTIC")
    print("=" * 50)
    
    test_backtesting_frequency()
    test_data_structure()
    test_vectorbt_frequency()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()