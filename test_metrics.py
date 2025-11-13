#!/usr/bin/env python3
"""
Test script to examine available metrics for summary system.
"""

import warnings
warnings.filterwarnings('ignore')
from stock_analyzer import StockAnalyzer

def test_metrics():
    """Test and display available metrics for summary system."""
    analyzer = StockAnalyzer()
    
    # Test with a simple ticker
    ticker = 'BBCA'
    print(f"Testing metrics for {ticker}...")
    
    result = analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
    
    if result['success']:
        print(f"\n✅ Successfully analyzed {ticker}")
        
        # Show available data
        print(f"\n📊 Available data keys:")
        for key in result.keys():
            if key not in ['portfolio', 'daily_data', 'intraday_data']:
                print(f"  - {key}: {result[key]}")
        
        # Show portfolio stats
        portfolio = result['portfolio']
        stats = portfolio.stats()
        
        print(f"\n📈 Available portfolio statistics ({len(stats)} metrics):")
        print("=" * 60)
        
        # Group metrics by category for better understanding
        performance_metrics = []
        risk_metrics = []
        trading_metrics = []
        other_metrics = []
        
        for key, value in sorted(stats.items()):
            if any(term in key.lower() for term in ['return', 'profit', 'win', 'loss']):
                performance_metrics.append((key, value))
            elif any(term in key.lower() for term in ['drawdown', 'volatility', 'sharpe', 'calmar', 'risk']):
                risk_metrics.append((key, value))
            elif any(term in key.lower() for term in ['trade', 'position', 'duration']):
                trading_metrics.append((key, value))
            else:
                other_metrics.append((key, value))
        
        print("\n🎯 PERFORMANCE METRICS:")
        for key, value in performance_metrics:
            print(f"  {key}: {value}")
        
        print("\n⚠️  RISK METRICS:")
        for key, value in risk_metrics:
            print(f"  {key}: {value}")
        
        print("\n📊 TRADING METRICS:")
        for key, value in trading_metrics:
            print(f"  {key}: {value}")
        
        print("\n📋 OTHER METRICS:")
        for key, value in other_metrics:
            print(f"  {key}: {value}")
            
        # Show regime information
        print(f"\n🎯 REGIME INFORMATION:")
        print(f"  Current Regime: {result['regime']}")
        print(f"  Confidence: {result['conf']:.2f}")
        
        # Show daily data features
        daily_data = result['daily_data']
        print(f"\n📅 DAILY DATA FEATURES (last 5 rows):")
        print(daily_data[['Close', 'regime_ml', 'regime_ml_conf']].tail())
        
    else:
        print(f"❌ Error analyzing {ticker}: {result['error']}")

if __name__ == "__main__":
    test_metrics()