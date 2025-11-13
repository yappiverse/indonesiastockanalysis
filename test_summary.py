#!/usr/bin/env python3
"""
Test script for SummaryAnalyzer functionality.
"""

import warnings
warnings.filterwarnings('ignore')
from stock_analyzer import StockAnalyzer
from summary_analyzer import SummaryAnalyzer

def test_summary_analyzer():
    """Test the SummaryAnalyzer with sample data."""
    print("🧪 Testing SummaryAnalyzer...")
    
    # Initialize analyzers
    stock_analyzer = StockAnalyzer()
    summary_analyzer = SummaryAnalyzer()
    
    # Test with BBCA
    ticker = 'BBCA'
    print(f"\n📊 Analyzing {ticker}...")
    
    # Get analysis result
    result = stock_analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
    
    # Generate summary
    summary = summary_analyzer.generate_summary(result)
    
    # Print formatted report
    report = summary_analyzer.format_summary_report(summary)
    print(report)
    
    # Show raw summary data for debugging
    print("\n🔍 RAW SUMMARY DATA:")
    for key, value in summary.items():
        if key != 'key_metrics':
            print(f"  {key}: {value}")
    
    print("\n📈 KEY METRICS DETAIL:")
    for metric, value in summary['key_metrics'].items():
        print(f"  {metric}: {value}")

def test_multiple_tickers():
    """Test summary generation for multiple tickers."""
    print("\n" + "="*60)
    print("🧪 TESTING MULTIPLE TICKERS")
    print("="*60)
    
    stock_analyzer = StockAnalyzer()
    summary_analyzer = SummaryAnalyzer()
    
    tickers = ['BBCA', 'ASII', 'TLKM']  # Sample tickers
    
    for ticker in tickers:
        print(f"\n📊 Analyzing {ticker}...")
        
        try:
            result = stock_analyzer.analyze_ticker(ticker, use_cache=False, verbose=False)
            summary = summary_analyzer.generate_summary(result)
            report = summary_analyzer.format_summary_report(summary)
            print(report)
            
            # Add separator
            print("\n" + "-"*40)
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            print("-"*40)

if __name__ == "__main__":
    test_summary_analyzer()
    test_multiple_tickers()