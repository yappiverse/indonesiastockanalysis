#!/usr/bin/env python3
"""
Test script for opening price functionality in CSV export.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher, CSVExporter

def test_data_fetcher():
    """Test the DataFetcher get_latest_price method."""
    print("🧪 Testing DataFetcher.get_latest_price()...")
    
    fetcher = DataFetcher()
    
    # Test with a known Indonesian stock
    ticker = "BBCA.JK"
    price_data = fetcher.get_latest_price(ticker)
    
    print(f"📊 Price data for {ticker}:")
    for key, value in price_data.items():
        print(f"   {key}: {value}")
    
    return price_data

def test_csv_exporter():
    """Test the CSVExporter with mock data including opening price."""
    print("\n🧪 Testing CSVExporter with opening price...")
    
    exporter = CSVExporter()
    
    # Create mock results with price data
    mock_results = [
        {
            "ticker": "BBCA.JK",
            "recommendation": "BUY",
            "confidence": "HIGH",
            "risk": "MODERATE",
            "rationale": "Strong performance in current regime",
            "key_metrics": {"total_return": 15.5, "sharpe_ratio": 1.8},
            "regime": "Bull",
            "regime_confidence": 0.85,
            "next_day_action": "Consider buying on next trading day",
            "summary_score": 4,
            "success": True,
            "price_data": {
                "ticker": "BBCA.JK",
                "opening_price": 9500.0,
                "current_price": 9550.0,
                "high_price": 9600.0,
                "low_price": 9450.0,
                "volume": 15000000,
                "timestamp": "2025-11-15 09:00:00",
                "success": True
            }
        },
        {
            "ticker": "ASII.JK",
            "recommendation": "HOLD",
            "confidence": "MODERATE",
            "risk": "LOW",
            "rationale": "Sideways movement expected",
            "key_metrics": {"total_return": 2.3, "sharpe_ratio": 0.8},
            "regime": "Neutral",
            "regime_confidence": 0.65,
            "next_day_action": "Hold current position, monitor for changes",
            "summary_score": 3,
            "success": True,
            "price_data": {
                "ticker": "ASII.JK",
                "opening_price": 7200.0,
                "current_price": 7180.0,
                "high_price": 7250.0,
                "low_price": 7150.0,
                "volume": 8000000,
                "timestamp": "2025-11-15 09:00:00",
                "success": True
            }
        },
        {
            "ticker": "FAILED.JK",
            "success": False,
            "rationale": "Analysis failed: No data available"
        }
    ]
    
    # Export to CSV
    test_filename = "test_output.csv"
    try:
        csv_path = exporter.export_to_csv(mock_results, test_filename)
        print(f"✅ Successfully exported test CSV to: {csv_path}")
        
        # Read and display the CSV content
        print("\n📄 CSV Content:")
        with open(test_filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        
        # Clean up
        os.remove(test_filename)
        print(f"🧹 Cleaned up test file: {test_filename}")
        
    except Exception as e:
        print(f"❌ Failed to export CSV: {e}")

def test_after_market_scenario():
    """Test the get_latest_price method handles after-market scenarios."""
    print("\n🧪 Testing after-market scenario handling...")
    
    fetcher = DataFetcher()
    
    # Test with multiple tickers to verify after-market behavior
    tickers = ["BBCA.JK", "ASII.JK", "CARE.JK"]
    
    for ticker in tickers:
        print(f"\n📈 Testing {ticker}:")
        price_data = fetcher.get_latest_price(ticker)
        
        if price_data["success"]:
            print(f"   ✅ Success - Opening Price: {price_data['opening_price']}")
            print(f"   📅 Timestamp: {price_data['timestamp']}")
            print(f"   💰 Current Price: {price_data['current_price']}")
        else:
            print(f"   ❌ Failed - Error: {price_data.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("🚀 Starting Opening Price Functionality Tests")
    print("=" * 50)
    
    # Test data fetcher
    price_data = test_data_fetcher()
    
    # Test CSV exporter
    test_csv_exporter()
    
    # Test after-market scenarios
    test_after_market_scenario()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")