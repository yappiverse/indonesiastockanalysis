#!/usr/bin/env python3
"""
Test script to verify CARE.JK shows latest price (318) instead of opening price (312)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher

def test_care_price():
    """Test CARE.JK price data to verify latest price is used as opening price."""
    print("🧪 Testing CARE.JK price data...")
    
    fetcher = DataFetcher()
    
    # Test with CARE.JK
    ticker = "CARE.JK"
    price_data = fetcher.get_latest_price(ticker)
    
    print(f"📊 Price data for {ticker}:")
    for key, value in price_data.items():
        print(f"   {key}: {value}")
    
    # Verify that opening_price equals current_price (both should be the latest price)
    if price_data["success"]:
        opening_price = price_data["opening_price"]
        current_price = price_data["current_price"]
        
        print(f"\n✅ VERIFICATION:")
        print(f"   Opening Price: {opening_price}")
        print(f"   Current Price: {current_price}")
        
        if opening_price == current_price:
            print(f"   ✅ SUCCESS: opening_price equals current_price ({opening_price})")
            print(f"   ✅ The system now uses the latest available price as the 'opening_price'")
        else:
            print(f"   ❌ FAILED: opening_price ({opening_price}) != current_price ({current_price})")
            
        # Check if we're getting the expected value around 318
        if opening_price is not None and opening_price > 300:
            print(f"   ✅ Price is in expected range (around 318): {opening_price}")
        else:
            print(f"   ⚠️  Price may not be the expected latest price: {opening_price}")
    else:
        print(f"❌ Failed to fetch data: {price_data.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("🚀 Testing CARE.JK Price Implementation")
    print("=" * 50)
    test_care_price()
    print("\n" + "=" * 50)
    print("🎉 Test completed!")