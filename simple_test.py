#!/usr/bin/env python3
"""
Simple test for opening price functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer import DataFetcher, CSVExporter

def main():
    print("Testing opening price functionality...")
    
    # Test DataFetcher
    fetcher = DataFetcher()
    price_data = fetcher.get_latest_price("BBCA.JK")
    print(f"Price data: {price_data}")
    
    # Test CSVExporter
    exporter = CSVExporter()
    print(f"CSV columns: {exporter.csv_columns}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()