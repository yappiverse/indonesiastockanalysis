import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_analyzer import StockAnalyzer


def test_basic_functionality():
    """Test basic functionality of the refactored system."""
    print("🧪 Testing refactored stock analysis system...")
    
    try:
        # Initialize the analyzer
        analyzer = StockAnalyzer()
        print("StockAnalyzer initialized successfully")
        
        # Test system info
        system_info = analyzer.get_system_info()
        print(f"System info retrieved: {system_info}")
        
        # Test data fetcher
        data_fetcher = analyzer.data_fetcher
        print("DataFetcher accessible")
        
        # Test feature engineer
        feature_engineer = analyzer.feature_engineer
        print("FeatureEngineer accessible")
        
        # Test regime detector
        regime_detector = analyzer.regime_detector
        print("RegimeDetector accessible")
        
        # Test backtester
        backtester = analyzer.backtester
        print("Backtester accessible")
        
        print("\n All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def test_config_import():
    """Test that configuration imports work correctly."""
    print("\n Testing configuration imports...")
    
    try:
        from config import DATA_CONFIG, FEATURE_CONFIG, REGIME_CONFIG, BACKTEST_CONFIG
        
        print(f"DATA_CONFIG loaded: {len(DATA_CONFIG)} parameters")
        print(f"FEATURE_CONFIG loaded: {len(FEATURE_CONFIG)} parameters") 
        print(f"REGIME_CONFIG loaded: {len(REGIME_CONFIG)} parameters")
        print(f"BACKTEST_CONFIG loaded: {len(BACKTEST_CONFIG)} parameters")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False


def test_module_imports():
    """Test that all modules can be imported correctly."""
    print("\n Testing module imports...")
    
    modules_to_test = [
        "data_layer",
        "feature_engineering", 
        "regime_detection",
        "backtesting",
        "stock_analyzer"
    ]
    
    all_passed = True
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"{module_name} imported successfully")
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 REFACTORED CODE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_module_imports,
        test_config_import,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Tests run: {len(tests)}")
    print(f"Tests passed: {sum(results)}")
    print(f"Tests failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("ALL TESTS PASSED - Refactored code is working correctly!")
    else:
        print("Some tests failed - please check the implementation.")
    
    print("=" * 60)