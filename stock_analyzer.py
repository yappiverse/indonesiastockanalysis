import warnings
import pandas as pd
from typing import Dict, Any, List, Optional
from data_layer import DataFetcher, format_ticker
from feature_engineering import FeatureEngineer, align_regime_to_intraday
from regime_detection import RegimeDetector, RegimeAnalyzer
from backtesting import Backtester, PerformanceReporter
from config import DATA_CONFIG, FEATURE_CONFIG, REGIME_CONFIG, BACKTEST_CONFIG


class StockAnalyzer:
    """
    Main orchestrator for stock analysis pipeline.
    Handles the complete workflow from data fetching to backtesting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        warnings.filterwarnings("ignore")
        pd.set_option('future.no_silent_downcasting', True)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = RegimeDetector()
        self.backtester = Backtester()
        self.performance_reporter = PerformanceReporter()
        self.regime_analyzer = RegimeAnalyzer()
        
        self._results_cache = {}
    
    def analyze_ticker(
        self, 
        ticker_raw: str,
        use_cache: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a single ticker.
        
        Args:
            ticker_raw: Raw ticker symbol (without suffix)
            use_cache: Whether to use cached data
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with analysis results
        """
        ticker = format_ticker(ticker_raw)
        
        if verbose:
            self._print_header(f"Processing {ticker}")
        
        try:
            # Step 1: Fetch and process daily data
            if verbose:
                print("Fetching daily data...")
            
            df_daily = self.data_fetcher.get_ohlcv(
                ticker, 
                period=DATA_CONFIG["period_daily"],
                interval=DATA_CONFIG["interval_daily"],
                use_cache=use_cache
            )
            
            df_daily = self.feature_engineer.add_daily_features(df_daily)
            df_daily = self.regime_detector.detect_ml(df_daily)
            
            # Get latest regime information
            last_record = df_daily.iloc[-1]
            current_regime = last_record["regime_ml"]
            confidence = float(last_record["regime_ml_conf"])
            
            # Step 2: Fetch and process intraday data
            if verbose:
                print("Fetching intraday data...")
            
            df_intraday = self.data_fetcher.get_ohlcv(
                ticker,
                period=DATA_CONFIG["period_intraday"],
                interval=DATA_CONFIG["interval_intraday"],
                use_cache=use_cache
            )
            
            df_intraday = self.feature_engineer.add_intraday_features(df_intraday)
            df_intraday = align_regime_to_intraday(df_daily, df_intraday)
            
            # Step 3: Backtest strategy
            if verbose:
                print("Running backtest...")
            
            portfolio = self.backtester.backtest_regime_strategy(df_intraday)
            
            # Step 4: Generate results
            if verbose:
                self._print_regime_summary(ticker, df_daily, current_regime, confidence)
                print(self.performance_reporter.generate_summary_report(
                    portfolio, ticker, current_regime, confidence
                ))
            
            result = {
                "ticker": ticker,
                "regime": current_regime,
                "conf": confidence,
                "stats": portfolio.stats(),
                "portfolio": portfolio,
                "daily_data": df_daily,
                "intraday_data": df_intraday,
                "success": True
            }
            
            # Cache the result
            self._results_cache[ticker] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {ticker}: {e}"
            if verbose:
                print(f"{error_msg}")
            
            return {
                "ticker": ticker,
                "error": str(e),
                "success": False
            }
    
    def analyze_multiple_tickers(
        self, 
        tickers: List[str],
        use_cache: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cached data
            verbose: Whether to print progress information
            
        Returns:
            List of analysis results
        """
        if verbose:
            print(f"\nStarting Multi-Ticker Analysis")
            print(f"Tickers: {', '.join(tickers)}")
        
        results = []
        
        for ticker in tickers:
            result = self.analyze_ticker(ticker, use_cache=use_cache, verbose=verbose)
            results.append(result)
        
        return results
    
    def get_regime_analysis(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get regime distribution analysis for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with regime statistics
        """
        if ticker not in self._results_cache:
            return None
        
        result = self._results_cache[ticker]
        if not result["success"]:
            return None
        
        return self.regime_analyzer.analyze_regime_distribution(result["daily_data"])
    
    def compare_strategies(
        self, 
        ticker: str,
        regime_columns: List[str] = ["regime", "regime_ml"]
    ) -> Optional[pd.DataFrame]:
        """
        Compare different regime strategies for a ticker.
        
        Args:
            ticker: Ticker symbol
            regime_columns: List of regime columns to compare
            
        Returns:
            DataFrame with strategy comparison
        """
        if ticker not in self._results_cache:
            return None
        
        result = self._results_cache[ticker]
        if not result["success"]:
            return None
        
        return self.backtester.compare_strategies(
            result["intraday_data"], 
            regime_columns,
            strategy_names=[f"Strategy_{col}" for col in regime_columns]
        )
    
    def clear_cache(self):
        """Clear all cached data and results."""
        self.data_fetcher.clear_cache()
        self._results_cache.clear()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the analysis system."""
        return {
            "data_cache_info": self.data_fetcher.get_cache_info(),
            "ml_model_info": self.regime_detector.get_model_info(),
            "cached_results": list(self._results_cache.keys())
        }
    
    def _print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
    
    def _print_regime_summary(self, ticker: str, df_daily: pd.DataFrame, regime: str, confidence: float):
        """Print regime summary information."""
        last_date = df_daily.index[-1].date()
        
        print(f"\n Latest Regime for {ticker}")
        print(f"Date:          {last_date}")
        print(f"Regime (ML):   {regime}")
        print(f"Confidence:    {confidence:.2f}")
        
        if regime in ["Bull", "Recovery"]:
            print("BUY SIGNAL")
        else:
            print("No long setup")