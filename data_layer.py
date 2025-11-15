#!/usr/bin/env python3
"""
Data layer module for fetching and processing stock data.
Provides robust data fetching with caching and error handling.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import csv
import os
from typing import Optional, Dict, Any, List
from config import DATA_CONFIG


class DataFetcher:
    """Handles data fetching and preprocessing for stock analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DATA_CONFIG
        self._cache = {}
        
    def get_ohlcv(
        self,
        ticker: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        tz_convert: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker with robust preprocessing.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (e.g., "2y", "5y")
            interval: Data interval (e.g., "1d", "1h")
            tz_convert: Timezone for conversion
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate cache key
        cache_key = f"{ticker}_{period}_{interval}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Use defaults from config if not provided
        period = period or self.config["period_daily"]
        interval = interval or self.config["interval_daily"]
        tz_convert = tz_convert or self.config["timezone"]
        
        try:
            # Use a fresh yf.download call for each request to avoid thread contamination
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False
            )

            if df is None or df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Process and standardize the dataframe
            df = self._standardize_dataframe(df, ticker)
            df = self._handle_timezone(df, tz_convert)
            
            # Cache the result
            self._cache[cache_key] = df.copy()
            
            return df

        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    
    def _standardize_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize dataframe columns and structure."""
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(c) for c in col]).strip() for col in df.columns]

        # Identify OHLCV columns robustly
        col_map = self._identify_ohlcv_columns(df, ticker)
        
        # Select and rename columns
        df = df[list(col_map.values())]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        return df.dropna()
    
    def _identify_ohlcv_columns(self, df: pd.DataFrame, ticker: str) -> Dict[str, str]:
        """Identify OHLCV columns in the dataframe."""
        def find_column(prefix: str) -> Optional[str]:
            for col in df.columns:
                if col.lower().startswith(prefix.lower()):
                    return col
            return None
        
        col_map = {
            "Open": find_column("open"),
            "High": find_column("high"),
            "Low": find_column("low"),
            "Close": find_column("close"),
            "Volume": find_column("volume"),
        }
        
        if None in col_map.values():
            raise KeyError(f"Missing OHLCV columns for {ticker}: {df.columns}")
        
        return col_map
    
    def _handle_timezone(self, df: pd.DataFrame, tz_convert: str) -> pd.DataFrame:
        """Handle timezone conversion for the dataframe."""
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        df.index = df.index.tz_convert(tz_convert)
        df.index = df.index.tz_localize(None)
        
        return df
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the cache."""
        return {
            "cache_size": len(self._cache),
            "cached_tickers": list(self._cache.keys())
        }

    def get_latest_price(self, ticker: str) -> Dict[str, Any]:
        """
        Get the latest price data for a ticker.
        This method uses the current/latest price as the "opening_price" value
        since the script typically runs after market hours.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with latest price information where opening_price = current_price
        """
        try:
            # Fetch the most recent data (1d interval for latest prices)
            df = self.get_ohlcv(
                ticker,
                period="1d",
                interval="1d",
                use_cache=False  # Don't cache to get fresh data
            )
            
            if df is None or df.empty:
                raise ValueError(f"No recent data available for {ticker}")
            
            # Get the latest record
            latest_record = df.iloc[-1]
            
            # Use the current/latest price as the "opening_price" value
            # since the script runs after market hours
            current_price = float(latest_record["Close"])
            
            return {
                "ticker": ticker,
                "opening_price": current_price,  # Use current price as opening price
                "current_price": current_price,
                "high_price": float(latest_record["High"]),
                "low_price": float(latest_record["Low"]),
                "volume": int(latest_record["Volume"]),
                "timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "ticker": ticker,
                "opening_price": None,
                "current_price": None,
                "high_price": None,
                "low_price": None,
                "volume": None,
                "timestamp": None,
                "success": False,
                "error": str(e)
            }


def format_ticker(ticker_raw: str) -> str:
    """Format ticker symbol with proper suffix."""
    return ticker_raw.upper() + DATA_CONFIG["ticker_suffix"]


class CSVExporter:
    """Handles CSV export functionality for stock analysis results."""
    
    def __init__(self):
        self.csv_columns = [
            "no", "ticker", "current_price", "regime", "confidence", "action", "risk", "score"
        ]
    
    def export_to_csv(self, results: List[Dict[str, Any]], filename: str = "stock_analysis.csv") -> str:
        """
        Export stock analysis results to CSV file.
        
        Args:
            results: List of analysis result dictionaries
            filename: Output CSV filename
            
        Returns:
            Path to the created CSV file
        """
        if not results:
            raise ValueError("No results to export")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
            writer.writeheader()
            
            for i, result in enumerate(results, 1):
                row = self._format_result_to_row(result, i)
                writer.writerow(row)
        
        return os.path.abspath(filename)
    
    def _format_result_to_row(self, result: Dict[str, Any], row_number: int) -> Dict[str, str]:
        """Format a single result dictionary to CSV row."""
        # Handle failed analyses
        if not result.get("success", False):
            return {
                "no": str(row_number),
                "ticker": result.get("ticker", "Unknown"),
                "current_price": "N/A",
                "regime": "FAILED",
                "confidence": "0.00",
                "action": "UNAVAILABLE",
                "risk": "UNKNOWN",
                "score": "1"
            }
        
        # Extract buy/sell/hold from recommendation
        recommendation = result.get("recommendation", "HOLD")
        buy_sell_hold = self._extract_buy_sell_hold(recommendation)
        
        # Format confidence as percentage
        confidence = result.get("regime_confidence", 0.0)
        confidence_str = f"{confidence:.2f}"
        
        # Get score (1-5 stars)
        score = result.get("summary_score", 3)
        
        # Get current price from price_data if available
        current_price = "N/A"
        price_data = result.get("price_data")
        if price_data and price_data.get("success"):
            current_price = f"{price_data.get('opening_price', 'N/A')}"  # opening_price now contains current price
        
        return {
            "no": str(row_number),
            "ticker": result.get("ticker", "Unknown"),
            "current_price": current_price,
            "regime": result.get("regime", "UNKNOWN"),
            "confidence": confidence_str,
            "action": buy_sell_hold,
            "risk": result.get("risk", "MODERATE"),
            "score": str(score)
        }
    
    def _extract_buy_sell_hold(self, recommendation: str) -> str:
        """Extract action from recommendation string."""
        recommendation_lower = recommendation.lower()
        
        if "buy" in recommendation_lower:
            return "BUY"
        elif "sell" in recommendation_lower:
            return "SELL"
        else:
            return "HOLD"