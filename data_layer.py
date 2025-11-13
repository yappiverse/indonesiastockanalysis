#!/usr/bin/env python3
"""
Data layer module for fetching and processing stock data.
Provides robust data fetching with caching and error handling.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
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


def format_ticker(ticker_raw: str) -> str:
    """Format ticker symbol with proper suffix."""
    return ticker_raw.upper() + DATA_CONFIG["ticker_suffix"]