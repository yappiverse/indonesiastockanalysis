import numpy as np
import pandas as pd
from typing import List, Dict, Any
from config import FEATURE_CONFIG


class FeatureEngineer:
    """Handles feature engineering for stock data analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or FEATURE_CONFIG
        
    def add_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add daily trading features to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Calculate moving averages
        df = self._add_moving_averages(df)
        
        # Calculate returns
        df = self._add_returns(df)
        
        # Calculate volatility
        df = self._add_volatility(df)
        
        # Calculate volume features
        df = self._add_volume_features(df)
        
        # Calculate ATR
        df = self._add_atr(df)
        
        return df.dropna()
    
    def add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intraday trading features to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added intraday features
        """
        df = df.copy()
        
        # Calculate returns
        df["ret"] = df["Close"].pct_change()
        
        # Calculate moving averages
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma60"] = df["Close"].rolling(60).mean()
        df["intraday_trend"] = df["ma20"] - df["ma60"]
        
        # Calculate volatility
        df["vol_20"] = df["ret"].rolling(20).std()
        
        return df.dropna()
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        ma_periods = self.config["moving_averages"]
        
        for period in ma_periods:
            df[f"ma{period}"] = df["Close"].rolling(period).mean()
        
        # Add trend difference
        if len(ma_periods) >= 2:
            df["trend_diff"] = df[f"ma{ma_periods[1]}"] - df[f"ma{ma_periods[2]}"]
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        return_periods = self.config["return_periods"]
        
        for period in return_periods:
            df[f"ret_{period}"] = df["Close"].pct_change(period)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        vol_periods = self.config["volatility_periods"]
        
        for period in vol_periods:
            df[f"vol_{period}"] = df["ret_1"].rolling(period).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        period = self.config["volume_zscore_period"]
        
        vol_mean = df["Volume"].rolling(period).mean()
        vol_std = df["Volume"].rolling(period).std()
        df["vol_z"] = (df["Volume"] - vol_mean) / vol_std
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range (ATR) feature."""
        period = self.config["atr_period"]
        
        # Calculate True Range
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        
        # Use vectorized operation for better performance
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.rolling(period).mean()
        
        return df
    
    def get_feature_list(self, feature_type: str = "daily") -> List[str]:
        """Get list of feature names for a given feature type."""
        if feature_type == "daily":
            return [
                "trend_diff", "ret_1", "ret_5", "ret_20",
                "vol_20", "vol_50", "vol_z", "atr_14"
            ]
        elif feature_type == "intraday":
            return ["ret", "ma20", "ma60", "intraday_trend", "vol_20"]
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")


def align_regime_to_intraday(
    daily_df: pd.DataFrame, 
    intraday_df: pd.DataFrame, 
    regime_col: str = "regime_ml"
) -> pd.DataFrame:
    """
    Align daily regime data to intraday timeframe.
    
    Args:
        daily_df: Daily dataframe with regime data
        intraday_df: Intraday dataframe
        regime_col: Name of the regime column
        
    Returns:
        Intraday dataframe with regime data
    """
    intraday = intraday_df.copy()
    
    # Reindex and forward fill regime data
    regime_data = daily_df[regime_col].copy()
    intraday_regime = regime_data.reindex(intraday.index, method="ffill")
    
    # Explicitly convert to string type to avoid downcasting warnings
    intraday[regime_col] = intraday_regime.astype(str)
    
    return intraday.dropna(subset=[regime_col])