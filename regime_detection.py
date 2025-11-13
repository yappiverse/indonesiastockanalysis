#!/usr/bin/env python3
"""
Regime detection module for stock analysis.
Provides rule-based and ML-based regime classification.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import REGIME_CONFIG


class RegimeDetector:
    """Handles regime detection using rule-based and ML methods."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or REGIME_CONFIG
        self._ml_pipeline = None
        self._is_fitted = False
        # Initialize with unique None object to prevent sharing in threading
        self._ml_pipeline = None
        self._is_fitted = False
        
    def detect_rule_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes using rule-based approach.
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            DataFrame with regime labels
        """
        df = df.copy()
        rb_config = self.config["rule_based"]
        
        # Calculate regime conditions
        high_vol = (
            df["vol_20"] > 
            df["vol_20"].rolling(rb_config["high_vol_window"])
                        .quantile(rb_config["high_vol_quantile"])
                        .shift(1)
        )
        
        strong_bull = (df["trend_diff"] > 0) & (df["ret_20"] > 0)
        strong_bear = (df["trend_diff"] < 0) & (df["ret_20"] < 0)
        crash = df["ret_5"] < rb_config["crash_threshold"]
        recovery = (df["ret_5"] > rb_config["recovery_threshold"]) & (df["trend_diff"] > 0)
        
        # Apply regime logic with optimized nested conditions
        regimes = self._apply_regime_logic(
            crash, high_vol, strong_bull, strong_bear, recovery
        )
        
        df["regime"] = regimes
        return df
    
    def _apply_regime_logic(self, crash, high_vol, strong_bull, strong_bear, recovery):
        """Apply regime logic with optimized conditional structure."""
        regimes = np.full(len(crash), "Sideways", dtype=object)
        
        # Apply conditions in priority order
        regimes[crash] = "Crash"
        regimes[high_vol & ~crash] = "HighVol"
        regimes[strong_bull & ~crash & ~high_vol] = "Bull"
        regimes[strong_bear & ~crash & ~high_vol & ~strong_bull] = "Bear"
        regimes[recovery & ~crash & ~high_vol & ~strong_bull & ~strong_bear] = "Recovery"
        
        return regimes
    
    def detect_ml(self, df: pd.DataFrame, retrain: bool = False) -> pd.DataFrame:
        """
        Detect market regimes using ML ensemble approach.
        
        Args:
            df: DataFrame with feature data
            retrain: Whether to retrain the model
            
        Returns:
            DataFrame with ML regime labels and confidence
        """
        df = df.copy()
        
        # First apply rule-based detection as baseline
        df = self.detect_rule_based(df)
        
        # Get features for ML
        features = self.config["ml"]["features"]
        X = df[features]
        y = df["regime"]
        
        # Train or use existing model
        if retrain or not self._is_fitted or self._ml_pipeline is None:
            self._train_ml_pipeline(X, y)
        
        # Make predictions
        df["regime_ml"] = self._ml_pipeline.predict(X)
        
        # Add confidence scores
        try:
            probabilities = self._ml_pipeline.predict_proba(X)
            df["regime_ml_conf"] = probabilities.max(axis=1)
        except Exception:
            df["regime_ml_conf"] = np.nan
        
        return df
    
    def _train_ml_pipeline(self, X: pd.DataFrame, y: pd.Series):
        """Train the ML ensemble pipeline."""
        ensemble_config = self.config["ml"]["ensemble"]
        
        # Initialize classifiers
        rf = RandomForestClassifier(**ensemble_config["random_forest"])
        gb = GradientBoostingClassifier(**ensemble_config["gradient_boosting"])
        svc = SVC(**ensemble_config["svc"])
        log = LogisticRegression(**ensemble_config["logistic_regression"])
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("svc", svc), ("log", log)],
            voting="soft"
        )
        
        # Create pipeline
        self._ml_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ensemble", ensemble),
        ])
        
        # Train the model
        self._ml_pipeline.fit(X, y)
        self._is_fitted = True
    
    def predict_single(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict regime for a single data point.
        
        Args:
            feature_dict: Dictionary of feature values
            
        Returns:
            Dictionary with regime prediction and confidence
        """
        if not self._is_fitted or self._ml_pipeline is None:
            raise ValueError("ML model not trained. Call detect_ml() first.")
        
        # Convert to DataFrame for prediction
        features = self.config["ml"]["features"]
        X_single = pd.DataFrame([feature_dict])[features]
        
        regime = self._ml_pipeline.predict(X_single)[0]
        
        try:
            proba = self._ml_pipeline.predict_proba(X_single)
            confidence = float(proba.max(axis=1)[0])
        except Exception:
            confidence = np.nan
        
        return {
            "regime": regime,
            "confidence": confidence
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained ML model."""
        if not self._is_fitted:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "features": self.config["ml"]["features"],
            "pipeline_steps": [step[0] for step in self._ml_pipeline.steps]
        }


class RegimeAnalyzer:
    """Provides analysis and insights for detected regimes."""
    
    def __init__(self):
        pass
    
    def analyze_regime_distribution(self, df: pd.DataFrame, regime_col: str = "regime_ml") -> pd.DataFrame:
        """
        Analyze distribution of regimes.
        
        Args:
            df: DataFrame with regime data
            regime_col: Name of the regime column
            
        Returns:
            DataFrame with regime statistics
        """
        regime_counts = df[regime_col].value_counts()
        regime_pct = df[regime_col].value_counts(normalize=True) * 100
        
        stats = pd.DataFrame({
            'count': regime_counts,
            'percentage': regime_pct
        }).sort_values('count', ascending=False)
        
        return stats
    
    def get_regime_transitions(self, df: pd.DataFrame, regime_col: str = "regime_ml") -> pd.DataFrame:
        """
        Analyze transitions between regimes.
        
        Args:
            df: DataFrame with regime data
            regime_col: Name of the regime column
            
        Returns:
            DataFrame with transition counts
        """
        regimes = df[regime_col]
        transitions = pd.crosstab(regimes.shift(1), regimes, dropna=False)
        return transitions