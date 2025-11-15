#!/usr/bin/env python3
"""
Configuration module for stock analysis system.
Centralizes all configuration parameters for better maintainability.
"""

# Data configuration
DATA_CONFIG = {
    "period_daily": "5y",
    "period_intraday": "2y",
    "interval_daily": "1d",
    "interval_intraday": "1h",
    "timezone": "Asia/Jakarta",
    "ticker_suffix": ".JK"
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "moving_averages": [20, 50, 200],
    "return_periods": [1, 5, 20],
    "volatility_periods": [20, 50],
    "atr_period": 14,
    "volume_zscore_period": 50
}

# Regime detection configuration
REGIME_CONFIG = {
    "rule_based": {
        "high_vol_quantile": 0.75,
        "high_vol_window": 252,
        "crash_threshold": -0.08,
        "recovery_threshold": 0.05
    },
    "ml": {
        "features": [
            "trend_diff", "ret_1", "ret_5", "ret_20",
            "vol_20", "vol_50", "vol_z", "atr_14"
        ],
        "ensemble": {
            "random_forest": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": 1
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42
            },
            "svc": {
                "kernel": "rbf",
                "C": 2.0,
                "probability": True,
                "gamma": "scale",
                "random_state": 42
            },
            "logistic_regression": {
                "max_iter": 2000,
                "n_jobs": 1
            },
            "xgboost": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "objective": "multi:softprob"
            }
        }
    }
}

# Backtesting configuration
BACKTEST_CONFIG = {
    "default": {
        "init_cash": 1_000_000,
        "fees": 0.0005,
        "slippage": 0.0005,
        "sl_stop": 0.02,
        "tp_stop": 0.05,
        "frequency": "1h",
        "trade_regimes": ["Bull", "Recovery"]
    }
}

# Parallel processing configuration
PARALLEL_CONFIG = {
    "default_jobs": -1
}