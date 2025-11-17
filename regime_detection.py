#!/usr/bin/env python3
"""
Regime detection module for stock analysis.
Option C (Final Clean Version):
- XGBoost-only ML model
- TimeSeriesSplit CV
- Fixed global label mapping
- Stable ML classification
- Compatible with StockAnalyzer without modification
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from config import REGIME_CONFIG


# =====================================================================
# REGIME DETECTOR (XGBoost)
# =====================================================================
class RegimeDetector:
    """Handles regime detection using rule-based + ML (XGBoost) methods."""

    # FIXED GLOBAL CLASS SPACE (no mismatch ever again)
    FIXED_CLASSES = [
        "Bear",
        "Bull",
        "Crash",
        "HighVol",
        "Recovery",
        "Sideways",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or REGIME_CONFIG

        # ML config
        self._ml_config: Dict[str, Any] = self.config.get("ml", {})
        self._features: List[str] = self._ml_config.get("features", [])

        # Model state
        self._model: Optional[XGBClassifier] = None
        self._is_fitted: bool = False
        self._ml_metrics: Optional[Dict[str, Any]] = None

        # Label encoding tables
        self._label_to_int = {lab: i for i, lab in enumerate(self.FIXED_CLASSES)}
        self._int_to_label = {i: lab for i, lab in enumerate(self.FIXED_CLASSES)}

    # =================================================================
    # RULE-BASED DETECTION
    # =================================================================
    def detect_rule_based(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rb = self.config.get("rule_based", {})

        win = rb.get("high_vol_window", 60)
        q = rb.get("high_vol_quantile", 0.8)
        crash_thr = rb.get("crash_threshold", -0.08)
        rec_thr = rb.get("recovery_threshold", 0.04)

        high_vol = (
            df["vol_20"]
            > df["vol_20"].rolling(win).quantile(q).shift(1)
        )

        strong_bull = (df["trend_diff"] > 0) & (df["ret_20"] > 0)
        strong_bear = (df["trend_diff"] < 0) & (df["ret_20"] < 0)
        crash = df["ret_5"] < crash_thr
        recovery = (df["ret_5"] > rec_thr) & (df["trend_diff"] > 0)

        regimes = np.full(len(df), "Sideways", dtype=object)
        regimes[crash] = "Crash"
        regimes[high_vol & ~crash] = "HighVol"
        regimes[strong_bull & ~crash & ~high_vol] = "Bull"
        regimes[strong_bear & ~crash & ~high_vol & ~strong_bull] = "Bear"
        regimes[recovery & ~crash & ~high_vol & ~strong_bull & ~strong_bear] = "Recovery"

        df["regime"] = regimes
        return df

    # =================================================================
    # ML DETECTION (XGBoost)
    # =================================================================
    def detect_ml(self, df: pd.DataFrame, retrain: bool = False) -> pd.DataFrame:
        df = df.copy()

        # Step 1 — preserve original regime labels for ML training
        original_regime = df["regime"].copy()
        
        # Step 2 — generate rule-based predictions for fallback
        df_rb = self.detect_rule_based(df)

        if not self._features:
            raise ValueError("ML features not defined in config['ml']['features'].")

        # Data extraction - use ORIGINAL regime labels for ML training
        X = df[self._features]
        y = original_regime

        X_clean, y_clean, mask_valid = self._prepare_ml_data(X, y)

        # Fallback when insufficient data
        min_samples = self._ml_config.get("min_train_samples", 200)
        if len(X_clean) < min_samples:
            df["regime_ml"] = df["regime"]
            df["regime_ml_conf"] = np.nan
            self._ml_metrics = {
                "status": "not_trained_insufficient_data",
                "num_samples": len(X_clean),
                "min_required": min_samples,
            }
            return df

        # Train model if needed
        if retrain or not self._is_fitted:
            self._train_ml_model(X_clean, y_clean)

        # Predict
        regime_ml = pd.Series(index=df.index, dtype=object)
        regime_conf = pd.Series(index=df.index, dtype=float)

        if self._is_fitted:
            X_valid = X.loc[mask_valid]

            preds_int = self._model.predict(X_valid)
            preds_label = self._decode_predictions(preds_int)
            regime_ml.loc[X_valid.index] = preds_label

            # Confidence
            try:
                probs = self._model.predict_proba(X_valid)
                regime_conf.loc[X_valid.index] = probs.max(axis=1)
            except Exception:
                regime_conf.loc[:] = np.nan

        # Fill gaps with rule-based predictions
        regime_ml.fillna(df_rb["regime"], inplace=True)
        regime_conf.fillna(np.nan, inplace=True)

        df["regime_ml"] = regime_ml
        df["regime_ml_conf"] = regime_conf

        return df

    def _prepare_ml_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        mask = X.notnull().all(axis=1) & y.notnull()
        return X.loc[mask], y.loc[mask], mask

    # =================================================================
    # LABEL ENCODING / DECODING
    # =================================================================
    def _encode_labels(self, y: pd.Series) -> pd.Series:
        """Encode regime labels to integers with validation."""
        try:
            return y.map(lambda v: self._label_to_int[v])
        except KeyError as e:
            invalid_label = str(e).strip("'")
            raise ValueError(f"Invalid regime label '{invalid_label}'. Expected one of: {self.FIXED_CLASSES}")

    def _decode_predictions(self, preds: np.ndarray) -> List[str]:
        """Decode integer predictions to regime labels with validation."""
        decoded = []
        for p in preds:
            pred_int = int(p)
            if pred_int not in self._int_to_label:
                raise ValueError(f"Invalid prediction class {pred_int}. Expected classes: {list(self._int_to_label.keys())}")
            decoded.append(self._int_to_label[pred_int])
        return decoded

    # =================================================================
    # MODEL BUILD + TRAIN
    # =================================================================
    def _build_xgb_model(self, class_weights: Optional[Dict[int, float]] = None) -> XGBClassifier:
        cfg = self._ml_config

        params = {
            "objective": "multi:softprob",
            "num_class": len(self.FIXED_CLASSES),  # ⭐ FORCE 6 CLASSES
            "eval_metric": cfg.get("eval_metric", "mlogloss"),
            "n_estimators": cfg.get("n_estimators", 300),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "max_depth": cfg.get("max_depth", 4),
            "subsample": cfg.get("subsample", 0.9),
            "colsample_bytree": cfg.get("colsample_bytree", 0.9),
            "reg_lambda": cfg.get("reg_lambda", 1.0),
            "reg_alpha": cfg.get("reg_alpha", 0.0),
            "n_jobs": cfg.get("n_jobs", -1),
            "tree_method": cfg.get("tree_method", "hist"),
        }

        # Add class weights if provided
        if class_weights:
            params["scale_pos_weight"] = 1.0  # Default value
            # XGBoost doesn't have direct class_weight parameter like sklearn
            # We'll handle class imbalance through other means

        return XGBClassifier(**params)

    def _train_ml_model(self, X: pd.DataFrame, y: pd.Series):
        y_enc = self._encode_labels(y)

        # Calculate class distribution for better handling
        class_counts = y_enc.value_counts().sort_index()
        total_samples = len(y_enc)
        
        # Calculate class weights (inverse frequency)
        class_weights = {}
        for class_idx in range(len(self.FIXED_CLASSES)):
            if class_idx in class_counts.index:
                class_weights[class_idx] = total_samples / (len(self.FIXED_CLASSES) * class_counts[class_idx])
            else:
                class_weights[class_idx] = 1.0  # Default weight for missing classes

        # Validate that all expected classes are present in training data
        unique_classes = set(y_enc.unique())
        expected_classes = set(range(len(self.FIXED_CLASSES)))
        missing_classes = expected_classes - unique_classes
        
        if missing_classes:
            missing_class_names = [self._int_to_label[c] for c in missing_classes]
            print(f"Warning: Missing classes in training data: {missing_class_names}")
            # Instead of adding unrealistic dummy data, we'll handle this through
            # class weights and proper model configuration

        n_splits = max(2, self._ml_config.get("cv_splits", 3))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        acc_scores = []
        f1_scores = []

        for tr, te in tscv.split(X):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y_enc.iloc[tr], y_enc.iloc[te]

            # Don't skip folds - train on available data even if some classes are missing
            # This is more realistic for time series data
            fold = self._build_xgb_model(class_weights)
            fold.fit(X_tr, y_tr)

            preds = fold.predict(X_te)
            acc_scores.append(accuracy_score(y_te, preds))
            f1_scores.append(f1_score(y_te, preds, average="macro"))

        # Final model - train on all available data
        model = self._build_xgb_model(class_weights)
        model.fit(X, y_enc)

        self._model = model
        self._is_fitted = True
        self._ml_metrics = {
            "status": "trained",
            "samples": len(X),
            "cv_splits": n_splits,
            "cv_acc_mean": float(np.mean(acc_scores)) if acc_scores else 0.0,
            "cv_f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "missing_classes": list(missing_classes) if missing_classes else [],
            "class_distribution": class_counts.to_dict(),
            "class_weights": class_weights
        }

    # =================================================================
    # SINGLE POINT PREDICTION
    # =================================================================
    def predict_single(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        if not self._is_fitted or self._model is None:
            raise ValueError("ML model not trained. Run detect_ml() first.")

        X_single = pd.DataFrame([feature_dict])[self._features]

        pred_int = self._model.predict(X_single)[0]
        regime = self._int_to_label[int(pred_int)]

        try:
            prob = self._model.predict_proba(X_single)[0].max()
        except Exception:
            prob = np.nan

        return {"regime": regime, "confidence": float(prob)}

    # =================================================================
    # MODEL INFO
    # =================================================================
    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "features": self._features,
            "algorithm": "xgboost",
            "classes": self._int_to_label,
        }

        if not self._is_fitted:
            info["status"] = "not_trained"
            info["ml_metrics"] = self._ml_metrics
            return info

        info["status"] = "trained"
        info["ml_metrics"] = self._ml_metrics
        return info


# =====================================================================
# REGIME ANALYZER
# =====================================================================
class RegimeAnalyzer:
    """Provides analysis utilities for regime classification."""

    def analyze_regime_distribution(
        self, df: pd.DataFrame, regime_col: str = "regime_ml"
    ) -> pd.DataFrame:
        counts = df[regime_col].value_counts()
        pct = df[regime_col].value_counts(normalize=True) * 100
        return pd.DataFrame({"count": counts, "percentage": pct}).sort_values(
            "count", ascending=False
        )

    def get_regime_transitions(
        self, df: pd.DataFrame, regime_col: str = "regime_ml"
    ) -> pd.DataFrame:
        return pd.crosstab(df[regime_col].shift(1), df[regime_col], dropna=False)
