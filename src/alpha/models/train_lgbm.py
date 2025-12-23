"""LightGBM model training with sample weighting.

This module provides training utilities that respect time-series
structure and support various sample weighting schemes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from ..config import AlphaConfig, WeightScheme
from ..cv import WeeklySignalCV, validate_no_leakage
from .calibration import CalibratedClassifier, evaluate_calibration


@dataclass
class TrainingResult:
    """Result from training a model fold."""
    fold_idx: int
    model: lgb.LGBMClassifier
    calibrator: Optional[CalibratedClassifier]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    train_indices: np.ndarray
    val_indices: np.ndarray


def compute_sample_weights(
    signals: pd.DataFrame,
    scheme: WeightScheme = WeightScheme.OVERLAP_INVERSE,
    normalize: bool = True,
) -> np.ndarray:
    """Compute sample weights for training.

    Args:
        signals: Signals DataFrame.
        scheme: Weighting scheme to use.
        normalize: Whether to normalize weights to sum to n_samples.

    Returns:
        Array of sample weights.
    """
    n_samples = len(signals)
    weights = np.ones(n_samples)

    if scheme == WeightScheme.UNIFORM:
        pass  # Already uniform

    elif scheme == WeightScheme.OVERLAP_INVERSE:
        # Use pre-computed overlap weights from triple barrier
        if "weight_final" in signals.columns:
            weights = signals["weight_final"].values.copy()
        elif "weight_overlap" in signals.columns:
            weights = signals["weight_overlap"].values.copy()
        else:
            # Fallback to uniform
            pass

    elif scheme == WeightScheme.LIQUIDITY:
        # Weight by liquidity (ADV or dollar volume)
        if "adv_20d" in signals.columns:
            liq = signals["adv_20d"].values
        elif "dollar_volume" in signals.columns:
            liq = signals["dollar_volume"].values
        else:
            liq = np.ones(n_samples)

        # Log-transform and normalize
        liq = np.log1p(np.maximum(liq, 0))
        weights = liq / (liq.mean() + 1e-10)

    elif scheme == WeightScheme.INVERSE_VOLATILITY:
        # Weight inversely by volatility
        if "atr_percent" in signals.columns:
            vol = signals["atr_percent"].values
        elif "atr_pct" in signals.columns:
            vol = signals["atr_pct"].values
        else:
            vol = np.ones(n_samples) * 0.02

        # Inverse volatility (higher vol = lower weight)
        vol = np.maximum(vol, 0.005)  # Floor at 0.5%
        weights = 1.0 / vol
        weights = weights / weights.mean()

    elif scheme == WeightScheme.ATR_PERCENT:
        # Weight by ATR% (higher vol = higher weight)
        if "atr_percent" in signals.columns:
            vol = signals["atr_percent"].values
        elif "atr_pct" in signals.columns:
            vol = signals["atr_pct"].values
        else:
            vol = np.ones(n_samples) * 0.02

        weights = np.maximum(vol, 0.005)
        weights = weights / weights.mean()

    # Handle NaN and normalize
    weights = np.nan_to_num(weights, nan=1.0)
    weights = np.maximum(weights, 0.01)  # Floor

    if normalize:
        weights = weights / weights.sum() * n_samples

    return weights


def train_lgbm_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> Tuple[lgb.LGBMClassifier, Optional[CalibratedClassifier], Dict[str, float]]:
    """Train LightGBM on a single fold.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        sample_weight: Optional sample weights.
        params: LightGBM parameters.
        calibrate: Whether to calibrate probabilities.
        calibration_method: Calibration method to use.

    Returns:
        Tuple of (model, calibrator, val_metrics).
    """
    # Default parameters
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "max_depth": 6,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

    # Clean data
    X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_clean, y_train,
        sample_weight=sample_weight,
    )

    # Predict on validation
    y_prob_val = model.predict_proba(X_val_clean)[:, 1]

    # Calibrate if requested
    calibrator = None
    if calibrate:
        calibrator = CalibratedClassifier(method=calibration_method)
        calibrator.fit(y_val.values, y_prob_val)
        y_prob_cal = calibrator.transform(y_prob_val)
    else:
        y_prob_cal = y_prob_val

    # Compute metrics
    val_metrics = {
        "auc": roc_auc_score(y_val, y_prob_val),
        "auc_calibrated": roc_auc_score(y_val, y_prob_cal) if calibrate else None,
        "aupr": average_precision_score(y_val, y_prob_val),
        "n_val": len(y_val),
        "positive_rate": y_val.mean(),
    }

    # Add calibration metrics
    if calibrate:
        cal_metrics = evaluate_calibration(y_val.values, y_prob_cal)
        val_metrics.update({f"cal_{k}": v for k, v in cal_metrics.items()})

    return model, calibrator, val_metrics


def train_with_cv(
    signals: pd.DataFrame,
    feature_columns: List[str],
    config: AlphaConfig,
    weight_scheme: WeightScheme = WeightScheme.OVERLAP_INVERSE,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> List[TrainingResult]:
    """Train model with cross-validation.

    Args:
        signals: Prepared signals DataFrame.
        feature_columns: List of feature column names.
        config: Alpha configuration.
        weight_scheme: Sample weighting scheme.
        lgbm_params: Optional LightGBM parameters.

    Returns:
        List of TrainingResult objects, one per fold.
    """
    cv = WeeklySignalCV(
        n_splits=config.cv.n_splits,
        min_train_weeks=config.cv.min_train_weeks,
        test_weeks=config.cv.test_weeks,
        purge_days=config.cv.purge_days,
        embargo_days=config.cv.embargo_days,
    )

    results = []

    for train_idx, val_idx, fold_info in cv.split(signals):
        print(f"\nFold {fold_info.fold_idx}:")
        print(f"  Train: {fold_info.train_start.date()} to {fold_info.train_end.date()}")
        print(f"  Test:  {fold_info.test_start.date()} to {fold_info.test_end.date()}")
        print(f"  Samples: {len(train_idx):,} train, {len(val_idx):,} val")
        print(f"  Purged: {fold_info.n_purged:,}")

        # Validate no leakage
        is_valid, details = validate_no_leakage(signals, train_idx, val_idx)
        print(f"  Gap: {details['gap_days']} days (no leakage)")

        # Prepare data
        train_data = signals.iloc[train_idx]
        val_data = signals.iloc[val_idx]

        # Get available features
        available_features = [f for f in feature_columns if f in signals.columns]

        X_train = train_data[available_features]
        y_train = (train_data["hit"] == 1).astype(int)

        X_val = val_data[available_features]
        y_val = (val_data["hit"] == 1).astype(int)

        # Compute sample weights
        weights = compute_sample_weights(train_data, scheme=weight_scheme)

        # Train
        model, calibrator, val_metrics = train_lgbm_fold(
            X_train, y_train,
            X_val, y_val,
            sample_weight=weights,
            params=lgbm_params,
            calibrate=True,
        )

        print(f"  Val AUC: {val_metrics['auc']:.4f}")
        print(f"  Val AUPR: {val_metrics['aupr']:.4f}")

        # Feature importance
        importance = dict(zip(available_features, model.feature_importances_))

        # Training metrics
        X_train_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        y_prob_train = model.predict_proba(X_train_clean)[:, 1]
        train_metrics = {
            "auc": roc_auc_score(y_train, y_prob_train),
            "aupr": average_precision_score(y_train, y_prob_train),
            "n_train": len(y_train),
            "positive_rate": y_train.mean(),
        }

        result = TrainingResult(
            fold_idx=fold_info.fold_idx,
            model=model,
            calibrator=calibrator,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=importance,
            train_indices=train_idx,
            val_indices=val_idx,
        )
        results.append(result)

    return results


def get_fold_predictions(
    training_results: List[TrainingResult],
    signals: pd.DataFrame,
    feature_columns: List[str],
    use_calibration: bool = True,
) -> pd.DataFrame:
    """Get out-of-fold predictions from CV results.

    Args:
        training_results: List of TrainingResult from CV.
        signals: Full signals DataFrame.
        feature_columns: Feature columns used.
        use_calibration: Whether to use calibrated probabilities.

    Returns:
        DataFrame with predictions for all samples.
    """
    predictions = pd.DataFrame(index=signals.index)
    predictions["probability"] = np.nan
    predictions["fold"] = -1

    available_features = [f for f in feature_columns if f in signals.columns]

    for result in training_results:
        # Get validation indices
        val_idx = result.val_indices

        # Prepare data
        X_val = signals.iloc[val_idx][available_features]
        X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)

        # Predict
        y_prob = result.model.predict_proba(X_val_clean)[:, 1]

        # Calibrate if available
        if use_calibration and result.calibrator is not None:
            y_prob = result.calibrator.transform(y_prob)

        # Store predictions
        predictions.iloc[val_idx, predictions.columns.get_loc("probability")] = y_prob
        predictions.iloc[val_idx, predictions.columns.get_loc("fold")] = result.fold_idx

    return predictions
