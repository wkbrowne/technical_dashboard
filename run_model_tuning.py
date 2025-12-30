#!/usr/bin/env python
"""
LightGBM Hyperparameter Tuning using Optuna.

Supports the 4-model system with per-model features and configuration tracking:
- LONG_NORMAL: Standard long momentum
- LONG_PARABOLIC: Extended momentum / trend persistence
- SHORT_NORMAL: Breakdown / fragility setups
- SHORT_PARABOLIC: Panic / regime shift scenarios

Features:
- **Future-biased pruning**: Evaluates trials on recent folds (default: 3,4)
  to avoid optimizing for stale market regimes
- **Multi-round HPO**: Runs optimization in rounds, narrowing search space
  using quantile-based refinement after each round
- **Composite objective**: Weighted combination of AUC, AUPR, calibration,
  tail performance, and stability metrics
- **Sample weights**: Uses triple barrier overlap inverse weighting

Usage:
    # Basic: 2 rounds, future-biased pruning on folds 3,4
    python run_model_tuning.py --model long_normal

    # Multi-round with more trials
    python run_model_tuning.py --model long_normal --rounds 3 --trials-per-round 150

    # All models with default settings
    python run_model_tuning.py --all-models

    # Legacy mode (prune on fold 0, single round)
    python run_model_tuning.py --model long_normal --prune-folds 0 --rounds 1

    # Weighted CV scoring (favor recent folds)
    python run_model_tuning.py --model long_normal --cv-score weighted

Output:
    - artifacts/hyperopt/{model_key}/best_params.json  # Per-model hyperparams
    - artifacts/hyperopt/model_configs.json             # Combined config registry
    - artifacts/hpo/{model_key}/{run_id}/report.md     # Detailed HPO report
"""

import gc
import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

warnings.filterwarnings('ignore', message='.*feature_name.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

# Set joblib temp folder
JOBLIB_TEMP = Path(__file__).parent / ".joblib_temp"
JOBLIB_TEMP.mkdir(exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import Parallel, delayed
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

sys.path.insert(0, str(Path(__file__).parent))

from src.config.model_keys import ModelKey, TARGET_CONFIGS
from src.feature_selection.base_features import get_featureset, CORE_FEATURES, HEAD_FEATURES
from src.config.hpo_config import (
    HPOConfig, PruningConfig as HPOPruningConfig, CVScoreConfig,
    RefinementConfig, RefineMethod, PruneMetric, CVScoreMethod,
    get_default_hpo_config, get_legacy_hpo_config,
)
from src.alpha.hpo.refinement import (
    SearchSpace, SearchSpaceParam, refine_search_space,
    get_default_lgbm_search_space,
)
from src.alpha.hpo.pruning import FoldEvaluator, should_prune_trial
from src.alpha.hpo.artifacts import (
    HPOArtifactWriter, HPORoundResult, PruningStats,
    compute_fold_auc_summary,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ObjectiveWeights:
    """
    Weights for composite objective function.

    The objective uses a "mean minus SE penalty" formulation:
        objective = S_mean - lambda_stability * S_se

    Where S_fold[i] is the per-fold composite score (weighted sum of metrics),
    and S_se = std(S_fold) / sqrt(n_folds) is the standard error.

    This aligns with the SNR-based gating used in feature selection:
    - Feature selection: AUC-only with mean gate + SNR gate (SE-based)
    - Hyperopt: composite score with SE penalty for stability

    Component weights (should sum to ~1.0 for interpretability):
    - auc, aupr: discrimination power
    - calibration: probability calibration (inverted Brier)
    - tail: precision in top decile (what we actually trade)
    - spread: separation between top and bottom decile
    """
    auc: float = 0.25
    aupr: float = 0.15
    calibration: float = 0.15
    tail: float = 0.25
    spread: float = 0.20
    # SE penalty coefficient: objective = S_mean - lambda_stability * S_se
    # Higher values penalize variance more heavily
    lambda_stability: float = 0.5


@dataclass
class PruningConfig:
    """Thresholds for early trial pruning (legacy format)."""
    min_auc: float = 0.54
    max_brier: float = 0.26
    max_cv_coef: float = 0.20


DEFAULT_WEIGHTS = ObjectiveWeights()
DEFAULT_PRUNING = PruningConfig()


# Default prune folds for future-biased pruning
DEFAULT_PRUNE_FOLDS = [3, 4]


# =============================================================================
# Data Loading
# =============================================================================

def load_model_features(model_key: ModelKey) -> list[str]:
    """
    Load features for a specific model from the feature registry.

    Uses the CORE + HEAD feature architecture from base_features.py.

    Args:
        model_key: The model key (LONG_NORMAL, LONG_PARABOLIC, etc.)

    Returns:
        List of feature names for this model
    """
    features = get_featureset(model_key, include_expansion=False, flat=True)
    return features


def load_selected_features_legacy() -> list[str]:
    """
    Load selected features from legacy feature selection output.

    DEPRECATED: Use load_model_features(model_key) for 4-model system.
    Falls back to selected_features.txt for backwards compatibility.
    """
    features_file = Path('artifacts/feature_selection/selected_features.txt')
    if not features_file.exists():
        raise FileNotFoundError(
            f"Selected features not found: {features_file}\n"
            "Run feature selection first: python run_feature_selection.py\n"
            "Or use --model flag for 4-model system."
        )

    features = []
    with open(features_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def get_model_target_column(model_key: ModelKey) -> str:
    """Get the target column name for a specific model."""
    target_col_map = {
        ModelKey.LONG_NORMAL: 'hit_long_normal',
        ModelKey.LONG_PARABOLIC: 'hit_long_parabolic',
        ModelKey.SHORT_NORMAL: 'hit_short_normal',
        ModelKey.SHORT_PARABOLIC: 'hit_short_parabolic',
    }
    return target_col_map.get(model_key, 'hit')


def load_and_prepare_data(
    selected_features: list[str],
    model_key: Optional[ModelKey] = None,
    max_symbols: int = 5000,
    min_samples_per_symbol: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Load features, targets, and sample weights.

    Args:
        selected_features: List of feature names to use
        model_key: Optional ModelKey for model-specific target labels
        max_symbols: Maximum symbols to include
        min_samples_per_symbol: Minimum samples per symbol

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary target array
        sample_weight: Overlap inverse weights
        dates: DatetimeIndex for CV splitting
        feature_names: List of feature names
    """
    print("Loading features...")
    features = pd.read_parquet('artifacts/features_complete.parquet')
    print(f"  Features shape: {features.shape}")

    print("Loading targets...")
    targets = pd.read_parquet('artifacts/targets_triple_barrier.parquet')
    print(f"  Targets shape: {targets.shape}")

    targets = targets.rename(columns={'t0': 'date'})

    # Filter symbols
    symbol_counts = targets.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= min_samples_per_symbol].index.tolist()
    if len(valid_symbols) > max_symbols:
        valid_symbols = symbol_counts.loc[valid_symbols].nlargest(max_symbols).index.tolist()

    features = features[features['symbol'].isin(valid_symbols)].copy()
    targets = targets[targets['symbol'].isin(valid_symbols)].copy()

    # Determine which columns to merge from targets
    target_cols = ['symbol', 'date', 'weight_final']

    # Get the appropriate hit column for this model
    if model_key is not None:
        hit_col = get_model_target_column(model_key)
        if hit_col in targets.columns:
            target_cols.append(hit_col)
            print(f"  Using model-specific target: {hit_col}")
        else:
            # Fallback to 'hit' if model-specific column doesn't exist
            target_cols.append('hit')
            hit_col = 'hit'
            print(f"  Warning: {get_model_target_column(model_key)} not found, using 'hit'")
    else:
        target_cols.append('hit')
        hit_col = 'hit'

    # Merge
    merged = features.merge(
        targets[target_cols],
        on=['symbol', 'date'],
        how='inner'
    )

    # Binary target (exclude neutral hit=0)
    merged = merged[merged[hit_col] != 0].copy()

    # For long models: upper barrier hit (1) = success
    # For short models: lower barrier hit (-1) = success
    if model_key is not None and model_key.is_short():
        merged['target'] = (merged[hit_col] == -1).astype(int)
    else:
        merged['target'] = (merged[hit_col] == 1).astype(int)

    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Check available features
    available_features = [f for f in selected_features if f in merged.columns]
    missing = set(selected_features) - set(available_features)
    if missing:
        print(f"  Warning: {len(missing)} features not in data: {list(missing)[:5]}...")

    print(f"  Using {len(available_features)} features")

    X = merged[available_features].values.astype(np.float32)
    y = merged['target'].values
    sample_weight = merged['weight_final'].values if 'weight_final' in merged.columns else None
    dates = pd.to_datetime(merged['date'])

    print(f"\nData prepared:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")
    print(f"  Date range: {dates.min().date()} to {dates.max().date()}")
    if sample_weight is not None:
        print(f"  Sample weights: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}")

    del features, targets, merged
    gc.collect()

    return X, y, sample_weight, dates, available_features


# =============================================================================
# Cross-Validation
# =============================================================================

def get_expanding_cv_splits(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    gap: int = 20,
    min_train_samples: int = 3000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding window CV splits with embargo gap.

    gap=20 matches max_horizon from triple barrier targets.
    """
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)

    min_train_dates = max(min_train_samples // 100, 50)
    available_dates = n_dates - min_train_dates
    test_size = available_dates // (n_splits + 1)

    splits = []
    for fold in range(n_splits):
        test_end_idx = n_dates - 1 - fold * test_size
        test_start_idx = test_end_idx - test_size + 1
        train_end_idx = test_start_idx - gap - 1

        if train_end_idx < min_train_dates:
            continue

        train_dates = unique_dates[:train_end_idx + 1]
        test_dates = unique_dates[test_start_idx:test_end_idx + 1]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= min_train_samples and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits[::-1]  # Chronological order


# =============================================================================
# Metrics
# =============================================================================

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all financial metrics for one CV fold."""
    n = len(y_pred)
    k = max(1, int(n * 0.10))

    sorted_idx = np.argsort(y_pred)
    top_idx = sorted_idx[-k:]
    bottom_idx = sorted_idx[:k]

    return {
        'auc': roc_auc_score(y_true, y_pred),
        'aupr': average_precision_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, np.clip(y_pred, 0, 1)),
        'precision_top_10': y_true[top_idx].mean(),
        'precision_bottom_10': y_true[bottom_idx].mean(),
    }


def compute_fold_composite_score(metrics: dict, weights: ObjectiveWeights) -> float:
    """
    Compute composite score for a single fold.

    This is the per-fold score S_fold[i] that gets aggregated across folds.
    Does NOT include any stability term - stability is handled at the
    aggregate level via SE penalty.

    Args:
        metrics: Dict with 'auc', 'aupr', 'brier', 'precision_top_10', 'precision_bottom_10'
        weights: Objective component weights

    Returns:
        Per-fold composite score (higher is better)
    """
    # Calibration: Brier score in [0, 0.25] for binary classification
    # Invert and clamp: perfect calibration (brier=0) -> 1.0, random (brier=0.25) -> 0.0
    calib = 1.0 - np.clip(metrics['brier'] / 0.25, 0.0, 1.0)

    # Spread: difference between top and bottom decile precision
    spread = metrics['precision_top_10'] - metrics['precision_bottom_10']

    # Weighted sum of components (no stability term here)
    score = (
        weights.auc * metrics['auc'] +
        weights.aupr * metrics['aupr'] +
        weights.calibration * calib +
        weights.tail * metrics['precision_top_10'] +
        weights.spread * spread
    )

    return score


@dataclass
class CompositeScoreResult:
    """Result from compute_composite_score with diagnostic info."""
    objective: float
    s_mean: float
    s_std: float
    s_se: float
    auc_mean: float
    auc_std: float
    auc_se: float
    aupr_mean: float
    brier_mean: float
    prec_top_mean: float
    prec_bot_mean: float
    fold_scores: List[float]
    pruned: bool = False
    prune_reason: Optional[str] = None


def compute_composite_score(
    fold_metrics: list[dict],
    weights: ObjectiveWeights,
    cv_score_method: str = "mean",
    fold_weights: Optional[List[float]] = None,
    baseline_auc_mean: Optional[float] = None,
    auc_floor_delta: float = 0.002,
) -> CompositeScoreResult:
    """
    Compute composite objective using per-fold scores with SE-based stability penalty.

    Philosophy:
    -----------
    This objective follows the "mean minus SE penalty" formulation that aligns with
    the SNR-based gating in feature selection:

        objective = S_mean - lambda_stability * S_se

    Where:
    - S_fold[i] = weighted sum of metrics for fold i (computed via compute_fold_composite_score)
    - S_mean = mean(S_fold) or weighted mean if cv_score_method="weighted"
    - S_se = std(S_fold, ddof=1) / sqrt(n_folds) is the standard error

    Why per-fold composite first, then aggregate?
    - Ensures stability penalty reflects true fold-to-fold variance in the
      objective we care about, not just AUC variance
    - Aligns with financial intuition: we want stable *overall* performance,
      not just stable discrimination

    Fold weighting:
    - In "weighted" mode, later folds get higher weight (recency bias)
    - SE penalty is SKIPPED in weighted mode because weighted variance estimation
      requires careful handling of effective sample size. For weighted CV, we
      assume the weighting itself provides regime-robustness.

    AUC floor constraint:
    - If baseline_auc_mean is provided, trials that sacrifice too much AUC
      (auc_mean < baseline - delta) are marked for pruning
    - This prevents the composite from trading off AUC for other metrics

    Args:
        fold_metrics: List of metric dicts from each fold
        weights: Objective component weights (including lambda_stability)
        cv_score_method: "mean" (with SE penalty) or "weighted" (no SE penalty)
        fold_weights: Weights per fold for weighted mode
        baseline_auc_mean: Reference AUC for floor constraint (optional)
        auc_floor_delta: Max allowable AUC drop from baseline (default: 0.002)

    Returns:
        CompositeScoreResult with objective value and diagnostics
    """
    n_folds = len(fold_metrics)

    # Compute per-fold composite scores
    fold_scores = [compute_fold_composite_score(m, weights) for m in fold_metrics]

    # Extract per-fold AUC for constraint checking
    fold_aucs = [m['auc'] for m in fold_metrics]

    # Compute per-fold means for diagnostics
    auc_values = np.array(fold_aucs)
    aupr_values = np.array([m['aupr'] for m in fold_metrics])
    brier_values = np.array([m['brier'] for m in fold_metrics])
    prec_top_values = np.array([m['precision_top_10'] for m in fold_metrics])
    prec_bot_values = np.array([m['precision_bottom_10'] for m in fold_metrics])

    # Handle fold weighting
    if cv_score_method == "weighted":
        # Generate default weights favoring later folds if not provided
        if fold_weights is None:
            # [0.5, 0.75, 1.0, 1.25, 1.5] for 5 folds
            base = 0.5
            step = 1.0 / (n_folds - 1) if n_folds > 1 else 0
            fold_weights = [base + i * step for i in range(n_folds)]

        # Normalize weights to sum to 1
        total = sum(fold_weights[:n_folds])
        norm_weights = np.array([w / total for w in fold_weights[:n_folds]])

        # Weighted means
        s_mean = float(np.sum(np.array(fold_scores) * norm_weights))
        auc_mean = float(np.sum(auc_values * norm_weights))
        aupr_mean = float(np.sum(aupr_values * norm_weights))
        brier_mean = float(np.sum(brier_values * norm_weights))
        prec_top_mean = float(np.sum(prec_top_values * norm_weights))
        prec_bot_mean = float(np.sum(prec_bot_values * norm_weights))

        # In weighted mode, skip SE penalty - the weighting itself provides
        # recency robustness, and proper weighted variance requires careful
        # effective-N handling that adds complexity without clear benefit
        s_std = float(np.std(fold_scores, ddof=1)) if n_folds > 1 else 0.0
        s_se = 0.0  # Explicitly zero - no SE penalty in weighted mode
        auc_std = float(np.std(auc_values, ddof=1)) if n_folds > 1 else 0.0
        auc_se = 0.0

        objective = s_mean
    else:
        # Simple mean with SE penalty
        s_mean = float(np.mean(fold_scores))
        s_std = float(np.std(fold_scores, ddof=1)) if n_folds > 1 else 0.0
        s_se = s_std / np.sqrt(n_folds) if n_folds > 1 else 0.0

        auc_mean = float(np.mean(auc_values))
        auc_std = float(np.std(auc_values, ddof=1)) if n_folds > 1 else 0.0
        auc_se = auc_std / np.sqrt(n_folds) if n_folds > 1 else 0.0

        aupr_mean = float(np.mean(aupr_values))
        brier_mean = float(np.mean(brier_values))
        prec_top_mean = float(np.mean(prec_top_values))
        prec_bot_mean = float(np.mean(prec_bot_values))

        # Core objective: mean minus lambda * SE
        objective = s_mean - weights.lambda_stability * s_se

    # Check AUC floor constraint
    pruned = False
    prune_reason = None
    if baseline_auc_mean is not None:
        if auc_mean < baseline_auc_mean - auc_floor_delta:
            pruned = True
            prune_reason = f"auc_floor_violated: {auc_mean:.4f} < {baseline_auc_mean:.4f} - {auc_floor_delta}"
            # Return a very low score to ensure this trial is not selected
            objective = -float('inf')

    return CompositeScoreResult(
        objective=objective,
        s_mean=s_mean,
        s_std=s_std,
        s_se=s_se,
        auc_mean=auc_mean,
        auc_std=auc_std,
        auc_se=auc_se,
        aupr_mean=aupr_mean,
        brier_mean=brier_mean,
        prec_top_mean=prec_top_mean,
        prec_bot_mean=prec_bot_mean,
        fold_scores=fold_scores,
        pruned=pruned,
        prune_reason=prune_reason,
    )


def compute_composite_score_value(
    fold_metrics: list[dict],
    weights: ObjectiveWeights,
    cv_score_method: str = "mean",
    fold_weights: Optional[List[float]] = None,
    baseline_auc_mean: Optional[float] = None,
    auc_floor_delta: float = 0.002,
) -> float:
    """
    Convenience wrapper that returns just the objective value.

    For backwards compatibility with code that expects a float return.
    """
    result = compute_composite_score(
        fold_metrics, weights, cv_score_method, fold_weights,
        baseline_auc_mean, auc_floor_delta
    )
    return result.objective


def compute_auc_score(fold_metrics: list[dict], variance_penalty: float) -> float:
    """
    Compute AUC-only objective with SE-based variance penalty.

    Uses the same "mean minus penalty * SE" formulation as the composite score:
        objective = auc_mean - variance_penalty * auc_se

    Args:
        fold_metrics: List of metric dicts from each fold
        variance_penalty: Coefficient for SE penalty (lambda)

    Returns:
        AUC objective value (higher is better)
    """
    auc_values = np.array([m['auc'] for m in fold_metrics])
    n_folds = len(auc_values)

    auc_mean = float(np.mean(auc_values))
    auc_std = float(np.std(auc_values, ddof=1)) if n_folds > 1 else 0.0
    auc_se = auc_std / np.sqrt(n_folds) if n_folds > 1 else 0.0

    return auc_mean - variance_penalty * auc_se


# =============================================================================
# Optuna Objective
# =============================================================================

def _train_single_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    params: dict,
) -> dict:
    """
    Train and evaluate a single CV fold.

    Designed to run in parallel via joblib.

    Returns:
        Dict with fold metrics
    """
    X_train = np.nan_to_num(X[train_idx], nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X[test_idx], nan=0.0, posinf=0.0, neginf=0.0)
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = sample_weight[train_idx] if sample_weight is not None else None

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict_proba(X_test)[:, 1]
    return compute_fold_metrics(y_test, y_pred)


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cv_splits: list,
    objective_mode: str,
    weights: ObjectiveWeights,
    pruning: PruningConfig,
    variance_penalty: float,
    threads_per_fold: int,
    parallel_folds: int,
    prune_folds: List[int] = None,
    prune_metric: str = "mean",
    prune_margin: float = 0.002,
    min_completed_trials: int = 30,
    eval_all_folds: bool = True,
    cv_score_method: str = "mean",
    fold_weights: Optional[List[float]] = None,
    search_space: Optional[SearchSpace] = None,
    fold_evaluator: Optional[FoldEvaluator] = None,
    pruning_stats: Optional[PruningStats] = None,
):
    """
    Create Optuna objective function with future-biased pruning.

    Args:
        X, y, sample_weight: Training data
        cv_splits: List of (train_idx, test_idx) tuples
        objective_mode: "composite" or "auc"
        weights: Objective component weights
        pruning: Absolute pruning thresholds
        variance_penalty: For AUC mode
        threads_per_fold: LightGBM threads
        parallel_folds: Number of folds to run in parallel
        prune_folds: Which folds to use for pruning (default: [3, 4])
        prune_metric: "mean" or "min" for prune fold aggregation
        prune_margin: Margin for relative pruning
        min_completed_trials: Trials before relative pruning activates
        eval_all_folds: Whether to evaluate all folds for survivors
        cv_score_method: "mean" or "weighted" for final score
        fold_weights: Weights per fold for weighted scoring
        search_space: SearchSpace object for refined bounds
        fold_evaluator: FoldEvaluator for tracking best prune metrics
        pruning_stats: PruningStats for tracking pruning behavior
    """
    if prune_folds is None:
        prune_folds = DEFAULT_PRUNE_FOLDS

    n_folds = len(cv_splits)

    # Create fold evaluator if not provided
    if fold_evaluator is None:
        fold_evaluator = FoldEvaluator(
            prune_folds=prune_folds,
            prune_metric=prune_metric,
            prune_margin=prune_margin,
            min_completed_trials=min_completed_trials,
            min_auc=pruning.min_auc,
            max_brier=pruning.max_brier,
            max_cv_coef=pruning.max_cv_coef,
            eval_all_folds=eval_all_folds,
        )

    # Sort folds: prune folds first, then remaining
    prune_fold_set = set(prune_folds)
    remaining_folds = [i for i in range(n_folds) if i not in prune_fold_set]

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters (use search_space if provided)
        if search_space is not None:
            # Use refined search space
            suggested_params = {}
            for name, param in search_space.params.items():
                if param.is_frozen():
                    suggested_params[name] = param.frozen_value
                elif param.param_type.value == 'float' or param.param_type.value == 'float_log':
                    suggested_params[name] = trial.suggest_float(
                        name, param.low, param.high, log=param.is_log_scale()
                    )
                elif param.param_type.value == 'int' or param.param_type.value == 'int_log':
                    suggested_params[name] = trial.suggest_int(
                        name, int(param.low), int(param.high), log=param.is_log_scale()
                    )
                elif param.param_type.value == 'categorical':
                    suggested_params[name] = trial.suggest_categorical(name, param.choices)

            num_leaves_exp = suggested_params.get('num_leaves_exp', 6.0)
            num_leaves = int(2 ** num_leaves_exp)
            min_child_samples = suggested_params.get('min_child_samples', 100)
        else:
            # Default search space (legacy behavior)
            num_leaves_exp = trial.suggest_float('num_leaves_exp', 4.0, 8.0)
            num_leaves = int(2 ** num_leaves_exp)
            min_child_samples = trial.suggest_int('min_child_samples', 50, 3000, log=True)
            suggested_params = {
                'num_leaves_exp': num_leaves_exp,
                'min_child_samples': min_child_samples,
                'max_depth': trial.suggest_categorical('max_depth', [-1, 4, 6, 8, 10]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
                'use_balanced': trial.suggest_categorical('use_balanced', [True, False]),
            }

        # Constraint: min_child_samples >= 2 * num_leaves for complexity control
        min_child_samples = max(min_child_samples, 2 * num_leaves)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'num_threads': threads_per_fold,
            'n_estimators': 5000,
            'max_depth': suggested_params.get('max_depth', 6),
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'learning_rate': suggested_params.get('learning_rate', 0.05),
            'reg_alpha': suggested_params.get('reg_alpha', 0.1),
            'reg_lambda': suggested_params.get('reg_lambda', 0.1),
            'min_split_gain': suggested_params.get('min_split_gain', 0.0),
            'subsample': suggested_params.get('subsample', 0.8),
            'subsample_freq': suggested_params.get('subsample_freq', 5),
            'colsample_bytree': suggested_params.get('colsample_bytree', 0.8),
            'max_bin': suggested_params.get('max_bin', 255),
        }

        # For non-unlimited max_depth, also constrain num_leaves
        if params['max_depth'] > 0:
            max_leaves = 2 ** params['max_depth']
            if params['num_leaves'] > max_leaves:
                params['num_leaves'] = max_leaves

        # Optional class balancing
        use_balanced = suggested_params.get('use_balanced', False)
        if use_balanced:
            params['scale_pos_weight'] = (y == 0).sum() / (y == 1).sum()

        # =====================================================================
        # Future-biased pruning: Evaluate prune folds first
        # =====================================================================

        all_fold_metrics = {}  # fold_idx -> metrics

        # Step 1: Evaluate prune folds (for early pruning decision)
        for fold_idx in prune_folds:
            if fold_idx >= n_folds:
                continue
            train_idx, test_idx = cv_splits[fold_idx]
            fold_metrics = _train_single_fold(
                train_idx, test_idx, X, y, sample_weight, params
            )
            all_fold_metrics[fold_idx] = fold_metrics

            # Store per-fold AUC for diagnostics
            trial.set_user_attr(f'fold_{fold_idx}_auc', fold_metrics['auc'])

        # Check pruning on prune folds
        prune_result = fold_evaluator.evaluate_prune_folds(all_fold_metrics)

        if prune_result.should_prune:
            trial.set_user_attr('prune_reason', prune_result.prune_reason)
            if pruning_stats:
                pruning_stats.record_prune(prune_result.prune_reason)
            raise optuna.TrialPruned()

        # Record prune metric for relative pruning tracking
        if pruning_stats:
            pruning_stats.prune_metric_distribution.append(prune_result.prune_metric_value)

        # Optuna intermediate reporting
        trial.report(prune_result.mean_auc, 0)
        if trial.should_prune():
            trial.set_user_attr('prune_reason', 'optuna_pruner')
            if pruning_stats:
                pruning_stats.record_prune('optuna_pruner')
            raise optuna.TrialPruned()

        # Step 2: Evaluate remaining folds (if eval_all_folds=True)
        if eval_all_folds and remaining_folds:
            remaining_metrics = Parallel(n_jobs=parallel_folds, prefer="threads")(
                delayed(_train_single_fold)(
                    cv_splits[fold_idx][0], cv_splits[fold_idx][1],
                    X, y, sample_weight, params
                )
                for fold_idx in remaining_folds
            )
            for fold_idx, metrics in zip(remaining_folds, remaining_metrics):
                all_fold_metrics[fold_idx] = metrics
                trial.set_user_attr(f'fold_{fold_idx}_auc', metrics['auc'])

        # Final pruning check on all folds
        if objective_mode == 'composite' and len(all_fold_metrics) > len(prune_folds):
            should_prune, reason = fold_evaluator.evaluate_all_folds(all_fold_metrics)
            if should_prune:
                trial.set_user_attr('prune_reason', reason)
                if pruning_stats:
                    pruning_stats.record_prune(reason)
                raise optuna.TrialPruned()

        # Record completed trial
        fold_evaluator.record_completed_trial(prune_result.prune_metric_value)

        # Convert to list ordered by fold index
        fold_metrics_list = [all_fold_metrics[i] for i in sorted(all_fold_metrics.keys())]

        # Compute objective
        if objective_mode == 'composite':
            # Get baseline AUC from best trial so far (for AUC floor constraint)
            baseline_auc = None
            try:
                study = trial.study
                if len(study.trials) > 0:
                    completed = [t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE]
                    if completed:
                        best_trial = max(completed, key=lambda t: t.value)
                        baseline_auc = best_trial.user_attrs.get('auc_mean')
            except Exception:
                pass  # No baseline available yet

            result = compute_composite_score(
                fold_metrics_list, weights,
                cv_score_method=cv_score_method,
                fold_weights=fold_weights,
                baseline_auc_mean=baseline_auc,
                auc_floor_delta=0.002,
            )

            # Store diagnostics in trial user attributes
            trial.set_user_attr('auc_mean', result.auc_mean)
            trial.set_user_attr('auc_std', result.auc_std)
            trial.set_user_attr('auc_se', result.auc_se)
            trial.set_user_attr('aupr_mean', result.aupr_mean)
            trial.set_user_attr('brier_mean', result.brier_mean)
            trial.set_user_attr('precision_top_10_mean', result.prec_top_mean)
            trial.set_user_attr('precision_bottom_10_mean', result.prec_bot_mean)
            trial.set_user_attr('s_mean', result.s_mean)
            trial.set_user_attr('s_std', result.s_std)
            trial.set_user_attr('s_se', result.s_se)
            trial.set_user_attr('fold_scores', result.fold_scores)

            # Handle AUC floor pruning
            if result.pruned:
                trial.set_user_attr('prune_reason', result.prune_reason)
                if pruning_stats:
                    pruning_stats.record_prune('auc_floor')
                raise optuna.TrialPruned()

            return result.objective
        else:
            # AUC-only mode with SE penalty
            auc_values = [m['auc'] for m in fold_metrics_list]
            auc_mean = float(np.mean(auc_values))
            auc_std = float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0
            auc_se = auc_std / np.sqrt(len(auc_values)) if len(auc_values) > 1 else 0.0

            trial.set_user_attr('auc_mean', auc_mean)
            trial.set_user_attr('auc_std', auc_std)
            trial.set_user_attr('auc_se', auc_se)

            # Store other metrics for diagnostics
            for metric in ['aupr', 'brier', 'precision_top_10', 'precision_bottom_10']:
                values = [m[metric] for m in fold_metrics_list]
                trial.set_user_attr(f'{metric}_mean', np.mean(values))

            return compute_auc_score(fold_metrics_list, variance_penalty)

    return objective, fold_evaluator


# =============================================================================
# Trial Callback
# =============================================================================

def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Print trial results with SE-based diagnostics."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        auc_mean = trial.user_attrs.get('auc_mean', 0)
        auc_se = trial.user_attrs.get('auc_se', 0)
        s_mean = trial.user_attrs.get('s_mean', 0)
        s_se = trial.user_attrs.get('s_se', 0)
        brier_mean = trial.user_attrs.get('brier_mean', 0)
        prec_top = trial.user_attrs.get('precision_top_10_mean', 0)

        is_best = study.best_trial.number == trial.number
        marker = " ** BEST **" if is_best else ""

        # Show objective with SE penalty decomposition
        if s_mean > 0:
            print(f"Trial {trial.number:3d}: Score={trial.value:.4f} "
                  f"(S={s_mean:.4f}±SE{s_se:.4f})  "
                  f"AUC={auc_mean:.4f}±SE{auc_se:.4f}  "
                  f"Brier={brier_mean:.4f}  "
                  f"Prec@10={prec_top:.4f}{marker}")
        else:
            # AUC-only mode
            print(f"Trial {trial.number:3d}: Score={trial.value:.4f}  "
                  f"AUC={auc_mean:.4f}±SE{auc_se:.4f}  "
                  f"Brier={brier_mean:.4f}  "
                  f"Prec@10={prec_top:.4f}{marker}")

    elif trial.state == optuna.trial.TrialState.PRUNED:
        reason = trial.user_attrs.get('prune_reason', 'optuna_pruner')
        print(f"Trial {trial.number:3d}: PRUNED ({reason})")


# =============================================================================
# Config Registry Management
# =============================================================================

def load_model_configs() -> Dict[str, Any]:
    """Load the combined model configs registry."""
    config_file = Path('artifacts/hyperopt/model_configs.json')
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {'models': {}, 'updated': None}


def save_model_configs(configs: Dict[str, Any]) -> None:
    """Save the combined model configs registry."""
    config_file = Path('artifacts/hyperopt/model_configs.json')
    config_file.parent.mkdir(parents=True, exist_ok=True)
    configs['updated'] = datetime.now().isoformat()
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)


def update_model_config(
    model_key: ModelKey,
    best_params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_names: List[str],
    target_config: Dict[str, Any],
) -> None:
    """
    Update the model configs registry with results for a single model.

    Args:
        model_key: The model key
        best_params: Best hyperparameters from Optuna
        metrics: CV metrics (auc_mean, aupr_mean, etc.)
        feature_names: List of features used
        target_config: Target configuration (ATR multiples, etc.)
    """
    configs = load_model_configs()

    configs['models'][model_key.value] = {
        'hyperparameters': {k: v for k, v in best_params.items() if not k.startswith('_')},
        'metrics': metrics,
        'features': {
            'count': len(feature_names),
            'core_count': len(CORE_FEATURES),
            'head_count': len(HEAD_FEATURES.get(model_key, [])),
            'names': feature_names,
        },
        'target_config': target_config,
        'tuning_timestamp': datetime.now().isoformat(),
    }

    save_model_configs(configs)


# =============================================================================
# Main
# =============================================================================

def run_hyperopt(
    model_key: Optional[ModelKey] = None,
    n_trials: int = 200,
    timeout: Optional[int] = None,
    threads_per_fold: Optional[int] = None,
    n_folds: int = 5,
    gap: int = 20,
    objective_mode: str = 'composite',
    variance_penalty: float = 1.0,
    resume: bool = False,
    # New HPO options
    rounds: int = 2,
    trials_per_round: Optional[int] = None,
    prune_folds: Optional[List[int]] = None,
    prune_metric: str = "mean",
    prune_margin: float = 0.002,
    min_completed_trials: int = 30,
    eval_all_folds: bool = True,
    cv_score_method: str = "mean",
    refine_method: str = "quantile",
    elite_frac: float = 0.15,
    refine_quantiles: tuple = (0.1, 0.9),
    refine_padding: float = 0.10,
    min_range_frac: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Run hyperparameter optimization for a specific model.

    Supports multi-round HPO with future-biased pruning and search space refinement.

    Args:
        model_key: The model to optimize (default: LONG_NORMAL)
        n_trials: Total trials (ignored if trials_per_round is set)
        timeout: Timeout in seconds per round
        threads_per_fold: LightGBM threads per fold (default: auto)
        n_folds: Number of CV folds
        gap: Embargo gap in days
        objective_mode: 'composite' or 'auc'
        variance_penalty: Penalty for AUC std (auc mode only)
        resume: Resume from saved study
        rounds: Number of HPO rounds (default: 2)
        trials_per_round: Trials per round (default: n_trials // rounds)
        prune_folds: Which folds for pruning (default: [3, 4])
        prune_metric: "mean" or "min" for prune fold aggregation
        prune_margin: Margin for relative pruning
        min_completed_trials: Trials before relative pruning
        eval_all_folds: Evaluate all folds for survivors
        cv_score_method: "mean" or "weighted"
        refine_method: "quantile" (simple) or "importance" (advanced)
        elite_frac: Fraction of trials for refinement
        refine_quantiles: Quantile bounds for numeric params
        refine_padding: Padding factor for refined bounds
        min_range_frac: Minimum range fraction
        seed: Random seed

    Returns:
        Dict with best params and results
    """
    # Default to LONG_NORMAL for backwards compatibility
    if model_key is None:
        model_key = ModelKey.LONG_NORMAL

    # Set default prune folds if not specified
    if prune_folds is None:
        prune_folds = DEFAULT_PRUNE_FOLDS

    # Calculate trials per round
    if trials_per_round is None:
        trials_per_round = max(50, n_trials // rounds)

    total_trials = rounds * trials_per_round

    # Compute parallelism
    n_cpus = os.cpu_count() or 8
    # For future-biased pruning, remaining folds are parallelized after prune folds
    n_remaining_folds = n_folds - len(prune_folds)
    parallel_folds = max(1, n_remaining_folds)
    if threads_per_fold is None:
        threads_per_fold = max(1, n_cpus // max(parallel_folds, 1))
    else:
        max_parallel = max(1, n_cpus // threads_per_fold)
        parallel_folds = min(parallel_folds, max_parallel)

    print("=" * 70)
    print("LightGBM Hyperparameter Optimization")
    print(f"  Model: {model_key.value.upper()}")
    print(f"  Objective: {objective_mode}")
    print(f"  Rounds: {rounds} × {trials_per_round} trials = {total_trials} total")
    print(f"  Prune folds: {prune_folds} (future-biased)")
    print(f"  CV score: {cv_score_method}")
    print("=" * 70)
    print()

    # Load features for this model from the registry
    selected_features = load_model_features(model_key)
    print(f"  {len(selected_features)} features for {model_key.value}")
    print(f"    - CORE: {len(CORE_FEATURES)} features")
    print(f"    - HEAD: {len(HEAD_FEATURES.get(model_key, []))} features\n")

    X, y, sample_weight, dates, feature_names = load_and_prepare_data(
        selected_features, model_key=model_key
    )

    # CV splits
    print(f"\nGenerating {n_folds}-fold expanding CV with {gap}-day embargo...")
    cv_splits = get_expanding_cv_splits(dates, n_splits=n_folds, gap=gap)
    print(f"  Generated {len(cv_splits)} folds")
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        prune_marker = " [PRUNE]" if i in prune_folds else ""
        print(f"    Fold {i}: {len(train_idx):,} train, {len(test_idx):,} test{prune_marker}")

    # Create output directories
    legacy_output_dir = Path('artifacts/hyperopt') / model_key.value
    legacy_output_dir.mkdir(parents=True, exist_ok=True)

    # Create artifact writer for detailed HPO artifacts
    artifact_writer = HPOArtifactWriter(
        base_dir="artifacts/hpo",
        study_name=model_key.value,
    )

    # Initialize search space
    search_space = get_default_lgbm_search_space()

    # Track results across rounds
    round_results = []
    global_best_params = None
    global_best_score = float('-inf')
    global_best_metrics = {}
    total_elapsed = 0.0

    start_time_total = time.time()

    # =========================================================================
    # Multi-round HPO loop
    # =========================================================================
    for round_num in range(rounds):
        print()
        print("=" * 70)
        print(f"ROUND {round_num + 1}/{rounds}")
        print("=" * 70)

        # Save current search space
        artifact_writer.save_search_space(round_num, search_space.to_dict())

        # Create study for this round
        study_name = f'lgbm_hyperopt_{model_key.value}_round_{round_num}'
        storage = f'sqlite:///{legacy_output_dir}/study_round_{round_num}.db'

        sampler = TPESampler(seed=seed + round_num, n_startup_trials=15, multivariate=True)
        pruner = HyperbandPruner(min_resource=1, max_resource=len(prune_folds), reduction_factor=2)

        # Delete existing study for fresh start (unless resuming)
        if not resume:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                pass

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
        )

        # Create fold evaluator and pruning stats for this round
        fold_evaluator = FoldEvaluator(
            prune_folds=prune_folds,
            prune_metric=prune_metric,
            prune_margin=prune_margin,
            min_completed_trials=min_completed_trials,
            min_auc=DEFAULT_PRUNING.min_auc,
            max_brier=DEFAULT_PRUNING.max_brier,
            max_cv_coef=DEFAULT_PRUNING.max_cv_coef,
            eval_all_folds=eval_all_folds,
        )
        pruning_stats = PruningStats()

        # Create objective with current search space
        objective, fold_evaluator = create_objective(
            X=X,
            y=y,
            sample_weight=sample_weight,
            cv_splits=cv_splits,
            objective_mode=objective_mode,
            weights=DEFAULT_WEIGHTS,
            pruning=DEFAULT_PRUNING,
            variance_penalty=variance_penalty,
            threads_per_fold=threads_per_fold,
            parallel_folds=parallel_folds,
            prune_folds=prune_folds,
            prune_metric=prune_metric,
            prune_margin=prune_margin,
            min_completed_trials=min_completed_trials,
            eval_all_folds=eval_all_folds,
            cv_score_method=cv_score_method,
            search_space=search_space if round_num > 0 else None,  # Use default for round 0
            fold_evaluator=fold_evaluator,
            pruning_stats=pruning_stats,
        )

        print(f"\nStarting round {round_num + 1}...")
        print(f"  Trials: {trials_per_round}")
        print(f"  Prune folds: {prune_folds} ({prune_metric})")
        print(f"  Sample weights: {'enabled' if sample_weight is not None else 'disabled'}")
        print()

        round_start = time.time()

        study.optimize(
            objective,
            n_trials=trials_per_round,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=False,
            gc_after_trial=True,
            callbacks=[trial_callback],
        )

        round_elapsed = time.time() - round_start
        total_elapsed += round_elapsed

        # Collect round results
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        print(f"\nRound {round_num + 1} complete: {n_completed} completed, {n_pruned} pruned")
        print(f"Time: {round_elapsed/60:.1f} minutes")

        if n_completed == 0:
            print("  WARNING: No completed trials in this round!")
            continue

        best = study.best_trial
        round_metrics = {
            'composite_score': study.best_value,
            'auc_mean': best.user_attrs.get('auc_mean'),
            'auc_std': best.user_attrs.get('auc_std'),
            'aupr_mean': best.user_attrs.get('aupr_mean'),
            'brier_mean': best.user_attrs.get('brier_mean'),
            'precision_top_10_mean': best.user_attrs.get('precision_top_10_mean'),
            'precision_bottom_10_mean': best.user_attrs.get('precision_bottom_10_mean'),
        }

        print(f"  Best score: {study.best_value:.4f}")
        print(f"  AUC: {round_metrics['auc_mean']:.4f} ± {round_metrics['auc_std']:.4f}")

        # Collect per-fold AUCs
        fold_aucs = {}
        for fold_idx in range(n_folds):
            auc = best.user_attrs.get(f'fold_{fold_idx}_auc')
            if auc is not None:
                fold_aucs[fold_idx] = auc

        # Update global best
        if study.best_value > global_best_score:
            global_best_score = study.best_value
            global_best_params = study.best_params.copy()
            global_best_metrics = round_metrics.copy()

        # Compute parameter importance
        param_importance = None
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            pass

        # Save round artifacts
        artifact_writer.save_best_params(round_num, study.best_params, round_metrics)
        artifact_writer.save_trials_history(round_num, study.trials_dataframe())
        artifact_writer.save_pruning_stats(round_num, pruning_stats)

        if param_importance:
            artifact_writer.save_param_importance(round_num, param_importance)

        # Create round result
        round_result = HPORoundResult(
            round_num=round_num,
            n_trials=trials_per_round,
            n_completed=n_completed,
            n_pruned=n_pruned,
            best_score=study.best_value,
            best_params=study.best_params,
            metrics=round_metrics,
            fold_aucs=fold_aucs,
            elapsed_seconds=round_elapsed,
            search_space=search_space.to_dict(),
            param_importance=param_importance,
        )
        round_results.append(round_result)
        artifact_writer.add_round_result(round_result)

        # =====================================================================
        # Refine search space for next round (if not last round)
        # =====================================================================
        if round_num < rounds - 1:
            print(f"\nRefining search space for round {round_num + 2}...")

            # Get elite trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            n_elite = max(5, int(len(completed_trials) * elite_frac))
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
            elite_trials = sorted_trials[:n_elite]

            # Extract elite parameters
            elite_params = [t.params for t in elite_trials]

            print(f"  Using top {n_elite} trials ({elite_frac*100:.0f}%) for refinement")

            # Refine search space
            if refine_method == "importance" and param_importance:
                search_space = refine_search_space(
                    search_space=search_space,
                    elite_params=elite_params,
                    q_low=refine_quantiles[0],
                    q_high=refine_quantiles[1],
                    padding=refine_padding,
                    min_range_frac=min_range_frac,
                    param_importance=param_importance,
                    freeze_low_importance=True,
                    importance_threshold=0.02,
                )
            else:
                # Simple quantile refinement (default)
                search_space = refine_search_space(
                    search_space=search_space,
                    elite_params=elite_params,
                    q_low=refine_quantiles[0],
                    q_high=refine_quantiles[1],
                    padding=refine_padding,
                    min_range_frac=min_range_frac,
                )

            # Report refinement changes
            frozen = search_space.get_frozen_params()
            if frozen:
                print(f"  Frozen parameters: {list(frozen.keys())}")

            for name, param in search_space.params.items():
                if param.is_numeric() and not param.is_frozen():
                    orig_range = param.get_original_range()
                    new_range = param.get_range()
                    if orig_range > 0:
                        shrink_pct = (1 - new_range / orig_range) * 100
                        if shrink_pct > 5:  # Only report significant changes
                            print(f"  {name}: narrowed by {shrink_pct:.0f}%")

        gc.collect()

    # =========================================================================
    # Final results
    # =========================================================================
    total_elapsed = time.time() - start_time_total

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Rounds completed: {len(round_results)}")

    # Aggregate trial counts
    total_completed = sum(r.n_completed for r in round_results)
    total_pruned = sum(r.n_pruned for r in round_results)
    print(f"Total trials: {total_completed} completed, {total_pruned} pruned")

    if global_best_params:
        print(f"\nGlobal Best Score: {global_best_score:.4f}")
        print(f"\nGlobal Best Metrics:")
        print(f"  AUC:           {global_best_metrics.get('auc_mean', 0):.4f} ± {global_best_metrics.get('auc_std', 0):.4f}")
        print(f"  AUPR:          {global_best_metrics.get('aupr_mean', 0):.4f}")
        print(f"  Brier:         {global_best_metrics.get('brier_mean', 0):.4f}")
        print(f"  Precision@10%: {global_best_metrics.get('precision_top_10_mean', 0):.4f}")

        print(f"\nGlobal Best Hyperparameters:")
        for key, value in global_best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

        # Save final results
        best_params = global_best_params.copy()
        best_params['_metrics'] = global_best_metrics
        best_params['_metadata'] = {
            'model_key': model_key.value,
            'n_rounds': rounds,
            'trials_per_round': trials_per_round,
            'total_trials': total_completed + total_pruned,
            'n_completed': total_completed,
            'n_pruned': total_pruned,
            'elapsed_seconds': total_elapsed,
            'timestamp': datetime.now().isoformat(),
            'objective_mode': objective_mode,
            'prune_folds': prune_folds,
            'prune_metric': prune_metric,
            'cv_score_method': cv_score_method,
            'refine_method': refine_method,
            'sample_weights_used': sample_weight is not None,
            'features': feature_names,
            'feature_count': len(feature_names),
            'core_features_count': len(CORE_FEATURES),
            'head_features_count': len(HEAD_FEATURES.get(model_key, [])),
        }

        # Save to legacy location
        with open(legacy_output_dir / 'best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\nBest params saved to: {legacy_output_dir / 'best_params.json'}")

        # Save final artifacts
        artifact_writer.save_final_best_params(
            global_best_params, global_best_metrics, best_params['_metadata']
        )

        # Generate report
        report_path = artifact_writer.generate_report()
        print(f"HPO report: {report_path}")

        # Update the combined model configs registry
        target_config = TARGET_CONFIGS.get(model_key, {})
        update_model_config(
            model_key=model_key,
            best_params=global_best_params,
            metrics=global_best_metrics,
            feature_names=feature_names,
            target_config=target_config,
        )
        print(f"Model config registry updated: artifacts/hyperopt/model_configs.json")

        print()
        print("=" * 70)
        print(f"Next step: python run_training.py --model {model_key.value}")
        print("=" * 70)

        return best_params
    else:
        print("\nWARNING: No completed trials across all rounds!")
        return {}


def run_all_models(
    n_trials: int = 100,
    timeout: Optional[int] = None,
    threads_per_fold: Optional[int] = None,
    n_folds: int = 5,
    gap: int = 20,
    objective_mode: str = 'composite',
    variance_penalty: float = 1.0,
    # New HPO options
    rounds: int = 2,
    trials_per_round: Optional[int] = None,
    prune_folds: Optional[List[int]] = None,
    prune_metric: str = "mean",
    cv_score_method: str = "mean",
    refine_method: str = "quantile",
) -> Dict[str, dict]:
    """
    Run hyperparameter optimization for all 4 models.

    Args:
        n_trials: Total trials per model (ignored if trials_per_round set)
        timeout: Timeout in seconds per model
        threads_per_fold: LightGBM threads per fold (default: auto)
        n_folds: Number of CV folds
        gap: Embargo gap in days
        objective_mode: 'composite' or 'auc'
        variance_penalty: Penalty for AUC std (auc mode only)
        rounds: Number of HPO rounds
        trials_per_round: Trials per round
        prune_folds: Which folds for pruning
        prune_metric: "mean" or "min"
        cv_score_method: "mean" or "weighted"
        refine_method: "quantile" or "importance"

    Returns:
        Dict mapping model_key to best params
    """
    if prune_folds is None:
        prune_folds = DEFAULT_PRUNE_FOLDS

    results = {}
    start_time = time.time()

    print("=" * 70)
    print("MULTI-MODEL HYPERPARAMETER OPTIMIZATION")
    print(f"  Models: {', '.join(mk.value for mk in ModelKey.all_keys())}")
    print(f"  Rounds: {rounds}, Prune folds: {prune_folds}")
    print("=" * 70)
    print()

    for i, model_key in enumerate(ModelKey.all_keys()):
        print(f"\n[{i+1}/4] Running hyperopt for {model_key.value.upper()}")
        print("=" * 70)

        best_params = run_hyperopt(
            model_key=model_key,
            n_trials=n_trials,
            timeout=timeout,
            threads_per_fold=threads_per_fold,
            n_folds=n_folds,
            gap=gap,
            objective_mode=objective_mode,
            variance_penalty=variance_penalty,
            resume=False,
            rounds=rounds,
            trials_per_round=trials_per_round,
            prune_folds=prune_folds,
            prune_metric=prune_metric,
            cv_score_method=cv_score_method,
            refine_method=refine_method,
        )
        results[model_key.value] = best_params

        gc.collect()

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("\nResults per model:")
    print("-" * 50)
    print(f"{'Model':<20} {'AUC':>12} {'AUPR':>12} {'Score':>12}")
    print("-" * 50)

    for model_key in ModelKey.all_keys():
        params = results.get(model_key.value, {})
        metrics = params.get('_metrics', {})
        auc = metrics.get('auc_mean', 0)
        aupr = metrics.get('aupr_mean', 0)
        score = metrics.get('composite_score', 0)
        print(f"{model_key.value:<20} {auc:>12.4f} {aupr:>12.4f} {score:>12.4f}")

    print("\nConfig registry: artifacts/hyperopt/model_configs.json")
    print("Next step: python run_training.py --all-models")

    return results


def parse_prune_folds(value: str) -> List[int]:
    """Parse comma-separated prune folds string."""
    return [int(x.strip()) for x in value.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='LightGBM Hyperparameter Optimization with Multi-Round Refinement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 2 rounds with future-biased pruning on folds 3,4
    python run_model_tuning.py --model long_normal

    # Multi-round with explicit settings
    python run_model_tuning.py --model long_normal --rounds 3 --trials-per-round 150

    # All models with weighted CV scoring
    python run_model_tuning.py --all-models --cv-score weighted

    # Legacy mode (single round, prune on fold 0)
    python run_model_tuning.py --model long_normal --rounds 1 --prune-folds 0

    # Importance-based refinement (advanced)
    python run_model_tuning.py --model long_normal --refine-method importance

Suggested Budgets:
    # LightGBM per model (thorough): 3 rounds × 150 trials = 450 total
    python run_model_tuning.py --model long_normal --rounds 3 --trials-per-round 150

    # Quick exploration: 2 rounds × 100 trials = 200 total
    python run_model_tuning.py --model long_normal --rounds 2 --trials-per-round 100
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default=None,
                        choices=['long_normal', 'long_parabolic', 'short_normal', 'short_parabolic'],
                        help='Model key to optimize (default: long_normal)')
    parser.add_argument('--all-models', action='store_true',
                        help='Run optimization for all 4 models')

    # Trial configuration
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Total trials (used if --trials-per-round not set)')
    parser.add_argument('--rounds', type=int, default=2,
                        help='Number of HPO rounds (default: 2)')
    parser.add_argument('--trials-per-round', type=int, default=None,
                        help='Trials per round (default: n_trials // rounds)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds per round')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved study')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Pruning configuration (future-biased)
    prune_group = parser.add_argument_group('Pruning (Future-Biased)')
    prune_group.add_argument('--prune-folds', type=str, default='3,4',
                             help='Comma-separated fold indices for pruning (default: 3,4)')
    prune_group.add_argument('--prune-metric', type=str, default='mean',
                             choices=['mean', 'min'],
                             help='How to aggregate prune fold AUCs (default: mean)')
    prune_group.add_argument('--prune-margin', type=float, default=0.002,
                             help='Margin for relative pruning (default: 0.002)')
    prune_group.add_argument('--prune-min-completed', type=int, default=30,
                             help='Min trials before relative pruning (default: 30)')
    prune_group.add_argument('--eval-all-folds', type=str, default='true',
                             choices=['true', 'false'],
                             help='Evaluate all folds for survivors (default: true)')

    # CV scoring
    cv_group = parser.add_argument_group('CV Scoring')
    cv_group.add_argument('--cv-score', type=str, default='mean',
                          choices=['mean', 'weighted'],
                          help='CV score aggregation (default: mean)')

    # Search space refinement
    refine_group = parser.add_argument_group('Search Space Refinement')
    refine_group.add_argument('--refine-method', type=str, default='quantile',
                              choices=['quantile', 'importance'],
                              help='Refinement method (default: quantile)')
    refine_group.add_argument('--elite-frac', type=float, default=0.15,
                              help='Fraction of trials for refinement (default: 0.15)')
    refine_group.add_argument('--refine-quantiles', type=str, default='0.1,0.9',
                              help='Quantile bounds for refinement (default: 0.1,0.9)')
    refine_group.add_argument('--refine-padding', type=float, default=0.10,
                              help='Padding factor for refined bounds (default: 0.10)')
    refine_group.add_argument('--min-range-frac', type=float, default=0.20,
                              help='Minimum range as fraction of original (default: 0.20)')

    # CV configuration
    cv_config = parser.add_argument_group('CV Configuration')
    cv_config.add_argument('--n-folds', type=int, default=5, help='CV folds (default: 5)')
    cv_config.add_argument('--gap', type=int, default=20, help='Embargo gap days (default: 20)')

    # Objective configuration
    obj_group = parser.add_argument_group('Objective')
    obj_group.add_argument('--objective', type=str, default='composite',
                           choices=['composite', 'auc'], help='Objective mode (default: composite)')
    obj_group.add_argument('--variance-penalty', type=float, default=1.0,
                           help='Variance penalty for AUC mode (default: 1.0)')

    # Resource configuration
    resource_group = parser.add_argument_group('Resources')
    resource_group.add_argument('--threads-per-fold', type=int, default=None,
                                help='LightGBM threads per fold (default: auto)')

    args = parser.parse_args()

    # Parse complex arguments
    prune_folds = parse_prune_folds(args.prune_folds)
    eval_all_folds = args.eval_all_folds.lower() == 'true'
    refine_quantiles = tuple(float(x) for x in args.refine_quantiles.split(','))

    if args.all_models:
        run_all_models(
            n_trials=args.n_trials,
            timeout=args.timeout,
            threads_per_fold=args.threads_per_fold,
            n_folds=args.n_folds,
            gap=args.gap,
            objective_mode=args.objective,
            variance_penalty=args.variance_penalty,
            rounds=args.rounds,
            trials_per_round=args.trials_per_round,
            prune_folds=prune_folds,
            prune_metric=args.prune_metric,
            cv_score_method=args.cv_score,
            refine_method=args.refine_method,
        )
    else:
        # Parse model key
        model_key = None
        if args.model:
            model_key = ModelKey(args.model)

        run_hyperopt(
            model_key=model_key,
            n_trials=args.n_trials,
            timeout=args.timeout,
            threads_per_fold=args.threads_per_fold,
            n_folds=args.n_folds,
            gap=args.gap,
            objective_mode=args.objective,
            variance_penalty=args.variance_penalty,
            resume=args.resume,
            rounds=args.rounds,
            trials_per_round=args.trials_per_round,
            prune_folds=prune_folds,
            prune_metric=args.prune_metric,
            prune_margin=args.prune_margin,
            min_completed_trials=args.prune_min_completed,
            eval_all_folds=eval_all_folds,
            cv_score_method=args.cv_score,
            refine_method=args.refine_method,
            elite_frac=args.elite_frac,
            refine_quantiles=refine_quantiles,
            refine_padding=args.refine_padding,
            min_range_frac=args.min_range_frac,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
