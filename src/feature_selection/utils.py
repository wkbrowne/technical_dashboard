"""Utility functions for feature selection.

This module provides helper functions for memory management,
data validation, and other common operations.
"""

import gc
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback without psutil
        return 0.0


def estimate_dataframe_memory_mb(df: pd.DataFrame) -> float:
    """Estimate memory usage of a DataFrame in MB.

    Args:
        df: DataFrame to estimate.

    Returns:
        Estimated memory in megabytes.
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    gc.collect()


def validate_data(
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = 100
) -> Tuple[bool, List[str]]:
    """Validate input data for feature selection.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        min_samples: Minimum required samples.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    # Check alignment
    if len(X) != len(y):
        issues.append(f"X and y have different lengths ({len(X)} vs {len(y)})")

    # Check minimum samples
    if len(X) < min_samples:
        issues.append(f"Too few samples ({len(X)} < {min_samples})")

    # Check for all-NaN columns
    nan_cols = X.columns[X.isna().all()].tolist()
    if nan_cols:
        issues.append(f"All-NaN columns found: {nan_cols[:5]}...")

    # Check for constant columns
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols[:5]}...")

    # Check target
    if y.isna().all():
        issues.append("Target is all NaN")

    # For classification, check class balance
    unique_values = y.dropna().unique()
    if len(unique_values) == 2:
        min_class_pct = y.value_counts(normalize=True).min()
        if min_class_pct < 0.05:
            issues.append(f"Severe class imbalance (minority class: {min_class_pct:.1%})")

    return len(issues) == 0, issues


def filter_features_by_variance(
    X: pd.DataFrame,
    threshold: float = 0.0
) -> List[str]:
    """Filter features with variance above threshold.

    Args:
        X: Feature DataFrame.
        threshold: Minimum variance threshold.

    Returns:
        List of feature names with sufficient variance.
    """
    variances = X.var()
    return variances[variances > threshold].index.tolist()


def filter_features_by_nan_rate(
    X: pd.DataFrame,
    max_nan_rate: float = 0.5
) -> List[str]:
    """Filter features with acceptable NaN rates.

    Args:
        X: Feature DataFrame.
        max_nan_rate: Maximum acceptable NaN rate.

    Returns:
        List of feature names with acceptable NaN rates.
    """
    nan_rates = X.isna().mean()
    return nan_rates[nan_rates <= max_nan_rate].index.tolist()


def filter_correlated_features(
    X: pd.DataFrame,
    features: List[str],
    correlation_threshold: float = 0.95
) -> List[str]:
    """Remove highly correlated features, keeping the first in each pair.

    Args:
        X: Feature DataFrame.
        features: List of features to analyze.
        correlation_threshold: Correlation threshold for removal.

    Returns:
        List of features with highly correlated duplicates removed.
    """
    valid_features = [f for f in features if f in X.columns]
    if len(valid_features) < 2:
        return valid_features

    corr_matrix = X[valid_features].corr().abs()

    # Set diagonal to 0
    np.fill_diagonal(corr_matrix.values, 0)

    # Find pairs above threshold
    to_remove = set()
    for i, feat_i in enumerate(valid_features):
        if feat_i in to_remove:
            continue
        for j, feat_j in enumerate(valid_features):
            if i >= j or feat_j in to_remove:
                continue
            if corr_matrix.loc[feat_i, feat_j] > correlation_threshold:
                to_remove.add(feat_j)

    return [f for f in valid_features if f not in to_remove]


def compute_information_coefficient(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """Compute Information Coefficient (Spearman correlation).

    Args:
        predictions: Model predictions.
        actuals: Actual values.

    Returns:
        Spearman correlation coefficient.
    """
    from scipy import stats

    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 3:
        return 0.0

    corr, _ = stats.spearmanr(predictions[mask], actuals[mask])
    return corr if not np.isnan(corr) else 0.0


def split_features_by_prefix(
    features: List[str]
) -> Dict[str, List[str]]:
    """Split features by timeframe prefix.

    Args:
        features: List of feature names.

    Returns:
        Dict mapping prefix to feature list.
    """
    result = {
        'daily': [],
        'weekly': [],
        'monthly': [],
        'other': []
    }

    for f in features:
        if f.startswith('w_'):
            result['weekly'].append(f)
        elif f.startswith('m_'):
            result['monthly'].append(f)
        elif f.startswith('d_'):
            result['daily'].append(f)
        else:
            # Assume daily if no prefix
            result['daily'].append(f)

    return result


def categorize_features(
    features: List[str]
) -> Dict[str, List[str]]:
    """Categorize features by type based on naming conventions.

    Args:
        features: List of feature names.

    Returns:
        Dict mapping category to feature list.
    """
    categories = {
        'momentum': [],
        'volatility': [],
        'volume': [],
        'trend': [],
        'breadth': [],
        'cross_sectional': [],
        'regime': [],
        'interaction': [],
        'other': []
    }

    patterns = {
        'momentum': ['rsi', 'macd', 'return', 'mom', 'roc'],
        'volatility': ['vol', 'atr', 'vix', 'std'],
        'volume': ['volume', 'vol_ratio', 'obv'],
        'trend': ['ma_', 'slope', 'trend', 'sma', 'ema'],
        'breadth': ['breadth', 'advance', 'decline', 'high_low'],
        'cross_sectional': ['rank', 'pct', 'xsec', 'relative', 'alpha'],
        'regime': ['regime', 'state', 'phase'],
        'interaction': ['_x_', '_AND_', '_div_'],
    }

    for f in features:
        f_lower = f.lower()
        categorized = False

        for category, keywords in patterns.items():
            if any(kw in f_lower for kw in keywords):
                categories[category].append(f)
                categorized = True
                break

        if not categorized:
            categories['other'].append(f)

    return categories


def print_feature_summary(features: List[str], title: str = "Feature Summary"):
    """Print a summary of selected features.

    Args:
        features: List of feature names.
        title: Title for the summary.
    """
    print(f"\n{'='*60}")
    print(f"{title} ({len(features)} features)")
    print('='*60)

    # By timeframe
    by_timeframe = split_features_by_prefix(features)
    print("\nBy Timeframe:")
    for tf, feats in by_timeframe.items():
        if feats:
            print(f"  {tf}: {len(feats)} features")

    # By category
    by_category = categorize_features(features)
    print("\nBy Category:")
    for cat, feats in by_category.items():
        if feats:
            print(f"  {cat}: {len(feats)} features")

    # List interaction features
    interactions = [f for f in features if '_x_' in f or '_AND_' in f]
    if interactions:
        print(f"\nInteraction Features ({len(interactions)}):")
        for f in interactions[:10]:  # Show first 10
            print(f"  {f}")
        if len(interactions) > 10:
            print(f"  ... and {len(interactions) - 10} more")

    print('='*60)


def estimate_compute_time(
    n_features: int,
    n_samples: int,
    n_folds: int,
    eval_time_per_1k_samples: float = 0.5
) -> float:
    """Estimate compute time for feature selection.

    Args:
        n_features: Number of candidate features.
        n_samples: Number of samples.
        n_folds: Number of CV folds.
        eval_time_per_1k_samples: Estimated time per 1000 samples per fold.

    Returns:
        Estimated total time in seconds.
    """
    # Rough estimates for each phase
    time_per_eval = eval_time_per_1k_samples * (n_samples / 1000) * n_folds

    # Initial ranking: 1 eval
    initial_time = time_per_eval

    # Forward selection: ~n_features * n_iterations (roughly n_features/2)
    forward_time = n_features * (n_features / 2) * time_per_eval * 0.5

    # Backward elimination: ~selected_features evaluations
    backward_time = (n_features / 3) * time_per_eval

    # Interactions: discovery + forward selection
    interaction_time = time_per_eval * 2 + (n_features / 4) * time_per_eval

    # Swapping: varies widely
    swap_time = (n_features / 5) * time_per_eval * 10

    total = initial_time + forward_time + backward_time + interaction_time + swap_time

    return total
