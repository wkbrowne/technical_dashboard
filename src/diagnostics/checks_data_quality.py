"""
Data quality checks for the diagnostic framework.

Checks:
- Missing feature rate (NaN/inf) per feature
- Constant/near-constant features (low variance)
- Duplicate features (high correlation)
- Feature coverage vs BASE_FEATURES
- Category-level NaN analysis
- Infinite values detection

Note: Uses numpy for correlation (not pandas) for performance.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import numpy as np
import pandas as pd

from .core import (
    DiagnosticFlag, DiagnosticResult, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def check_missing_features(
    X: np.ndarray,
    feature_names: List[str],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, float]]:
    """
    Check for features with high missing (NaN/inf) rates.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, nan_rates_dict)
    """
    flags = []
    nan_rates = {}
    high_nan_features = []
    critical_nan_features = []

    for i, feat_name in enumerate(feature_names):
        col = X[:, i]

        # Count NaN and inf
        nan_count = np.sum(np.isnan(col))
        inf_count = np.sum(np.isinf(col))
        missing_count = nan_count + inf_count
        missing_rate = missing_count / len(col)
        nan_rates[feat_name] = missing_rate

        if missing_rate > thresholds.nan_rate_critical:
            critical_nan_features.append({
                'feature': feat_name,
                'missing_rate': round(missing_rate * 100, 2),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
            })
        elif missing_rate > thresholds.nan_rate_warn:
            high_nan_features.append({
                'feature': feat_name,
                'missing_rate': round(missing_rate * 100, 2),
            })

    # Report as aggregated flags
    if critical_nan_features:
        flags.append(DiagnosticFlag(
            severity=Severity.CRITICAL,
            check_name='missing_features_critical',
            symptom=f"{len(critical_nan_features)} feature(s) have >{thresholds.nan_rate_critical*100:.0f}% missing values",
            why_it_matters="High missing rate can cause model instability and biased predictions",
            suggested_fix="Investigate data pipeline; check feature computation for these features",
            evidence={
                'features': critical_nan_features[:10],
                'total_critical': len(critical_nan_features),
                'threshold': thresholds.nan_rate_critical * 100,
            },
        ))

    if high_nan_features:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='missing_features_warn',
            symptom=f"{len(high_nan_features)} feature(s) have >{thresholds.nan_rate_warn*100:.0f}% missing values",
            why_it_matters="Moderate missing rate may affect model quality",
            suggested_fix="Review data pipeline; ensure proper interpolation",
            evidence={
                'features': high_nan_features[:10],
                'total_warn': len(high_nan_features),
                'threshold': thresholds.nan_rate_warn * 100,
            },
        ))

    return flags, nan_rates


def check_constant_features(
    X: np.ndarray,
    feature_names: List[str],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for constant or near-constant features (low variance).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []
    constant_features = []

    for i, feat_name in enumerate(feature_names):
        col = X[:, i]
        # Use nanvar to handle NaNs
        variance = np.nanvar(col)

        if variance < thresholds.constant_variance_threshold:
            constant_features.append({
                'feature': feat_name,
                'variance': float(variance),
                'unique_values': len(np.unique(col[~np.isnan(col)])),
            })

    if constant_features:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='constant_features',
            symptom=f"{len(constant_features)} feature(s) have near-zero variance",
            why_it_matters="Constant features provide no information and waste model capacity",
            suggested_fix="Remove constant features from the feature set",
            evidence={
                'constant_features': constant_features[:10],
                'total_constant': len(constant_features),
            },
        ))

    return flags


def check_duplicate_features(
    X: np.ndarray,
    feature_names: List[str],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
    sample_size: int = 10000,
) -> List[DiagnosticFlag]:
    """
    Check for duplicate features (very high correlation).

    Uses numpy for correlation computation (much faster than pandas).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        thresholds: Diagnostic thresholds
        sample_size: Sample size for correlation computation (for speed)

    Returns:
        List of diagnostic flags
    """
    flags = []
    n_samples, n_features = X.shape

    if n_features < 2:
        return flags

    # Sample for speed if needed
    if n_samples > sample_size:
        idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Clean data for correlation
    X_clean = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

    # Identify valid columns (non-zero variance)
    stds = np.std(X_clean, axis=0)
    valid_mask = stds > 1e-10

    if valid_mask.sum() < 2:
        return flags

    # Compute correlation matrix using numpy (fast)
    X_valid = X_clean[:, valid_mask]
    valid_feature_names = [f for f, v in zip(feature_names, valid_mask) if v]

    # Standardize for correlation
    X_centered = X_valid - X_valid.mean(axis=0)
    X_normed = X_centered / (np.std(X_centered, axis=0) + 1e-10)

    # Compute correlation matrix: (X.T @ X) / n
    corr_matrix = (X_normed.T @ X_normed) / X_normed.shape[0]

    # Find duplicates (upper triangle only)
    duplicates = []
    n_valid = len(valid_feature_names)

    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            corr = corr_matrix[i, j]
            if abs(corr) >= thresholds.duplicate_correlation_threshold:
                duplicates.append({
                    'feature_1': valid_feature_names[i],
                    'feature_2': valid_feature_names[j],
                    'correlation': round(float(corr), 6),
                })

    if duplicates:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='duplicate_features',
            symptom=f"{len(duplicates)} feature pair(s) have correlation >= {thresholds.duplicate_correlation_threshold}",
            why_it_matters="Duplicate features add redundancy and can cause multicollinearity issues",
            suggested_fix="Remove one feature from each highly correlated pair",
            evidence={
                'duplicate_pairs': duplicates[:10],
                'total_duplicates': len(duplicates),
            },
        ))

    return flags


def check_infinite_values(
    X: np.ndarray,
    feature_names: List[str],
) -> List[DiagnosticFlag]:
    """
    Check for infinite values in features.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        List of diagnostic flags
    """
    flags = []
    inf_features = []

    for i, feat_name in enumerate(feature_names):
        col = X[:, i]
        inf_count = np.sum(np.isinf(col))

        if inf_count > 0:
            inf_features.append({
                'feature': feat_name,
                'inf_count': int(inf_count),
                'inf_rate': round(inf_count / len(col) * 100, 4),
            })

    if inf_features:
        flags.append(DiagnosticFlag(
            severity=Severity.CRITICAL,
            check_name='infinite_values',
            symptom=f"{len(inf_features)} feature(s) contain infinite values",
            why_it_matters="Infinite values indicate data corruption or division by zero in feature computation",
            suggested_fix="Fix the feature computation pipeline to handle edge cases",
            evidence={
                'features': inf_features[:10],
                'total_with_inf': len(inf_features),
            },
        ))

    return flags


def check_feature_coverage(
    available_features: List[str],
    expected_features: List[str],
    feature_set_name: str = "BASE_FEATURES",
) -> List[DiagnosticFlag]:
    """
    Check coverage of expected features.

    Args:
        available_features: Features present in the data
        expected_features: Features that should be present
        feature_set_name: Name of the feature set for reporting

    Returns:
        List of diagnostic flags
    """
    flags = []

    available_set = set(available_features)
    expected_set = set(expected_features)

    missing = expected_set - available_set
    coverage = (len(expected_set) - len(missing)) / len(expected_set) * 100 if expected_set else 100

    if missing:
        severity = Severity.CRITICAL if coverage < 80 else Severity.WARN
        flags.append(DiagnosticFlag(
            severity=severity,
            check_name=f'missing_{feature_set_name.lower()}',
            symptom=f"{len(missing)}/{len(expected_set)} {feature_set_name} are missing ({coverage:.1f}% coverage)",
            why_it_matters=f"Missing features from {feature_set_name} may degrade model performance",
            suggested_fix="Re-run feature pipeline or check feature computation",
            evidence={
                'missing_features': sorted(list(missing))[:20],
                'total_missing': len(missing),
                'total_expected': len(expected_set),
                'coverage_pct': round(coverage, 1),
            },
        ))

    return flags


def compute_category_nan_rates(
    df: pd.DataFrame,
    feature_categories: Dict[str, Dict],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute NaN rates by feature category.

    Args:
        df: DataFrame with features
        feature_categories: Category definitions with patterns

    Returns:
        Dictionary of category -> stats
    """
    category_stats = {}

    for cat_name, cat_info in feature_categories.items():
        patterns = cat_info.get("patterns", [])
        matching = [c for c in df.columns if any(p in c.lower() for p in patterns)
                    and df[c].dtype in [np.float32, np.float64]]

        if matching:
            nan_rates = df[matching].isna().mean() * 100
            category_stats[cat_name] = {
                "features": matching,
                "count": len(matching),
                "nan_mean": round(nan_rates.mean(), 2),
                "nan_max": round(nan_rates.max(), 2),
                "nan_min": round(nan_rates.min(), 2),
                "expected_nan": cat_info.get("expected_nan", (0, 50)),
                "healthy": sum(1 for c in matching if df[c].isna().mean() * 100 < 50),
                "broken": [c for c in matching if df[c].isna().mean() * 100 >= 90],
            }

    return category_stats


def run_data_quality_checks(
    X: np.ndarray,
    feature_names: List[str],
    expected_features: Optional[List[str]] = None,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all data quality checks.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        expected_features: Expected feature list for coverage check
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # Missing features
    missing_flags, nan_rates = check_missing_features(X, feature_names, thresholds)
    flags.extend(missing_flags)

    # Infinite values
    flags.extend(check_infinite_values(X, feature_names))

    # Constant features
    flags.extend(check_constant_features(X, feature_names, thresholds))

    # Duplicate features
    flags.extend(check_duplicate_features(X, feature_names, thresholds))

    # Feature coverage
    if expected_features:
        flags.extend(check_feature_coverage(feature_names, expected_features))

    # Summary
    high_nan_count = sum(1 for r in nan_rates.values() if r > thresholds.nan_rate_warn)
    critical_nan_count = sum(1 for r in nan_rates.values() if r > thresholds.nan_rate_critical)

    summary = {
        'n_features': len(feature_names),
        'n_samples': X.shape[0],
        'high_nan_features': high_nan_count,
        'critical_nan_features': critical_nan_count,
        'nan_rates': {k: round(v * 100, 2) for k, v in sorted(nan_rates.items(), key=lambda x: -x[1])[:20]},
    }

    if expected_features:
        available_set = set(feature_names)
        expected_set = set(expected_features)
        summary['feature_coverage'] = {
            'expected': len(expected_set),
            'available': len(available_set & expected_set),
            'missing': len(expected_set - available_set),
            'coverage_pct': round(len(available_set & expected_set) / len(expected_set) * 100, 1) if expected_set else 100,
        }

    return flags, summary
