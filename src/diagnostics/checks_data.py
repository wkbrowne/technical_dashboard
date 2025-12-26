"""
CV leakage detection checks.

Model-training specific checks:
- Date leakage (CV overlap)
- Label leakage (suspiciously high AUC)
- Symbol exposure analysis

Note: General data quality checks (NaN rates, constant features, duplicates)
are handled by run_data_quality.py and checks_data_quality.py
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .core import (
    DiagnosticFlag, DiagnosticResult, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def check_cv_date_leakage(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    dates: pd.DatetimeIndex,
    gap: int = 20,
) -> List[DiagnosticFlag]:
    """
    Check that CV splits have proper embargo and no date overlap.

    Args:
        cv_splits: List of (train_idx, test_idx) tuples
        dates: DatetimeIndex for all samples
        gap: Expected embargo gap in days

    Returns:
        List of diagnostic flags
    """
    flags = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]

        train_max = train_dates.max()
        test_min = test_dates.min()

        # Check for overlap
        overlap = (train_dates.max() >= test_dates.min())
        if overlap:
            flags.append(DiagnosticFlag(
                severity=Severity.CRITICAL,
                check_name='cv_date_overlap',
                symptom=f"Fold {fold_idx+1}: Training dates overlap with test dates",
                why_it_matters="Date overlap causes severe data leakage and inflated metrics",
                suggested_fix="Fix CV split generation to ensure train_end < test_start - gap",
                evidence={
                    'fold': fold_idx + 1,
                    'train_max': str(train_max.date()),
                    'test_min': str(test_min.date()),
                },
            ))

        # Check embargo gap
        if not overlap:
            actual_gap = (test_min - train_max).days
            if actual_gap < gap:
                flags.append(DiagnosticFlag(
                    severity=Severity.WARN,
                    check_name='cv_embargo_too_small',
                    symptom=f"Fold {fold_idx+1}: Embargo gap is {actual_gap} days, expected {gap}",
                    why_it_matters="Insufficient embargo can leak information from overlapping target windows",
                    suggested_fix=f"Increase CV gap parameter to at least {gap} days",
                    evidence={
                        'fold': fold_idx + 1,
                        'actual_gap': actual_gap,
                        'expected_gap': gap,
                    },
                ))

    return flags


def check_label_leakage(
    fold_metrics: List[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for suspiciously high AUC that may indicate label leakage.

    Args:
        fold_metrics: List of per-fold metric dictionaries
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    aucs = [m.get('auc', 0) for m in fold_metrics]
    max_auc = max(aucs) if aucs else 0
    mean_auc = np.mean(aucs) if aucs else 0

    if mean_auc > thresholds.suspiciously_high_auc:
        flags.append(DiagnosticFlag(
            severity=Severity.CRITICAL,
            check_name='label_leakage_suspected',
            symptom=f"Mean AUC of {mean_auc:.4f} is suspiciously high (>{thresholds.suspiciously_high_auc})",
            why_it_matters="Very high AUC often indicates label leakage or data contamination",
            suggested_fix="Audit feature engineering for lookahead bias; check target generation",
            evidence={
                'mean_auc': round(mean_auc, 4),
                'max_auc': round(max_auc, 4),
                'per_fold_aucs': [round(a, 4) for a in aucs],
                'threshold': thresholds.suspiciously_high_auc,
            },
        ))
    elif max_auc > thresholds.suspiciously_high_auc:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='label_leakage_possible',
            symptom=f"At least one fold has AUC > {thresholds.suspiciously_high_auc} ({max_auc:.4f})",
            why_it_matters="Unusually high single-fold AUC may indicate partial leakage",
            suggested_fix="Investigate the specific fold for data issues",
            evidence={
                'max_auc': round(max_auc, 4),
                'mean_auc': round(mean_auc, 4),
                'per_fold_aucs': [round(a, 4) for a in aucs],
            },
        ))

    return flags


def check_symbol_exposure(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    symbols: np.ndarray,
    dates: pd.DatetimeIndex,
) -> List[DiagnosticFlag]:
    """
    Analyze symbol exposure across train/test splits.

    Reports what percentage of test symbols also appear in training.
    For panel data, this is expected but worth monitoring.

    Args:
        cv_splits: List of (train_idx, test_idx) tuples
        symbols: Symbol array for all samples
        dates: DatetimeIndex for all samples

    Returns:
        List of diagnostic flags
    """
    flags = []
    exposure_stats = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_symbols = set(symbols[train_idx])
        test_symbols = set(symbols[test_idx])

        overlap_symbols = train_symbols & test_symbols
        overlap_rate = len(overlap_symbols) / len(test_symbols) if test_symbols else 0

        exposure_stats.append({
            'fold': fold_idx + 1,
            'train_symbols': len(train_symbols),
            'test_symbols': len(test_symbols),
            'overlap_symbols': len(overlap_symbols),
            'overlap_rate': round(overlap_rate * 100, 1),
        })

    mean_overlap = np.mean([s['overlap_rate'] for s in exposure_stats])

    # This is informational - symbol overlap is expected in panel data
    flags.append(DiagnosticFlag(
        severity=Severity.INFO,
        check_name='symbol_exposure',
        symptom=f"Average {mean_overlap:.1f}% of test symbols appear in training data",
        why_it_matters="Symbol overlap is expected in time-series CV but worth monitoring for model generalization",
        suggested_fix="Consider if new symbol generalization is important for your use case",
        evidence={
            'mean_overlap_rate': round(mean_overlap, 1),
            'per_fold_stats': exposure_stats,
        },
    ))

    return flags


def run_leakage_checks(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    fold_metrics: List[Dict[str, float]],
    gap: int = 20,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run CV leakage detection checks.

    Note: General data quality checks (NaN, constant, duplicate features)
    are handled separately by run_data_quality.py

    Args:
        cv_splits: CV splits
        dates: Date index
        symbols: Symbol array
        fold_metrics: Per-fold metrics from CV
        gap: Expected embargo gap
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # CV date leakage
    flags.extend(check_cv_date_leakage(cv_splits, dates, gap))

    # Label leakage
    flags.extend(check_label_leakage(fold_metrics, thresholds))

    # Symbol exposure
    flags.extend(check_symbol_exposure(cv_splits, symbols, dates))

    # Summary
    summary = {
        'n_cv_folds': len(cv_splits),
        'n_unique_symbols': len(np.unique(symbols)),
        'date_range': {
            'start': str(dates.min().date()),
            'end': str(dates.max().date()),
        },
    }

    return flags, summary
