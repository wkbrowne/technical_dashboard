"""
CV quality and stability checks.

Checks:
- Fold-level metrics (AUC, AUPR, Brier, Precision@10, Spread@10)
- AUC variance / coefficient of variation
- Trend in performance over time
- Effective training size per fold
- Positive rate per fold / drift
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from .core import (
    DiagnosticFlag, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def check_auc_stability(
    fold_metrics: List[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for AUC instability across folds.

    Args:
        fold_metrics: List of per-fold metric dictionaries
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    aucs = [m.get('auc', 0) for m in fold_metrics]
    if not aucs:
        return flags

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    cv = std_auc / (mean_auc + 1e-8)  # Coefficient of variation

    if cv > thresholds.auc_cv_critical:
        flags.append(DiagnosticFlag(
            severity=Severity.CRITICAL,
            check_name='auc_instability_critical',
            symptom=f"AUC coefficient of variation is {cv:.3f} (>{thresholds.auc_cv_critical})",
            why_it_matters="Extreme AUC instability indicates model unreliability across time periods",
            suggested_fix="Increase regularization; check for regime-specific data issues; consider larger training windows",
            evidence={
                'auc_mean': round(mean_auc, 4),
                'auc_std': round(std_auc, 4),
                'auc_cv': round(cv, 4),
                'per_fold_aucs': [round(a, 4) for a in aucs],
            },
        ))
    elif cv > thresholds.auc_cv_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='auc_instability_warn',
            symptom=f"AUC coefficient of variation is {cv:.3f} (>{thresholds.auc_cv_warn})",
            why_it_matters="High AUC variance suggests model performance varies significantly across periods",
            suggested_fix="Consider stronger regularization or more stable features",
            evidence={
                'auc_mean': round(mean_auc, 4),
                'auc_std': round(std_auc, 4),
                'auc_cv': round(cv, 4),
                'per_fold_aucs': [round(a, 4) for a in aucs],
            },
        ))

    return flags


def check_performance_degradation(
    fold_metrics: List[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check if model performance degrades over time (later folds worse than early).

    Args:
        fold_metrics: List of per-fold metric dictionaries (chronological order)
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    aucs = [m.get('auc', 0) for m in fold_metrics]
    if len(aucs) < 2:
        return flags

    # Compare first half average to second half average
    mid = len(aucs) // 2
    first_half_avg = np.mean(aucs[:mid])
    second_half_avg = np.mean(aucs[mid:])

    degradation = first_half_avg - second_half_avg

    if degradation > thresholds.performance_degradation_threshold:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='performance_degradation',
            symptom=f"Model performance degrades over time: early AUC {first_half_avg:.4f} vs late {second_half_avg:.4f}",
            why_it_matters="Declining performance suggests model may not generalize to recent data or market regime change",
            suggested_fix="Consider retraining more frequently; investigate if features are becoming stale; check for concept drift",
            evidence={
                'first_half_auc': round(first_half_avg, 4),
                'second_half_auc': round(second_half_avg, 4),
                'degradation': round(degradation, 4),
                'per_fold_aucs': [round(a, 4) for a in aucs],
            },
        ))

    # Also check for monotonic decline
    if len(aucs) >= 3:
        declines = sum(1 for i in range(1, len(aucs)) if aucs[i] < aucs[i-1])
        decline_rate = declines / (len(aucs) - 1)

        if decline_rate > 0.7 and degradation > thresholds.performance_degradation_threshold / 2:
            flags.append(DiagnosticFlag(
                severity=Severity.WARN,
                check_name='monotonic_decline',
                symptom=f"AUC shows monotonic decline: {decline_rate*100:.0f}% of folds worse than previous",
                why_it_matters="Consistent decline suggests systematic model degradation",
                suggested_fix="Investigate data quality over time; consider adaptive retraining strategy",
                evidence={
                    'decline_rate': round(decline_rate, 2),
                    'per_fold_aucs': [round(a, 4) for a in aucs],
                },
            ))

    return flags


def check_fold_sizes(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check that each fold has sufficient training samples.

    Args:
        cv_splits: List of (train_idx, test_idx) tuples
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []
    small_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        n_train = len(train_idx)
        n_test = len(test_idx)

        if n_train < thresholds.min_train_samples_warn:
            small_folds.append({
                'fold': fold_idx + 1,
                'n_train': n_train,
                'n_test': n_test,
            })

    if small_folds:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='small_training_folds',
            symptom=f"{len(small_folds)} fold(s) have fewer than {thresholds.min_train_samples_warn} training samples",
            why_it_matters="Small training sets can lead to overfitting and unreliable metric estimates",
            suggested_fix="Reduce number of CV folds or increase data coverage",
            evidence={
                'small_folds': small_folds,
                'threshold': thresholds.min_train_samples_warn,
            },
        ))

    return flags


def check_positive_rate_drift(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for significant drift in positive rate across folds.

    Args:
        cv_splits: List of (train_idx, test_idx) tuples
        y: Target array
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []
    fold_pos_rates = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_pos_rate = y[train_idx].mean()
        test_pos_rate = y[test_idx].mean()

        fold_pos_rates.append({
            'fold': fold_idx + 1,
            'train_pos_rate': round(train_pos_rate * 100, 2),
            'test_pos_rate': round(test_pos_rate * 100, 2),
        })

    test_rates = [f['test_pos_rate'] for f in fold_pos_rates]
    train_rates = [f['train_pos_rate'] for f in fold_pos_rates]

    if test_rates:
        max_rate = max(test_rates)
        min_rate = min(test_rates)
        ratio = max_rate / (min_rate + 0.01)  # Avoid division by zero

        if ratio > thresholds.positive_rate_drift_ratio:
            flags.append(DiagnosticFlag(
                severity=Severity.WARN,
                check_name='positive_rate_drift',
                symptom=f"Test positive rate varies from {min_rate:.1f}% to {max_rate:.1f}% (ratio {ratio:.2f})",
                why_it_matters="Large positive rate drift indicates changing market conditions or data quality issues",
                suggested_fix="Consider stratifying CV by positive rate; investigate time periods with extreme rates",
                evidence={
                    'min_test_pos_rate': min_rate,
                    'max_test_pos_rate': max_rate,
                    'ratio': round(ratio, 2),
                    'per_fold_rates': fold_pos_rates,
                },
            ))

    # Also report train/test distribution shift
    for fold_data in fold_pos_rates:
        train_test_diff = abs(fold_data['train_pos_rate'] - fold_data['test_pos_rate'])
        if train_test_diff > 10:  # More than 10 percentage points
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='train_test_pos_rate_shift',
                symptom=f"Fold {fold_data['fold']}: Train/test positive rate differs by {train_test_diff:.1f} pp",
                why_it_matters="Large train/test distribution shift may affect model calibration",
                suggested_fix="Monitor but typically acceptable in time-series CV",
                evidence={
                    'fold': fold_data['fold'],
                    'train_pos_rate': fold_data['train_pos_rate'],
                    'test_pos_rate': fold_data['test_pos_rate'],
                    'difference': round(train_test_diff, 2),
                },
            ))
            break  # Only report once

    return flags


def compute_fold_metrics_summary(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compute summary statistics for fold metrics.

    Args:
        fold_metrics: List of per-fold metric dictionaries

    Returns:
        Summary dictionary
    """
    if not fold_metrics:
        return {}

    summary = {}
    metric_names = ['auc', 'aupr', 'brier', 'precision_top_10', 'precision_bottom_10']

    for metric in metric_names:
        values = [m.get(metric, np.nan) for m in fold_metrics]
        valid_values = [v for v in values if not np.isnan(v)]

        if valid_values:
            summary[metric] = {
                'mean': round(np.mean(valid_values), 4),
                'std': round(np.std(valid_values), 4),
                'min': round(np.min(valid_values), 4),
                'max': round(np.max(valid_values), 4),
                'per_fold': [round(v, 4) if not np.isnan(v) else None for v in values],
            }

    # Compute spread
    if 'precision_top_10' in summary and 'precision_bottom_10' in summary:
        spreads = []
        for m in fold_metrics:
            top = m.get('precision_top_10', np.nan)
            bottom = m.get('precision_bottom_10', np.nan)
            if not np.isnan(top) and not np.isnan(bottom):
                spreads.append(top - bottom)

        if spreads:
            summary['spread_top_bottom_10'] = {
                'mean': round(np.mean(spreads), 4),
                'std': round(np.std(spreads), 4),
                'per_fold': [round(s, 4) for s in spreads],
            }

    return summary


def run_cv_checks(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    fold_metrics: List[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all CV quality and stability checks.

    Args:
        cv_splits: CV splits
        y: Target array
        fold_metrics: Per-fold metrics from CV
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # AUC stability
    flags.extend(check_auc_stability(fold_metrics, thresholds))

    # Performance degradation
    flags.extend(check_performance_degradation(fold_metrics, thresholds))

    # Fold sizes
    flags.extend(check_fold_sizes(cv_splits, thresholds))

    # Positive rate drift
    flags.extend(check_positive_rate_drift(cv_splits, y, thresholds))

    # Summary
    summary = {
        'n_folds': len(cv_splits),
        'fold_sizes': [
            {
                'fold': i + 1,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
            }
            for i, (train_idx, test_idx) in enumerate(cv_splits)
        ],
        'metrics': compute_fold_metrics_summary(fold_metrics),
        'overall_positive_rate': round(y.mean() * 100, 2),
    }

    return flags, summary
