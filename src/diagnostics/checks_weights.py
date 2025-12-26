"""
Sample weighting sanity checks.

Checks:
- Weight distribution (min/median/max, concentration)
- Effective sample size (ESS)
- Metric sensitivity to weights
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .core import (
    DiagnosticFlag, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size from sample weights.

    ESS = (sum(w))^2 / sum(w^2)

    Args:
        weights: Sample weight array

    Returns:
        Effective sample size
    """
    sum_w = weights.sum()
    sum_w_sq = (weights ** 2).sum()
    return (sum_w ** 2) / (sum_w_sq + 1e-10)


def check_weight_distribution(
    sample_weight: np.ndarray,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check weight distribution for concentration issues.

    Args:
        sample_weight: Sample weight array
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if sample_weight is None or len(sample_weight) == 0:
        return flags

    n_samples = len(sample_weight)
    total_weight = sample_weight.sum()

    # Normalize weights for analysis
    norm_weights = sample_weight / (total_weight + 1e-10)

    # Top 1% concentration
    sorted_weights = np.sort(norm_weights)[::-1]
    top_1_pct_count = max(1, int(n_samples * 0.01))
    top_1_pct_share = sorted_weights[:top_1_pct_count].sum()

    if top_1_pct_share > thresholds.weight_top_1pct_share_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='weight_concentration',
            symptom=f"Top 1% of samples hold {top_1_pct_share*100:.1f}% of total weight",
            why_it_matters="Highly concentrated weights mean model is effectively trained on fewer samples",
            suggested_fix="Review weight calculation; consider clipping extreme weights",
            evidence={
                'top_1pct_share': round(top_1_pct_share * 100, 2),
                'top_1pct_count': top_1_pct_count,
                'threshold': thresholds.weight_top_1pct_share_warn * 100,
            },
        ))

    # Check for very small weights (effectively zero)
    min_weight = sample_weight.min()
    small_weight_count = (sample_weight < 0.01 * sample_weight.mean()).sum()
    small_weight_rate = small_weight_count / n_samples

    if small_weight_rate > 0.10:
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='many_small_weights',
            symptom=f"{small_weight_rate*100:.1f}% of samples have weight < 1% of mean",
            why_it_matters="Many tiny weights reduce effective training diversity",
            suggested_fix="Check if overlap weighting is too aggressive",
            evidence={
                'small_weight_rate': round(small_weight_rate * 100, 2),
                'small_weight_count': int(small_weight_count),
                'min_weight': round(min_weight, 6),
            },
        ))

    return flags


def check_effective_sample_size(
    sample_weight: np.ndarray,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check that effective sample size is reasonable.

    Args:
        sample_weight: Sample weight array
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if sample_weight is None or len(sample_weight) == 0:
        return flags

    n_samples = len(sample_weight)
    ess = compute_effective_sample_size(sample_weight)
    ess_ratio = ess / n_samples

    if ess_ratio < thresholds.ess_ratio_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='low_effective_sample_size',
            symptom=f"ESS is {ess:.0f} ({ess_ratio*100:.1f}% of {n_samples} samples)",
            why_it_matters="Low ESS means weighting is aggressive and model may underfit",
            suggested_fix="Review overlap weighting parameters; consider less aggressive down-weighting",
            evidence={
                'ess': round(ess, 0),
                'n_samples': n_samples,
                'ess_ratio': round(ess_ratio * 100, 2),
                'threshold': thresholds.ess_ratio_warn * 100,
            },
        ))

    return flags


def check_weight_metric_sensitivity(
    fold_metrics_weighted: List[Dict[str, float]],
    fold_metrics_unweighted: Optional[List[Dict[str, float]]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Compare metrics with and without sample weights.

    Args:
        fold_metrics_weighted: Metrics computed with sample weights
        fold_metrics_unweighted: Metrics computed without sample weights (optional)
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if fold_metrics_unweighted is None or len(fold_metrics_unweighted) == 0:
        return flags

    # Compare AUC
    aucs_weighted = [m.get('auc', np.nan) for m in fold_metrics_weighted]
    aucs_unweighted = [m.get('auc', np.nan) for m in fold_metrics_unweighted]

    valid_w = [a for a in aucs_weighted if not np.isnan(a)]
    valid_uw = [a for a in aucs_unweighted if not np.isnan(a)]

    if valid_w and valid_uw:
        auc_diff = abs(np.mean(valid_w) - np.mean(valid_uw))

        if auc_diff > thresholds.weight_metric_sensitivity:
            flags.append(DiagnosticFlag(
                severity=Severity.INFO,
                check_name='weight_sensitive_auc',
                symptom=f"AUC differs by {auc_diff:.4f} with vs without weights",
                why_it_matters="Large difference suggests weighting significantly impacts model evaluation",
                suggested_fix="Understand which samples are down-weighted and why; both metrics may be valid",
                evidence={
                    'auc_weighted': round(np.mean(valid_w), 4),
                    'auc_unweighted': round(np.mean(valid_uw), 4),
                    'auc_diff': round(auc_diff, 4),
                },
            ))

    return flags


def compute_weight_summary(sample_weight: np.ndarray) -> Dict[str, Any]:
    """
    Compute summary statistics for sample weights.

    Args:
        sample_weight: Sample weight array

    Returns:
        Summary dictionary
    """
    if sample_weight is None or len(sample_weight) == 0:
        return {'weights_available': False}

    n_samples = len(sample_weight)
    ess = compute_effective_sample_size(sample_weight)

    return {
        'weights_available': True,
        'n_samples': n_samples,
        'ess': round(ess, 0),
        'ess_ratio': round(ess / n_samples * 100, 2),
        'weight_stats': {
            'min': round(sample_weight.min(), 6),
            'max': round(sample_weight.max(), 6),
            'mean': round(sample_weight.mean(), 6),
            'median': round(np.median(sample_weight), 6),
            'std': round(sample_weight.std(), 6),
        },
        'weight_percentiles': {
            '1': round(np.percentile(sample_weight, 1), 6),
            '5': round(np.percentile(sample_weight, 5), 6),
            '25': round(np.percentile(sample_weight, 25), 6),
            '50': round(np.percentile(sample_weight, 50), 6),
            '75': round(np.percentile(sample_weight, 75), 6),
            '95': round(np.percentile(sample_weight, 95), 6),
            '99': round(np.percentile(sample_weight, 99), 6),
        },
    }


def run_weight_checks(
    sample_weight: Optional[np.ndarray],
    fold_metrics_weighted: List[Dict[str, float]],
    fold_metrics_unweighted: Optional[List[Dict[str, float]]] = None,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all weighting sanity checks.

    Args:
        sample_weight: Sample weight array (can be None)
        fold_metrics_weighted: Metrics computed with sample weights
        fold_metrics_unweighted: Metrics without weights (optional, for sensitivity check)
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    if sample_weight is None:
        summary = {'weights_available': False}
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='no_sample_weights',
            symptom="No sample weights provided",
            why_it_matters="Without overlap weighting, overlapping trajectories may bias training",
            suggested_fix="Consider enabling sample weights from triple barrier overlap inverse weighting",
            evidence={},
        ))
        return flags, summary

    # Weight distribution
    flags.extend(check_weight_distribution(sample_weight, thresholds))

    # Effective sample size
    flags.extend(check_effective_sample_size(sample_weight, thresholds))

    # Metric sensitivity
    flags.extend(check_weight_metric_sensitivity(
        fold_metrics_weighted, fold_metrics_unweighted, thresholds
    ))

    # Summary
    summary = compute_weight_summary(sample_weight)

    return flags, summary
