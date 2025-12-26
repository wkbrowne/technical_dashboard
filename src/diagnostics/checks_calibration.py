"""
Calibration and ranking sanity checks.

Checks:
- Reliability curve / calibration summary
- Brier score sanity relative to base rate
- Probability distribution (predictions near 0/1)
- Precision@K curve
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .core import (
    DiagnosticFlag, Severity,
    DiagnosticThresholds, DEFAULT_THRESHOLDS,
)


def compute_reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute reliability curve (calibration curve) statistics.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Dictionary with calibration statistics
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_true_freqs = []
    bin_pred_means = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_true_freqs.append(y_true[mask].mean())
            bin_pred_means.append(y_pred[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_true_freqs.append(np.nan)
            bin_pred_means.append(np.nan)
            bin_counts.append(0)

    # Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = sum(bin_counts)
    for i in range(n_bins):
        if bin_counts[i] > 0 and not np.isnan(bin_true_freqs[i]):
            ece += (bin_counts[i] / total_samples) * abs(bin_true_freqs[i] - bin_pred_means[i])

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0 and not np.isnan(bin_true_freqs[i]):
            mce = max(mce, abs(bin_true_freqs[i] - bin_pred_means[i]))

    return {
        'bin_centers': [round(c, 2) for c in bin_centers],
        'bin_true_freqs': [round(f, 4) if not np.isnan(f) else None for f in bin_true_freqs],
        'bin_pred_means': [round(m, 4) if not np.isnan(m) else None for m in bin_pred_means],
        'bin_counts': [int(c) for c in bin_counts],
        'ece': round(ece, 4),
        'mce': round(mce, 4),
    }


def check_brier_sanity(
    fold_metrics: List[Dict[str, float]],
    base_rate: float,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check if Brier score is reasonable relative to the baseline.

    Baseline Brier = base_rate * (1 - base_rate) for a constant predictor.

    Args:
        fold_metrics: List of per-fold metric dictionaries
        base_rate: Overall positive rate
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    briers = [m.get('brier', np.nan) for m in fold_metrics]
    valid_briers = [b for b in briers if not np.isnan(b)]

    if not valid_briers:
        return flags

    mean_brier = np.mean(valid_briers)
    baseline_brier = base_rate * (1 - base_rate)

    # Model should beat baseline
    brier_ratio = mean_brier / (baseline_brier + 1e-8)

    if brier_ratio > thresholds.brier_baseline_multiplier_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='brier_worse_than_baseline',
            symptom=f"Mean Brier {mean_brier:.4f} is worse than baseline {baseline_brier:.4f} (ratio {brier_ratio:.2f})",
            why_it_matters="Model calibration is worse than predicting the base rate for all samples",
            suggested_fix="Review probability calibration; consider Platt scaling or isotonic regression",
            evidence={
                'mean_brier': round(mean_brier, 4),
                'baseline_brier': round(baseline_brier, 4),
                'brier_ratio': round(brier_ratio, 2),
                'base_rate': round(base_rate, 4),
                'per_fold_briers': [round(b, 4) if not np.isnan(b) else None for b in briers],
            },
        ))

    return flags


def check_prediction_distribution(
    all_predictions: np.ndarray,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check for extreme predictions (near 0 or 1).

    Args:
        all_predictions: Array of predicted probabilities from all folds
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if len(all_predictions) == 0:
        return flags

    # Count predictions outside [0.05, 0.95]
    extreme_low = (all_predictions < 0.05).sum()
    extreme_high = (all_predictions > 0.95).sum()
    total_extreme = extreme_low + extreme_high
    extreme_rate = total_extreme / len(all_predictions)

    # Distribution statistics
    pred_mean = all_predictions.mean()
    pred_std = all_predictions.std()
    pred_min = all_predictions.min()
    pred_max = all_predictions.max()

    if extreme_rate > thresholds.extreme_pred_warn:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='extreme_predictions',
            symptom=f"{extreme_rate*100:.1f}% of predictions are outside [0.05, 0.95]",
            why_it_matters="Extreme predictions suggest overconfidence and poor calibration",
            suggested_fix="Increase regularization; consider probability calibration post-hoc",
            evidence={
                'extreme_rate': round(extreme_rate * 100, 2),
                'n_below_0.05': int(extreme_low),
                'n_above_0.95': int(extreme_high),
                'pred_mean': round(pred_mean, 4),
                'pred_std': round(pred_std, 4),
                'pred_range': [round(pred_min, 4), round(pred_max, 4)],
            },
        ))

    # Check for predictions all concentrated in narrow range
    if pred_std < 0.05:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='narrow_prediction_range',
            symptom=f"Prediction std is only {pred_std:.4f} - predictions highly concentrated",
            why_it_matters="Very narrow prediction range suggests model is not differentiating between samples",
            suggested_fix="Check model is learning; may need stronger signal features or different architecture",
            evidence={
                'pred_mean': round(pred_mean, 4),
                'pred_std': round(pred_std, 4),
                'pred_range': [round(pred_min, 4), round(pred_max, 4)],
            },
        ))

    return flags


def compute_precision_at_k_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_percentiles: List[float] = [0.01, 0.05, 0.10, 0.20],
) -> Dict[str, Any]:
    """
    Compute precision at various K percentiles.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        k_percentiles: List of percentiles to compute (0.01 = top 1%)

    Returns:
        Dictionary with precision at each K
    """
    n = len(y_pred)
    sorted_idx = np.argsort(y_pred)[::-1]  # Descending order

    results = {}
    for k_pct in k_percentiles:
        k = max(1, int(n * k_pct))
        top_k_idx = sorted_idx[:k]
        precision = y_true[top_k_idx].mean()

        results[f'precision_at_{int(k_pct*100)}pct'] = {
            'k': k,
            'precision': round(precision, 4),
        }

    return results


def check_precision_at_k(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    base_rate: float,
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> List[DiagnosticFlag]:
    """
    Check that precision@K provides reasonable lift over base rate.

    Args:
        all_y_true: True labels from all folds
        all_y_pred: Predictions from all folds
        base_rate: Overall positive rate
        thresholds: Diagnostic thresholds

    Returns:
        List of diagnostic flags
    """
    flags = []

    if len(all_y_pred) == 0:
        return flags

    # Compute precision at 5%
    n = len(all_y_pred)
    k = max(1, int(n * 0.05))
    sorted_idx = np.argsort(all_y_pred)[::-1]
    top_k_idx = sorted_idx[:k]
    precision_at_5 = all_y_true[top_k_idx].mean()

    # Check lift
    lift = precision_at_5 / (base_rate + 1e-8)

    if lift < thresholds.precision_at_k_min_lift:
        flags.append(DiagnosticFlag(
            severity=Severity.WARN,
            check_name='low_precision_lift',
            symptom=f"Precision@5% ({precision_at_5:.4f}) provides only {lift:.2f}x lift over base rate ({base_rate:.4f})",
            why_it_matters="Model is not effectively ranking samples; top predictions not much better than random",
            suggested_fix="Improve feature engineering; consider different model architecture; check for data issues",
            evidence={
                'precision_at_5pct': round(precision_at_5, 4),
                'base_rate': round(base_rate, 4),
                'lift': round(lift, 2),
                'threshold_lift': thresholds.precision_at_k_min_lift,
            },
        ))

    # Also check bottom 5% (should be worse than base rate)
    bottom_k_idx = sorted_idx[-k:]
    precision_bottom = all_y_true[bottom_k_idx].mean()

    if precision_bottom > base_rate * 0.9:
        flags.append(DiagnosticFlag(
            severity=Severity.INFO,
            check_name='bottom_not_worse',
            symptom=f"Bottom 5% precision ({precision_bottom:.4f}) is close to base rate ({base_rate:.4f})",
            why_it_matters="Model is not effectively identifying low-probability samples",
            suggested_fix="Consider whether short-side prediction is working properly",
            evidence={
                'precision_bottom_5pct': round(precision_bottom, 4),
                'base_rate': round(base_rate, 4),
            },
        ))

    return flags


def run_calibration_checks(
    fold_predictions: List[Tuple[np.ndarray, np.ndarray]],
    base_rate: float,
    fold_metrics: List[Dict[str, float]],
    thresholds: DiagnosticThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[List[DiagnosticFlag], Dict[str, Any]]:
    """
    Run all calibration and ranking sanity checks.

    Args:
        fold_predictions: List of (y_true, y_pred) tuples from each fold
        base_rate: Overall positive rate
        fold_metrics: Per-fold metrics
        thresholds: Diagnostic thresholds

    Returns:
        Tuple of (flags, summary_dict)
    """
    flags = []

    # Combine all predictions
    all_y_true = np.concatenate([y for y, _ in fold_predictions])
    all_y_pred = np.concatenate([p for _, p in fold_predictions])

    # Brier sanity
    flags.extend(check_brier_sanity(fold_metrics, base_rate, thresholds))

    # Prediction distribution
    flags.extend(check_prediction_distribution(all_y_pred, thresholds))

    # Precision at K
    flags.extend(check_precision_at_k(all_y_true, all_y_pred, base_rate, thresholds))

    # Summary
    reliability = compute_reliability_curve(all_y_true, all_y_pred)
    precision_curve = compute_precision_at_k_curve(all_y_true, all_y_pred)

    summary = {
        'reliability_curve': reliability,
        'precision_at_k': precision_curve,
        'prediction_distribution': {
            'mean': round(all_y_pred.mean(), 4),
            'std': round(all_y_pred.std(), 4),
            'min': round(all_y_pred.min(), 4),
            'max': round(all_y_pred.max(), 4),
            'percentiles': {
                '5': round(np.percentile(all_y_pred, 5), 4),
                '25': round(np.percentile(all_y_pred, 25), 4),
                '50': round(np.percentile(all_y_pred, 50), 4),
                '75': round(np.percentile(all_y_pred, 75), 4),
                '95': round(np.percentile(all_y_pred, 95), 4),
            },
        },
        'base_rate': round(base_rate, 4),
    }

    return flags, summary
